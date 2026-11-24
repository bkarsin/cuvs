/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../cuda_event.hpp"
#include "common.cuh"
#include "direct_edge_prune.cuh"
#include "hash_prune.cuh"
#include "leaf.cuh"
#include "partition.cuh"
#include "segmented_topk.cuh"

#include <algorithm>
#include <cstddef>
#include <cuvs/neighbors/gpipnn.hpp>

#include <raft/core/device_mdspan.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/resource/device_id.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/dry_run_flag.hpp>
#include <raft/core/resources.hpp>
#include <raft/matrix/init.cuh>
#include <raft/util/cuda_rt_essentials.hpp>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/kernel_launch.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <pthread.h>

#include <algorithm>
#include <condition_variable>
#include <exception>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <semaphore>
#include <type_traits>
#include <utility>
#include <vector>

namespace cuvs::neighbors::experimental::gpipnn::detail {

using cuvs::neighbors::detail::cuda_event;

inline std::mutex log_mutex;

template <typename... Args>
void log_message(Args&&... args)
{
  std::lock_guard<std::mutex> lock(log_mutex);
  std::cout << std::dec;
  (std::cout << ... << std::forward<Args>(args));
  std::cout << std::endl;
}

/**
 * Leaf members one range may carry so the candidate buffer stays inside its budget.
 *
 * A member emits `leaf_degree` selected neighbours, each becoming a forward and a reverse record
 * of `sizeof(key) + sizeof(target)` bytes, and the radix sort needs a second buffer of the same
 * size. Never returns less than 1, so a single oversized leaf still makes progress.
 */
inline auto leaf_batch_max_members(const index_params& params, int64_t n_rows, int64_t leaf_degree)
  -> int64_t
{
  // Candidate buffer budget per row when `index_params::leaf_batch_mib` is left on auto, with a
  // floor and ceiling on the result: one batch for small builds, a bounded reservation for large.
  constexpr int64_t kAutoLeafBatchBytesPerRow = 256;
  constexpr int64_t kMinLeafBatchBytes        = int64_t{512} << 20;
  constexpr int64_t kMaxLeafBatchBytes        = int64_t{6} << 30;

  const int64_t budget =
    params.leaf_batch_mib > 0
      ? static_cast<int64_t>(params.leaf_batch_mib) << 20
      : std::clamp(n_rows * kAutoLeafBatchBytesPerRow, kMinLeafBatchBytes, kMaxLeafBatchBytes);
  const int64_t per_member =
    leaf_degree * 2 * static_cast<int64_t>(sizeof(uint64_t) + sizeof(id_t)) * 2;
  return std::max<int64_t>(1, budget / per_member);
}

class buffer_slot {
public:
  buffer_slot(const buffer_slot&) = delete;
  auto operator=(const buffer_slot&) -> buffer_slot& = delete;
  auto operator=(buffer_slot&&) noexcept -> buffer_slot& = delete;

  virtual ~buffer_slot() = default;

  [[nodiscard]] virtual auto data() noexcept -> void* = 0;
  [[nodiscard]] virtual auto size() const noexcept -> std::size_t = 0;

protected:
  buffer_slot() = default;
  buffer_slot(buffer_slot&&) noexcept = default;
};

// A buffer slot or call to get_slot outliving the cpu buffer is undefined behavior
class cpu_buffer {
public:
  class cpu_buffer_slot final : public buffer_slot {
    friend class cpu_buffer;

  public:
    cpu_buffer_slot(const cpu_buffer_slot&) = delete;
    auto operator=(const cpu_buffer_slot&) -> cpu_buffer_slot& = delete;

    cpu_buffer_slot(cpu_buffer_slot&& other) noexcept
      : cpu_buffer_(other.cpu_buffer_),
        data_(std::move(other.data_)),
        leased_(std::exchange(other.leased_, false))
    {
    }
    auto operator=(cpu_buffer_slot&&) -> cpu_buffer_slot& = delete;

    ~cpu_buffer_slot() override
    {
      if (!leased_) { return; }
      leased_ = false;
      {
        std::lock_guard<std::mutex> lock(cpu_buffer_->buffer_mtx_);
        cpu_buffer_->available_slots_.push_back(std::move(*this));
      }
      cpu_buffer_->num_available_slots_.release();
    }

    [[nodiscard]] auto data() noexcept -> void* override
    {
      return data_.data();
    }

    [[nodiscard]] auto size() const noexcept -> std::size_t override
    {
      return data_.size();
    }

  private:
    explicit cpu_buffer_slot(cpu_buffer* buffer, std::size_t buffer_size)
      : cpu_buffer_(buffer), data_(buffer_size)
    {
    }

    cpu_buffer* const cpu_buffer_;
    std::vector<uint8_t> data_;
    bool leased_{false};
  };

  explicit cpu_buffer(const std::ptrdiff_t num_slots, const std::size_t buffer_size)
    : num_available_slots_{validate_num_slots(num_slots)}
  {
    available_slots_.reserve(static_cast<std::size_t>(num_slots));
    for (std::ptrdiff_t slot = 0; slot < num_slots; ++slot) {
      available_slots_.push_back(cpu_buffer_slot{this, buffer_size});
    }
  }

  [[nodiscard]] auto get_slot() -> cpu_buffer_slot
  {
    num_available_slots_.acquire();
    std::lock_guard<std::mutex> lock(buffer_mtx_);
    auto slot = std::move(available_slots_.back());
    available_slots_.pop_back();
    slot.leased_ = true;
    return slot;
  }

private:
  static auto validate_num_slots(std::ptrdiff_t num_slots) -> std::ptrdiff_t
  {
    RAFT_EXPECTS(num_slots > 0, "cpu_buffer requires at least one slot");
    RAFT_EXPECTS(num_slots <= std::counting_semaphore<>::max(),
                  "cpu_buffer slot count exceeds the semaphore maximum");
    return num_slots;
  }

  std::vector<cpu_buffer_slot> available_slots_;
  std::counting_semaphore<> num_available_slots_; // threads wait till a slot is available
  std::mutex buffer_mtx_; // mutex to assign distinct slots to requesting threads

};

// must outlive all jobs that reference it
class compute_job_synchronizer {
  friend class cpu_job;
  friend class gpu_job;

public:
  void synchronize()
  {
    std::unique_lock<std::mutex> lock(mutex_);
    empty_.wait(lock, [this] { return num_outstanding_jobs_ == 0; });
  }

private:
  void record_start()
  {
    std::lock_guard<std::mutex> lock(mutex_);
    ++num_outstanding_jobs_;
  }

  void record_end()
  {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      RAFT_EXPECTS(num_outstanding_jobs_ > 0, "Compute job synchronizer count underflow");
      --num_outstanding_jobs_;
      if (num_outstanding_jobs_ != 0) { return; }
    }
    empty_.notify_all();
  }

  std::mutex mutex_;
  std::condition_variable empty_;
  std::size_t num_outstanding_jobs_{0};
};

class cpu_job {
public:
  cpu_job(const cpu_job&) = delete;
  auto operator=(const cpu_job&) -> cpu_job& = delete;
  cpu_job(cpu_job&&) = delete;
  auto operator=(cpu_job&&) -> cpu_job& = delete;

  virtual ~cpu_job() { synchronizer_.record_end(); }
  void execute() noexcept
  {
    try {
      execute_impl();
    } catch (const std::exception& error) {
      std::cerr << "CPU job failed: " << error.what() << std::endl;
      std::terminate();
    } catch (...) {
      std::cerr << "CPU job failed with an unknown exception" << std::endl;
      std::terminate();
    }
  }

protected:
  explicit cpu_job(compute_job_synchronizer& synchronizer) : synchronizer_{synchronizer}
  {
    synchronizer_.record_start();
  }
  virtual void execute_impl() = 0;

private:
  compute_job_synchronizer& synchronizer_;
};

class gpu_job {
public:
  gpu_job(const gpu_job&) = delete;
  auto operator=(const gpu_job&) -> gpu_job& = delete;
  gpu_job(gpu_job&&) = delete;
  auto operator=(gpu_job&&) -> gpu_job& = delete;

  virtual ~gpu_job() { synchronizer_.record_end(); }
  void execute(cuda::stream_ref h2d_copy_stream,
               cuda::stream_ref compute_stream,
               cuda::stream_ref d2h_copy_stream) noexcept
  {
    try {
      execute_impl(h2d_copy_stream, compute_stream, d2h_copy_stream);
    } catch (const std::exception& error) {
      std::cerr << "GPU job failed: " << error.what() << std::endl;
      std::terminate();
    } catch (...) {
      std::cerr << "GPU job failed with an unknown exception" << std::endl;
      std::terminate();
    }
  }

protected:
  explicit gpu_job(compute_job_synchronizer& synchronizer)
    : synchronizer_{synchronizer}
  {
    synchronizer_.record_start();
  }

  virtual void execute_impl(cuda::stream_ref h2d_copy_stream,
                            cuda::stream_ref compute_stream,
                            cuda::stream_ref d2h_copy_stream) = 0;

private:
  compute_job_synchronizer& synchronizer_;
};

// queue must outlive all workers
template <typename JobT>
class work_queue {
public:
  [[nodiscard]] auto pop_job() -> std::unique_ptr<JobT>
  {
    std::unique_lock<std::mutex> lock(work_queue_mtx_);
    work_queue_ready_.wait(lock, [this] { return closed_ || !work_queue_.empty(); });
    if (work_queue_.empty()) {
      // queue must be closed and all remaining jobs were processed already
      return nullptr;
    }

    auto job = std::move(work_queue_.front());
    work_queue_.pop();
    return job;
  }

  void push_job(std::unique_ptr<JobT> job)
  {
    RAFT_EXPECTS(job != nullptr, "Cannot push a null job");
    {
      std::lock_guard<std::mutex> lock(work_queue_mtx_);
      RAFT_EXPECTS(!closed_, "Cannot push a job after the work queue is closed");
      work_queue_.push(std::move(job));
    }
    work_queue_ready_.notify_one();
  }

  void close()
  {
    {
      std::lock_guard<std::mutex> lock(work_queue_mtx_);
      if (closed_) { return; }
      closed_ = true;
    }
    work_queue_ready_.notify_all();
  }

private:
  std::queue<std::unique_ptr<JobT>> work_queue_;
  std::mutex work_queue_mtx_;
  std::condition_variable work_queue_ready_;
  bool closed_{false};
};

using cpu_work_queue = work_queue<cpu_job>;
using gpu_work_queue = work_queue<gpu_job>;

class gpipnn_host_manager {
public:
  gpipnn_host_manager(std::ptrdiff_t num_workers, cpu_work_queue& cpu_work_queue)
    : num_workers_{validate_num_workers(num_workers)},
      cpu_work_queue_{cpu_work_queue},
      workers_(num_workers_),
      worker_contexts_(num_workers_)
  {
    for (std::size_t worker_id = 0; worker_id < num_workers_; ++worker_id) {
      worker_contexts_[worker_id] =
        worker_context{.manager = this, .worker_id = worker_id};
    }

    const int result = pthread_create(&manager_thread_,
                                      nullptr,
                                      &gpipnn_host_manager::manager_start_routine,
                                      this);
    if (result != 0) {
      std::cerr << "pthread_create failed for gpipnn_host_manager thread with error " << result
                << std::endl;
      std::terminate();
    }
  }

  gpipnn_host_manager(const gpipnn_host_manager&) = delete;
  auto operator=(const gpipnn_host_manager&) -> gpipnn_host_manager& = delete;
  gpipnn_host_manager(gpipnn_host_manager&&) = delete;
  auto operator=(gpipnn_host_manager&&) -> gpipnn_host_manager& = delete;

  ~gpipnn_host_manager() noexcept
  {
    cpu_work_queue_.close();

    const int result = pthread_join(manager_thread_, nullptr);
    if (result != 0) {
      std::cerr << "Failed to join gpipnn_host_manager thread with error " << result << std::endl;
      std::terminate();
    }
  }

  void push_job(std::unique_ptr<cpu_job> job)
  {
    cpu_work_queue_.push_job(std::move(job));
  }

private:
  struct worker_context {
    gpipnn_host_manager* manager;
    std::size_t worker_id;
  };

  class cpu_worker {
  public:
    explicit cpu_worker(cpu_work_queue& cpu_work_queue) : cpu_work_queue_{cpu_work_queue} {}

    void run_worker(const worker_context& context)
    {
      const auto worker_id = context.worker_id;
      while (true) {
        auto job = cpu_work_queue_.pop_job();
        if (job == nullptr) {
          log_message("CPU worker ", worker_id, " exiting after queue close");
          return;
        }
        job->execute();
      }
    }

  private:
    cpu_work_queue& cpu_work_queue_;
  };

  static auto validate_num_workers(std::ptrdiff_t num_workers) -> std::size_t
  {
    RAFT_EXPECTS(num_workers > 0, "num_workers must be at least 1");
    return static_cast<std::size_t>(num_workers);
  }

  static auto manager_start_routine(void* argument) noexcept -> void*
  {
    static_cast<gpipnn_host_manager*>(argument)->run_manager();
    return nullptr;
  }

  void run_manager() noexcept
  {
    for (std::size_t worker = 0; worker < num_workers_; ++worker) {
      const int result = pthread_create(&workers_[worker],
                                        nullptr,
                                        &gpipnn_host_manager::worker_start_routine,
                                        &worker_contexts_[worker]);
      if (result != 0) {
        std::cerr << "CPU worker creation failed with error " << result << std::endl;
        std::terminate();
      }
    }
    for (std::size_t worker = 0; worker < num_workers_; ++worker) {
      const int result = pthread_join(workers_[worker], nullptr);
      if (result != 0) {
        std::cerr << "CPU worker join failed with error " << result << std::endl;
        std::terminate();
      }
    }
    workers_.clear();
  }

  static auto worker_start_routine(void* argument) noexcept -> void*
  {
    auto& context = *static_cast<worker_context*>(argument);
    try {
      cpu_worker worker{context.manager->cpu_work_queue_};
      worker.run_worker(context);
    } catch (const std::exception& error) {
      std::cerr << "CPU worker " << context.worker_id
                << " exiting after exception: " << error.what() << std::endl;
      std::terminate();
    } catch (...) {
      std::cerr << "CPU worker " << context.worker_id << " exiting after unknown exception"
                << std::endl;
      std::terminate();
    }
    return nullptr;
  }

  const std::size_t num_workers_;
  cpu_work_queue& cpu_work_queue_;
  pthread_t manager_thread_{};
  std::vector<pthread_t> workers_;
  std::vector<worker_context> worker_contexts_;
};

// device memory buffer for single GPU worker (max pipelined cuda streams is effectively number of slots)
class gpu_buffer {
public:
  class gpu_buffer_slot final : public buffer_slot {
    friend class gpu_buffer;

  public:
    gpu_buffer_slot(const gpu_buffer_slot&) = delete;
    auto operator=(const gpu_buffer_slot&) -> gpu_buffer_slot& = delete;
    gpu_buffer_slot(gpu_buffer_slot&& other) noexcept
      : data_{std::move(other.data_)}, event_{std::move(other.event_)}
    {
    }
    auto operator=(gpu_buffer_slot&&) noexcept -> gpu_buffer_slot& = delete;

    ~gpu_buffer_slot() override = default;

    [[nodiscard]] auto data() noexcept -> void* override { return data_.data(); }
    [[nodiscard]] auto size() const noexcept -> std::size_t override { return data_.size(); }

    void event_record(cuda::stream_ref stream)
    {
      RAFT_CUDA_TRY(cudaEventRecord(event_.value(), stream.get()));
    }

  private:
    gpu_buffer_slot(std::size_t buffer_size, cuda::stream_ref allocation_stream)
      : data_{buffer_size, allocation_stream}
    {
      RAFT_CUDA_TRY(cudaEventRecord(event_.value(), allocation_stream.get()));
    }

    rmm::device_buffer data_;
    cuda_event event_;
  };

  gpu_buffer(const std::ptrdiff_t num_slots,
             const std::size_t buffer_size,
             cuda::stream_ref allocation_stream)
  {
    RAFT_EXPECTS(num_slots > 0, "gpu_buffer requires at least one slot");
    slots_.reserve(static_cast<std::size_t>(num_slots));
    for (std::ptrdiff_t slot = 0; slot < num_slots; ++slot) {
      slots_.push_back(gpu_buffer_slot{buffer_size, allocation_stream});
    }
  }

  ~gpu_buffer()
  {
    for (const auto& slot : slots_) {
      RAFT_CUDA_TRY_NO_THROW(cudaEventSynchronize(slot.event_.value()));
    }
  }

  [[nodiscard]] auto get_slot() -> gpu_buffer_slot&
  {
    auto& slot = slots_[curr_slot_];
    RAFT_CUDA_TRY(
      cudaEventSynchronize(slot.event_.value())); // synchronizes allocation on first call
    curr_slot_ = (curr_slot_ + 1) % slots_.size();
    return slot;
  }

private:
  std::vector<gpu_buffer_slot> slots_;
  std::size_t curr_slot_{0}; // The current slot that is staged for return in get_slot()
};

class gpipnn_dev_manager {
public:
  gpipnn_dev_manager(raft::resources const& res, gpu_work_queue& gpu_work_queue)
    : worker_context_{
        .res              = res,
        .gpu_work_queue   = gpu_work_queue}
  {
    const int result =
      pthread_create(
        &gpu_worker_, nullptr, &gpipnn_dev_manager::worker_start_routine, &worker_context_);
    if (result != 0) {
      std::cerr << "pthread_create failed for gpipnn_dev_manager with error " << result
                << std::endl;
      std::terminate();
    }
  }

  gpipnn_dev_manager(const gpipnn_dev_manager&) = delete;
  auto operator=(const gpipnn_dev_manager&) -> gpipnn_dev_manager& = delete;
  gpipnn_dev_manager(gpipnn_dev_manager&&) = delete;
  auto operator=(gpipnn_dev_manager&&) -> gpipnn_dev_manager& = delete;

  ~gpipnn_dev_manager() noexcept
  {
    worker_context_.gpu_work_queue.close();

    const int result = pthread_join(gpu_worker_, nullptr);
    if (result != 0) {
      std::cerr << "Failed to join gpipnn_dev_manager thread with error " << result << std::endl;
      std::terminate();
    }
  }

private:
  struct worker_context {
    raft::resources const& res;
    gpu_work_queue& gpu_work_queue;
  };

  class gpu_worker {
  public:
    explicit gpu_worker(raft::resources const& res)
    {
      try {
        RAFT_CUDA_TRY(cudaSetDevice(raft::resource::get_device_id(res)));
        RAFT_CUDA_TRY(cudaStreamCreateWithFlags(&h2d_stream_handle_, cudaStreamNonBlocking));
        RAFT_CUDA_TRY(cudaStreamCreateWithFlags(&compute_stream_handle_, cudaStreamNonBlocking));
        RAFT_CUDA_TRY(cudaStreamCreateWithFlags(&d2h_stream_handle_, cudaStreamNonBlocking));
        h2d_stream_     = cuda::stream_ref{h2d_stream_handle_};
        compute_stream_ = cuda::stream_ref{compute_stream_handle_};
        d2h_stream_     = cuda::stream_ref{d2h_stream_handle_};
      } catch (...) {
        destroy_streams();
        throw;
      }
    }

    gpu_worker(const gpu_worker&) = delete;
    auto operator=(const gpu_worker&) -> gpu_worker& = delete;
    gpu_worker(gpu_worker&&) = delete;
    auto operator=(gpu_worker&&) -> gpu_worker& = delete;

    ~gpu_worker() noexcept { destroy_streams(); }

    void run_worker(const worker_context& context)
    {
      while (true) {
        auto job = context.gpu_work_queue.pop_job();
        if (job == nullptr) {
          log_message("GPU worker exiting after queue close");
          return;
        }
        job->execute(h2d_stream_, compute_stream_, d2h_stream_);
      }
    }

  private:
    void destroy_streams() noexcept
    {
      if (d2h_stream_handle_ != nullptr) {
        RAFT_CUDA_TRY_NO_THROW(cudaStreamDestroy(d2h_stream_handle_));
        d2h_stream_handle_ = nullptr;
      }
      if (compute_stream_handle_ != nullptr) {
        RAFT_CUDA_TRY_NO_THROW(cudaStreamDestroy(compute_stream_handle_));
        compute_stream_handle_ = nullptr;
      }
      if (h2d_stream_handle_ != nullptr) {
        RAFT_CUDA_TRY_NO_THROW(cudaStreamDestroy(h2d_stream_handle_));
        h2d_stream_handle_ = nullptr;
      }
      h2d_stream_     = cuda::stream_ref{cudaStream_t{nullptr}};
      compute_stream_ = cuda::stream_ref{cudaStream_t{nullptr}};
      d2h_stream_     = cuda::stream_ref{cudaStream_t{nullptr}};
    }

    cudaStream_t h2d_stream_handle_{nullptr};
    cudaStream_t compute_stream_handle_{nullptr};
    cudaStream_t d2h_stream_handle_{nullptr};
    cuda::stream_ref h2d_stream_{cudaStream_t{nullptr}};
    cuda::stream_ref compute_stream_{cudaStream_t{nullptr}};
    cuda::stream_ref d2h_stream_{cudaStream_t{nullptr}};
  };

  static auto worker_start_routine(void* argument) noexcept -> void*
  {
    auto& context = *static_cast<worker_context*>(argument);
    try {
      gpu_worker worker{context.res};
      worker.run_worker(context);
    } catch (const std::exception& error) {
      std::cerr << "GPU worker initialization failed: " << error.what() << std::endl;
      std::terminate();
    } catch (...) {
      std::cerr << "GPU worker initialization failed with an unknown exception" << std::endl;
      std::terminate();
    }
    return nullptr;
  }

  worker_context worker_context_;
  pthread_t gpu_worker_{};
};

class hash_prune_merge_cpu_job : public cpu_job {
  public:
    hash_prune_merge_cpu_job(cuda_event event,
                             cpu_buffer::cpu_buffer_slot buffer_slot,
                             compute_job_synchronizer& synchronizer)
      : cpu_job{synchronizer},
        event_{std::move(event)},
        buffer_slot_{std::move(buffer_slot)}
    {
    }
  
    void execute_impl() override
    {
      RAFT_CUDA_TRY(cudaEventSynchronize(event_.value()));
      log_message("hash_prune_merge");
    }
  
  private:
    cuda_event event_;
    cpu_buffer::cpu_buffer_slot buffer_slot_;
  };

class test_cpu_job final : public cpu_job {
public:
  test_cpu_job(compute_job_synchronizer& synchronizer,
               std::size_t job_id,
               std::size_t dependency_id,
               cpu_buffer::cpu_buffer_slot buffer_slot,
               cuda_event dependency_event)
    : cpu_job{synchronizer},
      job_id_{job_id},
      dependency_id_{dependency_id},
      buffer_slot_{std::move(buffer_slot)},
      dependency_event_{std::move(dependency_event)}
  {
  }

private:
  void execute_impl() override
  {
    RAFT_CUDA_TRY(cudaEventSynchronize(dependency_event_.value()));
    log_message("Executing test CPU job ",
                job_id_,
                " after dependency ",
                dependency_id_,
                " with buffer ",
                buffer_slot_.data());
  }

  std::size_t job_id_;
  std::size_t dependency_id_;
  cpu_buffer::cpu_buffer_slot buffer_slot_;
  cuda_event dependency_event_;
};

class test_gpu_job final : public gpu_job {
public:
  test_gpu_job(compute_job_synchronizer& synchronizer,
               std::size_t job_id,
               gpu_buffer::gpu_buffer_slot& gpu_buffer_slot,
               cpu_buffer::cpu_buffer_slot cpu_buffer_slot,
               cpu_work_queue& cpu_work_queue)
    : gpu_job{synchronizer},
      synchronizer_{synchronizer},
      job_id_{job_id},
      gpu_buffer_slot_{gpu_buffer_slot},
      cpu_buffer_slot_{std::move(cpu_buffer_slot)},
      cpu_work_queue_{cpu_work_queue}
  {
  }

private:
  void execute_impl(cuda::stream_ref h2d_copy_stream,
                    cuda::stream_ref compute_stream,
                    cuda::stream_ref d2h_copy_stream) override
  {
    log_message("Executing test GPU job ",
                job_id_,
                " with GPU buffer ",
                gpu_buffer_slot_.data(),
                " and CPU buffer ",
                cpu_buffer_slot_.data());
    cuda_event completion_event;
    RAFT_CUDA_TRY(cudaEventRecord(completion_event.value(), d2h_copy_stream.get()));
    gpu_buffer_slot_.event_record(d2h_copy_stream);
    auto cpu_job = std::make_unique<test_cpu_job>(
      synchronizer_,
      job_id_ + 1,
      job_id_,
      std::move(cpu_buffer_slot_),
      std::move(completion_event));
    cpu_work_queue_.push_job(std::move(cpu_job));
  }

  compute_job_synchronizer& synchronizer_;
  std::size_t job_id_;
  gpu_buffer::gpu_buffer_slot& gpu_buffer_slot_;
  cpu_buffer::cpu_buffer_slot cpu_buffer_slot_;
  cpu_work_queue& cpu_work_queue_;
};

template <typename DataT>
class gpipnn_main_manager {

public:

  explicit gpipnn_main_manager(const index_params& params)
    : params_(params)
  {
  }

  void build(raft::resources const& res,
    raft::device_matrix_view<const DataT, int64_t, raft::row_major> dataset,
    raft::device_matrix_view<id_t, int64_t, raft::row_major> graph)
  {
    constexpr std::ptrdiff_t kNumWorkers = 5;
    constexpr std::ptrdiff_t kNumCPUBufferSlots = 5; // will be calculated
    constexpr std::ptrdiff_t kNumGPUBufferSlots = 3; // will be calculated
    constexpr std::size_t kCPUBufferSize = std::size_t{1} << 20; // will be calculate
    constexpr std::size_t kGPUBufferSize = std::size_t{1} << 10; // will be calculate

    // define buffers first so they are destroyed last
    auto gpu_alloc_stream =raft::resource::get_cuda_stream(res);
    cpu_buffer host_buffer{kNumCPUBufferSlots, kCPUBufferSize};
    gpu_buffer device_buffer{kNumGPUBufferSlots, kGPUBufferSize, gpu_alloc_stream};

    compute_job_synchronizer job_synchronizer; // used in construction of jobs
    cpu_work_queue cpu_work_queue;
    gpu_work_queue gpu_work_queue;

    gpipnn_host_manager host_manager{kNumWorkers, cpu_work_queue};
    gpipnn_dev_manager dev_manager{res, gpu_work_queue};

    constexpr std::size_t kNumTestJobs = 10;
    for (std::size_t job = 0; job < kNumTestJobs; ++job) {
      const auto gpu_job_id = job * 2;
      auto cpu_buffer_slot = host_buffer.get_slot();
      auto& gpu_buffer_slot = device_buffer.get_slot();
      gpu_work_queue.push_job(std::make_unique<test_gpu_job>(job_synchronizer,
                                                            gpu_job_id,
                                                            gpu_buffer_slot,
                                                            std::move(cpu_buffer_slot),
                                                            cpu_work_queue));
    }

    job_synchronizer.synchronize();
  }

private:
  const index_params& params_;
};

/**
 * Builds a GPiPNN proximity graph.
 *
 * The four phases are strictly sequential, but the third does not wait for the second to finish:
 * the candidates are transient and HashPrune folds them into a fixed-size reservoir and
 * nothing downstream reads them again. So, the leaf join runs a range of leaves at a time and
 * each range's records are merged and released before the next is built. Peak memory is the
 * HashPrune reservoirs plus the batch budget, and batches has no impact on the resulting graph.
 *
 * See cuvs::neighbors::experimental::gpipnn::build_graph.
 */
template <typename DataT>
void build_graph(raft::resources const& res,
                 const index_params& params,
                 raft::device_matrix_view<const DataT, int64_t, raft::row_major> dataset,
                 raft::device_matrix_view<id_t, int64_t, raft::row_major> graph)
{
  static_assert(std::is_same_v<DataT, float> || std::is_same_v<DataT, half> ||
                  std::is_same_v<DataT, int8_t> || std::is_same_v<DataT, uint8_t>,
                "gpipnn: only float, half, int8 and uint8 datasets are supported");

  const auto arch_problem = gpipnn_arch_problem();
  RAFT_EXPECTS(arch_problem.empty(), "%s", arch_problem.c_str());

  const int64_t n_rows = dataset.extent(0);
  const int64_t dim    = dataset.extent(1);
  check_params(params, n_rows, dim, graph);

  gpipnn_main_manager<DataT> manager{params};
  manager.build(res, dataset, graph);

  auto stream = raft::resource::get_cuda_stream(res).get();

  // All scoring runs on half operands with fp32 accumulation and every other type is converted
  // here. A dataset whose magnitudes do not fit half is rescaled by one shared power of two
  // first, which preserves the neighbour ordering; see `half_cast_scale`.
  //
  // Cosine is also handled entirely here by scaling each row to unit norm as it is converted (see
  // `normalize_rows_to_half_kernel`), which bounds every element by 1 and needs no rescaling.
  const bool normalize = params.metric == cuvs::distance::DistanceType::CosineExpanded;
  rmm::device_uvector<half> half_storage(0, stream);
  const half* scoring_data = nullptr;
  if constexpr (std::is_same_v<DataT, half>) {
    if (!normalize) { scoring_data = dataset.data_handle(); }
  }
  if (scoring_data == nullptr) {
    half_storage.resize(n_rows * dim, stream);
    // Non-empty only on the float path below, and only when the peak was asserted rather than
    // measured, which is the one case where it can be wrong. A measured peak cannot overflow by
    // construction, since `half_cast_scale` lands it just under the top of the range.
    rmm::device_uvector<int> overflowed(0, stream);
    if (normalize) {
      raft::launch_kernel(res,
                          n_rows,
                          256,
                          normalize_rows_to_half_kernel<DataT, 256>,
                          dataset.data_handle(),
                          half_storage.data(),
                          n_rows,
                          dim);
    } else {
      // Only float can fall outside the range half represents: int8 and uint8 are bounded by
      // 255, and a half input is representable by definition.
      float scale = 1.0f;
      if constexpr (std::is_same_v<DataT, float>) {
        float max_abs = params.dataset_max_abs;
        if (max_abs > 0.0f) {
          overflowed.resize(1, stream);
          raft::matrix::fill(
            res, raft::make_device_vector_view<int, int64_t>(overflowed.data(), 1), 0);
        } else {
          max_abs = finite_max_abs(res, dataset.data_handle(), n_rows * dim);
        }
        scale = half_cast_scale(max_abs);
        if (scale != 1.0f) {
          RAFT_LOG_DEBUG(
            "gpipnn: dataset magnitudes peak at %g, outside the range half holds usefully; "
            "scaling by %g before conversion, which leaves the neighbour ordering unchanged",
            static_cast<double>(max_abs),
            static_cast<double>(scale));
        }
      }
      cast_to_half(res,
                   dataset.data_handle(),
                   half_storage.data(),
                   n_rows * dim,
                   scale,
                   overflowed.is_empty() ? nullptr : overflowed.data());
    }
    // Reading the flag back is skipped under dry run, where `overflowed` aliases the shared probe
    // buffer and holds nothing. It only decides whether to throw, never a size, so assuming no
    // overflow leaves the measured footprint exactly what it would have been.
    if (!overflowed.is_empty() && !raft::resource::get_dry_run_flag(res)) {
      // Costs a sync, but only on the path that skipped the one `finite_max_abs` does, so
      // asserting the peak still removes a pass over the dataset rather than trading for it.
      int flag = 0;
      raft::update_host(&flag, overflowed.data(), 1, stream);
      raft::resource::sync_stream(res);
      RAFT_EXPECTS(flag == 0,
                   "gpipnn: dataset_max_abs (%g) is below the dataset's actual peak, so the "
                   "conversion to half overflowed; every distance touching those elements would "
                   "be NaN. Raise it, or leave it at 0 to have the build measure it",
                   static_cast<double>(params.dataset_max_abs));
    }
    scoring_data = half_storage.data();
  }
  auto scoring_view =
    raft::make_device_matrix_view<const half, int64_t, raft::row_major>(scoring_data, n_rows, dim);

  rmm::device_uvector<float> norms(n_rows, stream);
  row_norms(scoring_data, n_rows, dim, norms.data(), stream);
  auto norms_view = raft::make_device_vector_view<const float, int64_t>(norms.data(), n_rows);

  carve_stats stats;
  auto leaves = ball_carve(res, scoring_view, norms_view, params, stats);
  RAFT_LOG_DEBUG(
    "gpipnn: %ld leaves over %ld memberships after %ld rounds (%ld chopped, %ld merged, %ld still "
    "oversized)",
    leaves.count(),
    leaves.total(),
    stats.rounds_run,
    stats.chopped_runs,
    stats.merged_runs,
    stats.oversized_left);

  // The paper picks top-2 while we use default of 4 which we found to be better empirically.
  const auto leaf_degree = static_cast<int64_t>(params.leaf_degree);
  const auto hash_bits   = static_cast<int32_t>(params.hash_bits);

  auto sketches = compute_sketches(res, scoring_view, hash_bits, params.seed + 7);
  auto sketches_view =
    raft::make_device_vector_view<const float, int64_t>(sketches.data(), n_rows * hash_bits);
  auto acc = make_accumulator(res, n_rows, params.max_candidates);
  rmm::device_uvector<int64_t> row_begin(n_rows + 1, stream);

  // Leaves are packed into ranges by member count, since that is what the record array is
  // proportional to. A single leaf always forms a range on its own if it has to, so the loop
  // makes progress whatever the budget.
  const int64_t max_members = leaf_batch_max_members(params, n_rows, leaf_degree);

  int64_t n_batches = 0;
  int64_t n_records = 0;
  for (int64_t lo = 0; lo < leaves.count();) {
    int64_t hi      = lo;
    int64_t members = 0;
    while (hi < leaves.count() && (hi == lo || members + leaves.size_of(hi) <= max_members)) {
      members += leaves.size_of(hi);
      hi++;
    }
    auto records = leaf_pick_range(res, scoring_view, norms_view, leaves, leaf_degree, lo, hi);
    n_records += records.n_valid;
    hash_prune_merge(res, sketches_view, records, params, row_begin, acc);
    lo = hi;
    n_batches++;
  }
  RAFT_LOG_DEBUG("gpipnn: %ld candidate edge records over %ld leaf batches (<= %ld members each)",
                 n_records,
                 n_batches,
                 max_members);

  // The prune's row order is the last thing the partition is needed for. See `build_prune_order`
  // for details.
  auto prune_order = build_prune_order(res, leaves, n_rows);

  // Nothing below reads the partition, the sketches, the owner bounds or the per-entry hashes.
  leaves.members.resize(0, stream);
  leaves.members.shrink_to_fit(stream);
  sketches.resize(0, stream);
  sketches.shrink_to_fit(stream);
  row_begin.resize(0, stream);
  row_begin.shrink_to_fit(stream);
  acc.release_hashes(stream);

  direct_edge_prune(
    res,
    acc,
    params,
    graph,
    static_cast<int64_t>(prune_order.size()) == n_rows ? prune_order.data() : nullptr);
}

}  // namespace cuvs::neighbors::experimental::gpipnn::detail
