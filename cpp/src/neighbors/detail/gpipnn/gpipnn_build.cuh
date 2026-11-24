/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "common.cuh"
#include "direct_edge_prune.cuh"
#include "hash_prune.cuh"
#include "leaf.cuh"
#include "partition.cuh"
#include "segmented_topk.cuh"

#include <algorithm>
#include <cuvs/neighbors/gpipnn.hpp>

#include <raft/core/device_mdspan.hpp>
#include <raft/core/logger.hpp>
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

class cpu_job {
public:
  enum class ExitStatus { kSuccess, kFailure };

  virtual ~cpu_job() = default;
  [[nodiscard]] auto execute() noexcept -> ExitStatus
  {
    try {
      return execute_impl();
    } catch (const std::exception& error) {
      std::cerr << "CPU job failed: " << error.what() << std::endl;
    } catch (...) {
      std::cerr << "CPU job failed with an unknown exception" << std::endl;
    }
    return ExitStatus::kFailure;
  }

protected:
  cpu_job() = default;
  [[nodiscard]] virtual auto execute_impl() -> ExitStatus = 0;
};

// cuda_event must remain valid until job execution completes
class hash_prune_merge_job : public cpu_job {
public:
  hash_prune_merge_job(cudaEvent_t cuda_event, cpu_buffer::cpu_buffer_slot buffer_slot)
    : cuda_event_{cuda_event}, buffer_slot_{std::move(buffer_slot)}
  {
  }

  [[nodiscard]] auto execute_impl() -> ExitStatus override
  {
    RAFT_CUDA_TRY(cudaEventSynchronize(cuda_event_));
    std::cout << "hash_prune_merge" << std::endl;
    return ExitStatus::kSuccess;
  }

private:
  cudaEvent_t cuda_event_;
  cpu_buffer::cpu_buffer_slot buffer_slot_;
};

// queue must outlive all workers
class cpu_work_queue {
public:
  [[nodiscard]] auto pop_job() -> std::unique_ptr<cpu_job>
  {
    std::unique_lock<std::mutex> lock(work_queue_mtx_);
    work_queue_ready_.wait(lock, [this] { return closed_ || !work_queue_.empty(); });
    if (work_queue_.empty()) { return nullptr; } // queue must be closed and all remaining jobs were processed already

    auto job = std::move(work_queue_.front());
    work_queue_.pop();
    return job;
  }

  void push_job(std::unique_ptr<cpu_job> job)
  {
    RAFT_EXPECTS(job != nullptr, "Cannot push a null CPU job");
    {
      std::lock_guard<std::mutex> lock(work_queue_mtx_);
      RAFT_EXPECTS(!closed_, "Cannot push a CPU job after the work queue is closed");
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
  std::queue<std::unique_ptr<cpu_job>> work_queue_;
  std::mutex work_queue_mtx_;
  std::condition_variable work_queue_ready_;
  bool closed_{false};
};

class gpipnn_host_manager {
  template <typename DataT>
  friend class gpipnn_main_manager;

public:
  enum class WorkerStatus { kSuccess, kFailure };

  explicit gpipnn_host_manager(std::ptrdiff_t num_workers)
    : num_workers_{validate_num_workers(num_workers)},
      workers_(num_workers_),
      worker_contexts_(num_workers_),
      exit_statuses_(num_workers_, WorkerStatus::kSuccess)
  {
    for (std::size_t worker_id = 0; worker_id < num_workers_; ++worker_id) {
      worker_contexts_[worker_id] = worker_context{.manager=this, .worker_id=worker_id};
    }

    const int result = pthread_create(&manager_thread_,
                                      nullptr,
                                      &gpipnn_host_manager::manager_start_routine,
                                      this);
    RAFT_EXPECTS(result == 0, "pthread_create failed for gpipnn_host_manager thread with error %d", result);
  }

  gpipnn_host_manager(const gpipnn_host_manager&) = delete;
  auto operator=(const gpipnn_host_manager&) -> gpipnn_host_manager& = delete;
  gpipnn_host_manager(gpipnn_host_manager&&) = delete;
  auto operator=(gpipnn_host_manager&&) -> gpipnn_host_manager& = delete;

  ~gpipnn_host_manager() noexcept
  {
    try {
      work_queue_.close();

      const int result = pthread_join(manager_thread_, nullptr);
      if (result != 0) {
        std::cerr << "Failed to join gpipnn_host_manager thread with error " << result << std::endl;
      } else if (manager_status_ == WorkerStatus::kFailure) {
        std::cerr << "One or more CPU workers failed before host manager destruction" << std::endl;
      }
    } catch (const std::exception& error) {
      std::cerr << "Failed to join CPU workers during host manager destruction: " << error.what()
                << std::endl;
    } catch (...) {
      std::cerr << "Failed to join CPU workers during host manager destruction" << std::endl;
    }
  }

  void push_job(std::unique_ptr<cpu_job> job)
  {
    work_queue_.push_job(std::move(job));
  }

private:
  struct worker_context {
    gpipnn_host_manager* manager;
    std::size_t worker_id;
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
    std::size_t num_launched = 0;
    for (; num_launched < num_workers_; ++num_launched) {
      const int result = pthread_create(&workers_[num_launched],
                                        nullptr,
                                        &gpipnn_host_manager::worker_start_routine,
                                        &worker_contexts_[num_launched]);
      if (result != 0) {
        std::cerr << "CPU worker creation failed with error " << result << std::endl;
        manager_status_ = WorkerStatus::kFailure;
        work_queue_.close();
        break;
      }
    }

    int join_error = 0;
    for (std::size_t worker = 0; worker < num_launched; ++worker) {
      const int result = pthread_join(workers_[worker], nullptr);
      if (join_error == 0 && result != 0) { join_error = result; }
    }
    workers_.clear();

    if (join_error != 0) {
      std::cerr << "CPU worker join failed with error " << join_error << std::endl;
      manager_status_ = WorkerStatus::kFailure;
    }
    if (std::ranges::any_of(exit_statuses_.begin(),
                            exit_statuses_.begin() + num_launched,
                            [](auto status) { return status == WorkerStatus::kFailure; })) {
      manager_status_ = WorkerStatus::kFailure;
    }
    work_queue_.close();
  }

  static auto worker_start_routine(void* argument) noexcept -> void*
  {
    auto& context = *static_cast<worker_context*>(argument);
    context.manager->run_worker(context.worker_id);
    return nullptr;
  }

  void run_worker(std::size_t worker_id) noexcept
  {
    auto worker_status = WorkerStatus::kSuccess;
    try {
      while (true) {
        auto job             = work_queue_.pop_job();
        if (job == nullptr) {
          std::cout << "CPU worker " << worker_id << " exiting after queue close" << std::endl;
          break;
        }
        const auto job_status = job->execute();
        if (job_status == cpu_job::ExitStatus::kSuccess) { continue; }

        if (job_status == cpu_job::ExitStatus::kFailure) {
          std::cerr << "CPU worker " << worker_id << " exiting with failure" << std::endl;
          worker_status = WorkerStatus::kFailure;
          break;
        } else {
          RAFT_FAIL("Unsupported CPU job exit status");
        }
      }
    } catch (const std::exception& error) {
      std::cerr << "CPU worker " << worker_id << " exiting after exception: " << error.what()
                << std::endl;
      worker_status = WorkerStatus::kFailure;
    } catch (...) {
      std::cerr << "CPU worker " << worker_id << " exiting after unknown exception" << std::endl;
      worker_status = WorkerStatus::kFailure;
    }
    exit_statuses_[worker_id] = worker_status;
  }

  const std::size_t num_workers_;
  cpu_work_queue work_queue_;
  pthread_t manager_thread_{};
  std::vector<pthread_t> workers_;
  std::vector<worker_context> worker_contexts_;
  std::vector<WorkerStatus> exit_statuses_;
  WorkerStatus manager_status_{WorkerStatus::kSuccess};
};

// device memory buffer for single GPU worker (max pipelined cuda streams is effectively number of slots)
class gpu_buffer {
public:
  class gpu_buffer_slot final : public buffer_slot {
    friend class gpu_buffer;

  public:
    gpu_buffer_slot(const gpu_buffer_slot&) = delete;
    auto operator=(const gpu_buffer_slot&) -> gpu_buffer_slot& = delete;
    gpu_buffer_slot(gpu_buffer_slot&&) noexcept = default;
    auto operator=(gpu_buffer_slot&&) noexcept -> gpu_buffer_slot& = delete;

    [[nodiscard]] auto data() noexcept -> void* override { return data_.data(); }
    [[nodiscard]] auto size() const noexcept -> std::size_t override { return data_.size(); }
    [[nodiscard]] auto event() const noexcept -> cudaEvent_t { return event_; }

  private:
    gpu_buffer_slot(std::size_t buffer_size, cuda::stream_ref stream)
      : data_{buffer_size, stream}
    {
    }

    rmm::device_buffer data_;
    cudaEvent_t event_{nullptr};
  };

  gpu_buffer(const std::ptrdiff_t num_slots,
             const std::size_t buffer_size,
             cuda::stream_ref stream)
  {
    const auto validated_num_slots = validate_num_slots(num_slots);
    slots_.reserve(static_cast<std::size_t>(validated_num_slots));
    for (std::ptrdiff_t slot = 0; slot < validated_num_slots; ++slot) {
      slots_.push_back(gpu_buffer_slot{buffer_size, stream});
    }
  }

  // TODO(sdickman): check if necessary synchronization step
  ~gpu_buffer()
  {
    for (const auto& slot : slots_) {
      if (slot.event_ != nullptr) { RAFT_CUDA_TRY_NO_THROW(cudaEventSynchronize(slot.event_)); }
    }
  }

  // event represents the point at which the slot can be released
  [[nodiscard]] auto get_slot(cudaEvent_t event) -> gpu_buffer_slot&
  {
    RAFT_EXPECTS(event != nullptr, "event dependency must not be nullptr");

    auto& slot = slots_[next_slot_];
    if (slot.event_ != nullptr) { RAFT_CUDA_TRY(cudaEventSynchronize(slot.event_)); }
    slot.event_ = event;
    next_slot_ = (next_slot_ + 1) % slots_.size();
    return slot;
  }

private:
  static auto validate_num_slots(std::ptrdiff_t num_slots) -> std::ptrdiff_t
  {
    RAFT_EXPECTS(num_slots > 0, "gpu_buffer requires at least one slot");
    return num_slots;
  }

  std::vector<gpu_buffer_slot> slots_;
  std::size_t next_slot_{0};
};

class gpipnn_dev_manager {
  template <typename DataT>
  friend class gpipnn_main_manager;

};

template <typename DataT>
class gpipnn_main_manager {

public:
  void build(raft::resources const& res,
    const index_params& params,
    raft::device_matrix_view<const DataT, int64_t, raft::row_major> dataset,
    raft::device_matrix_view<id_t, int64_t, raft::row_major> graph) {
    
  }

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
