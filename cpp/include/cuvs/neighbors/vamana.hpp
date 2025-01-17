/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "common.hpp"
#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/common.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_device_accessor.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/mdspan.hpp>
#include <raft/core/resource/stream_view.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/integer_utils.hpp>
#include <rmm/cuda_stream_view.hpp>

#include <optional>
#include <variant>

namespace cuvs::neighbors::vamana {
/**
 * @defgroup vamana_cpp_index_params Vamana index build parameters
 * @{
 */

struct index_params : cuvs::neighbors::index_params {
  /** Degree of output graph. */
  size_t graph_degree = 32;
  /** Number of Vamana iterations. */
  size_t vamana_iters = 1;
  /** Maximum fraction of dataset inserted per batch **/
  float max_batchsize = 0.06;
  /** Maximum number of visited nodes per search **/
  float max_visited = 64;

  /**
   * Specify compression parameters if compression is desired.
   *
   * NOTE: this is experimental new API, consider it unsafe.
   */
  std::optional<cuvs::neighbors::vpq_params> compression = std::nullopt;
};

/**
 * @}
 */

/**
 * @defgroup vamana_cpp_search_params Vamana index search parameters
 * @{
 */
//enum class search_algo {
//};

enum class hash_mode { HASH, SMALL, AUTO };

struct search_params : cuvs::neighbors::search_params {
  /** Maximum number of queries to search at the same time (batch size). Auto select when 0.*/
  size_t max_queries = 0;

  /** Upper limit of search iterations. Auto select when 0.*/
  size_t max_iterations = 0;
};

/**
 * @}
 */
static_assert(std::is_aggregate_v<index_params>);
static_assert(std::is_aggregate_v<search_params>);

/**
 * @defgroup vamana_cpp_index Vamana index type
 * @{
 */

/**
 * @brief Vamana index.
 *
 * The index stores the dataset and a kNN graph in device memory.
 *
 * @tparam T data element type
 * @tparam IdxT type of the vector indices (represent dataset.extent(0))
 *
 */
template <typename T, typename IdxT>
struct index : cuvs::neighbors::index {
  static_assert(!raft::is_narrowing_v<uint32_t, IdxT>,
                "IdxT must be able to represent all values of uint32_t");

 public:
  /** Distance metric used for clustering. */
  [[nodiscard]] constexpr inline auto metric() const noexcept -> cuvs::distance::DistanceType
  {
    return metric_;
  }

  /** Total length of the index (number of vectors). */
  [[nodiscard]] constexpr inline auto size() const noexcept -> IdxT
  {
    auto data_rows = dataset_->n_rows();
    return data_rows > 0 ? data_rows : graph_view_.extent(0);
  }

  /** Dimensionality of the data. */
  [[nodiscard]] constexpr inline auto dim() const noexcept -> uint32_t { return dataset_->dim(); }
  /** Graph degree */
  [[nodiscard]] constexpr inline auto graph_degree() const noexcept -> uint32_t
  {
    return graph_view_.extent(1);
  }

  [[nodiscard]] inline auto dataset() const noexcept
    -> raft::device_matrix_view<const T, int64_t, raft::layout_stride>
  {
    auto p = dynamic_cast<strided_dataset<T, int64_t>*>(dataset_.get());
    if (p != nullptr) { return p->view(); }
    auto d = dataset_->dim();
    return raft::make_device_strided_matrix_view<const T, int64_t>(nullptr, 0, d, d);
  }

  /** Dataset [size, dim] */
  [[nodiscard]] inline auto data() const noexcept -> const cuvs::neighbors::dataset<int64_t>&
  {
    return *dataset_;
  }

  /** neighborhood graph [size, graph-degree] */
  [[nodiscard]] inline auto graph() const noexcept
    -> raft::device_matrix_view<const IdxT, int64_t, raft::row_major>
  {
    return graph_view_;
  }

  // Don't allow copying the index for performance reasons (try avoiding copying data)
  index(const index&)                    = delete;
  index(index&&)                         = default;
  auto operator=(const index&) -> index& = delete;
  auto operator=(index&&) -> index&      = default;
  ~index()                               = default;

  /** Construct an empty index. */
  index(raft::resources const& res,
        cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Expanded)
    : cuvs::neighbors::index(),
      metric_(metric),
      graph_(raft::make_device_matrix<IdxT, int64_t>(res, 0, 0)),
      dataset_(new cuvs::neighbors::empty_dataset<int64_t>(0))
  {
  }

/** Construct an index from dataset and vamana graph
   *
   */
  template<typename data_accessor, typename graph_accessor>
  index(raft::resources const& res,
        cuvs::distance::DistanceType metric,
        raft::mdspan<const T, raft::matrix_extent<int64_t>, raft::row_major, data_accessor> dataset,
        raft::mdspan<const IdxT, raft::matrix_extent<int64_t>, raft::row_major, graph_accessor>
          vamana_graph)
    : cuvs::neighbors::index(),
      metric_(metric),
      graph_(raft::make_device_matrix<IdxT, int64_t>(res, 0, 0)),
      dataset_(make_aligned_dataset(res, dataset, 16))
  {
//    RAFT_EXPECTS(dataset.extent(0) == vamana_graph.extent(0),
//                 "Dataset and knn_graph must have equal number of rows");
//    update_graph(res, vamana_graph);

    raft::resource::sync_stream(res);
  }


  /**
   * Replace the dataset with a new dataset.
   *
   * If the new dataset rows are aligned on 16 bytes, then only a reference is stored to the
   * dataset. It is the caller's responsibility to ensure that dataset stays alive as long as the
   * index.
   */
  void update_dataset(raft::resources const& res,
                      raft::device_matrix_view<const T, int64_t, raft::row_major> dataset)
  {
    dataset_ = make_aligned_dataset(res, dataset, 16);
  }

  /** Set the dataset reference explicitly to a device matrix view with padding. */
  void update_dataset(raft::resources const& res,
                      raft::device_matrix_view<const T, int64_t, raft::layout_stride> dataset)
  {
    dataset_ = make_aligned_dataset(res, dataset, 16);
  }

  /**
   * Replace the dataset with a new dataset.
   *
   * We create a copy of the dataset on the device. The index manages the lifetime of this copy.
   */
  void update_dataset(raft::resources const& res,
                      raft::host_matrix_view<const T, int64_t, raft::row_major> dataset)
  {
    dataset_ = make_aligned_dataset(res, dataset, 16);
  }

  /** Replace the dataset with a new dataset. */
  template <typename DatasetT>
  auto update_dataset(raft::resources const& res, DatasetT&& dataset)
    -> std::enable_if_t<std::is_base_of_v<cuvs::neighbors::dataset<int64_t>, DatasetT>>
  {
    dataset_ = std::make_unique<DatasetT>(std::move(dataset));
  }

  template <typename DatasetT>
  auto update_dataset(raft::resources const& res, std::unique_ptr<DatasetT>&& dataset)
    -> std::enable_if_t<std::is_base_of_v<neighbors::dataset<int64_t>, DatasetT>>
  {
    dataset_ = std::move(dataset);
  }

  /**
   * Replace the graph with a new graph.
   *
   * Since the new graph is a device array, we store a reference to that, and it is
   * the caller's responsibility to ensure that knn_graph stays alive as long as the index.
   */
  void update_graph(raft::resources const& res,
                    raft::device_matrix_view<const IdxT, int64_t, raft::row_major> knn_graph)
  {
    graph_view_ = knn_graph;
  }

  /**
   * Replace the graph with a new graph.
   *
   * We create a copy of the graph on the device. The index manages the lifetime of this copy.
   */
  void update_graph(raft::resources const& res,
                    raft::host_matrix_view<const IdxT, int64_t, raft::row_major> knn_graph)
  {
    RAFT_LOG_DEBUG("Copying Vamana knn graph from host to device");

    if ((graph_.extent(0) != knn_graph.extent(0)) || (graph_.extent(1) != knn_graph.extent(1))) {
      // clear existing memory before allocating to prevent OOM errors on large graphs
      if (graph_.size()) { graph_ = raft::make_device_matrix<IdxT, int64_t>(res, 0, 0); }
      graph_ =
        raft::make_device_matrix<IdxT, int64_t>(res, knn_graph.extent(0), knn_graph.extent(1));
    }
    raft::copy(graph_.data_handle(),
               knn_graph.data_handle(),
               knn_graph.size(),
               raft::resource::get_cuda_stream(res));
    graph_view_ = graph_.view();
  }

 private:
  cuvs::distance::DistanceType metric_;
  raft::device_matrix<IdxT, int64_t, raft::row_major> graph_;
  raft::device_matrix_view<const IdxT, int64_t, raft::row_major> graph_view_;
  std::unique_ptr<neighbors::dataset<int64_t>> dataset_;
};
/**
 * @}
 */

/**
 * @defgroup vamana_cpp_index_build Vamana index build functions
 * @{
 */
/**
 * @brief Build the index from the dataset for efficient search.
 *
 */
auto build(raft::resources const& handle,
           const cuvs::neighbors::vamana::index_params& params,
           raft::device_matrix_view<const float, int64_t, raft::row_major> dataset)
  -> cuvs::neighbors::vamana::index<float, uint32_t>;

auto build(raft::resources const& handle,
           const cuvs::neighbors::vamana::index_params& params,
           raft::host_matrix_view<const float, int64_t, raft::row_major> dataset)
  -> cuvs::neighbors::vamana::index<float, uint32_t>;

auto build(raft::resources const& handle,
           const cuvs::neighbors::vamana::index_params& params,
           raft::device_matrix_view<const int8_t, int64_t, raft::row_major> dataset)
  -> cuvs::neighbors::vamana::index<int8_t, uint32_t>;

auto build(raft::resources const& handle,
           const cuvs::neighbors::vamana::index_params& params,
           raft::host_matrix_view<const int8_t, int64_t, raft::row_major> dataset)
  -> cuvs::neighbors::vamana::index<int8_t, uint32_t>;

auto build(raft::resources const& handle,
           const cuvs::neighbors::vamana::index_params& params,
           raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> dataset)
  -> cuvs::neighbors::vamana::index<uint8_t, uint32_t>;

auto build(raft::resources const& handle,
           const cuvs::neighbors::vamana::index_params& params,
           raft::host_matrix_view<const uint8_t, int64_t, raft::row_major> dataset)
  -> cuvs::neighbors::vamana::index<uint8_t, uint32_t>;
/**
 * @}
 */

/*
void build_device(raft::resources const& handle,
                  const cuvs::neighbors::vamana::index_params& params,
                  raft::device_matrix_view<const float, int64_t, raft::row_major> dataset,
                  cuvs::neighbors::vamana::index<float, uint32_t>& idx);

void build_host(raft::resources const& handle,
                const cuvs::neighbors::vamana::index_params& params,
                raft::host_matrix_view<const float, int64_t, raft::row_major> dataset,
                cuvs::neighbors::vamana::index<float, uint32_t>& idx);

void build_device(raft::resources const& handle,
                  const cuvs::neighbors::vamana::index_params& params,
                  raft::device_matrix_view<const int8_t, int64_t, raft::row_major> dataset,
                  cuvs::neighbors::vamana::index<int8_t, uint32_t>& idx);

void build_host(raft::resources const& handle,
                const cuvs::neighbors::vamana::index_params& params,
                raft::host_matrix_view<const int8_t, int64_t, raft::row_major> dataset,
                cuvs::neighbors::vamana::index<int8_t, uint32_t>& idx);

void build_device(raft::resources const& handle,
                  const cuvs::neighbors::vamana::index_params& params,
                  raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> dataset,
                  cuvs::neighbors::vamana::index<uint8_t, uint32_t>& idx);

void build_host(raft::resources const& handle,
                const cuvs::neighbors::vamana::index_params& params,
                raft::host_matrix_view<const uint8_t, int64_t, raft::row_major> dataset,
                cuvs::neighbors::vamana::index<uint8_t, uint32_t>& idx);
*/

/**
 * @defgroup vamana_cpp_serialize Vamana serialize functions
 * @{
 */
/**
 * Save the index to file.
 */

void serialize_file(raft::resources const& handle,
                    const std::string& filename,
                    const cuvs::neighbors::vamana::index<float, uint32_t>& index,
                    bool include_dataset = true);

void deserialize_file(raft::resources const& handle,
                      const std::string& filename,
                      cuvs::neighbors::vamana::index<float, uint32_t>* index);
void serialize(raft::resources const& handle,
               std::string& str,
               const cuvs::neighbors::vamana::index<float, uint32_t>& index,
               bool include_dataset = true);

void deserialize(raft::resources const& handle,
                 const std::string& str,
                 cuvs::neighbors::vamana::index<float, uint32_t>* index);

void serialize_file(raft::resources const& handle,
                    const std::string& filename,
                    const cuvs::neighbors::vamana::index<int8_t, uint32_t>& index,
                    bool include_dataset = true);

void deserialize_file(raft::resources const& handle,
                      const std::string& filename,
                      cuvs::neighbors::vamana::index<int8_t, uint32_t>* index);
void serialize(raft::resources const& handle,
               std::string& str,
               const cuvs::neighbors::vamana::index<int8_t, uint32_t>& index,
               bool include_dataset = true);

void deserialize(raft::resources const& handle,
                 const std::string& str,
                 cuvs::neighbors::vamana::index<int8_t, uint32_t>* index);

void serialize_file(raft::resources const& handle,
                    const std::string& filename,
                    const cuvs::neighbors::vamana::index<uint8_t, uint32_t>& index,
                    bool include_dataset = true);

void deserialize_file(raft::resources const& handle,
                      const std::string& filename,
                      cuvs::neighbors::vamana::index<uint8_t, uint32_t>* index);
void serialize(raft::resources const& handle,
               std::string& str,
               const cuvs::neighbors::vamana::index<uint8_t, uint32_t>& index,
               bool include_dataset = true);

void deserialize(raft::resources const& handle,
                 const std::string& str,
                 cuvs::neighbors::vamana::index<uint8_t, uint32_t>* index);
/**
 * @}
 */

}  // namespace cuvs::neighbors::vamana
