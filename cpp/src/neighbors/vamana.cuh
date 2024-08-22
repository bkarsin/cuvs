/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include "detail/vamana/vamana_build.cuh"
//#include "detail/vamana/vamana_search.cuh"

#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_device_accessor.hpp>
#include <raft/core/mdspan.hpp>
#include <raft/core/resources.hpp>

#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/vamana.hpp>

#include <cuvs/neighbors/common.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cuvs::neighbors::vamana {

/**
 * @defgroup VAMANA ANN Graph-based nearest neighbor search
 * @{
 */


/**
 * @brief Build the index from the dataset for efficient search.
 *
 * The build consist of two steps: build an intermediate knn-graph, and optimize it to
 * create the final graph. The index_params struct controls the node degree of these
 * graphs.
 *
 * It is required that dataset and the optimized graph fit the GPU memory.
 *
 * To customize the parameters for knn-graph building and pruning, and to reuse the
 * intermediate results, you could build the index in two steps using
 * [cagra::build_knn_graph](#cagra::build_knn_graph) and [cagra::optimize](#cagra::optimize).
 *
 * The following distance metrics are supported:
 * - L2
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   cagra::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   auto index = cagra::build(res, index_params, dataset);
 *   // use default search parameters
 *   cagra::search_params search_params;
 *   // search K nearest neighbours
 *   auto neighbors = raft::make_device_matrix<uint32_t>(res, n_queries, k);
 *   auto distances = raft::make_device_matrix<float>(res, n_queries, k);
 *   cagra::search(res, search_params, index, queries, neighbors, distances);
 * @endcode
 *
 * @tparam T data element type
 * @tparam IdxT type of the indices in the source dataset
 *
 * @param[in] res
 * @param[in] params parameters for building the index
 * @param[in] dataset a matrix view (host or device) to a row-major matrix [n_rows, dim]
 *
 * @return the constructed cagra index
 */
template <typename T,
          typename IdxT     = uint32_t,
          typename Accessor = raft::host_device_accessor<std::experimental::default_accessor<T>,
                                                         raft::memory_type::host>>
index<T, IdxT> build(
  raft::resources const& res,
  const index_params& params,
  raft::mdspan<const T, raft::matrix_extent<int64_t>, raft::row_major, Accessor> dataset)
{
  return cuvs::neighbors::vamana::detail::build<T, IdxT, Accessor>(res, params, dataset);
}

/*
template <typename T, typename IdxT>
void search(raft::resources const& res,
            const search_params& params,
            const index<T, IdxT>& idx,
            raft::device_matrix_view<const T, int64_t, raft::row_major> queries,
            raft::device_matrix_view<IdxT, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances)
{
  using none_filter_type = cuvs::neighbors::filtering::none_cagra_sample_filter;
  return cagra::search_with_filtering<T, IdxT, none_filter_type>(
    res, params, idx, queries, neighbors, distances, none_filter_type{});
}
*/

/** @} */  // end group vamana

}  // namespace cuvs::neighbors::cagra
