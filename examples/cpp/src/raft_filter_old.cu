/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <cstdint>
#include <optional>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/neighbors/cagra.cuh>
#include <raft/util/cudart_utils.hpp>
#include <raft/random/make_blobs.cuh>

#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/counting_iterator.h>

struct cagra_filter {
  raft::device_vector_view<int32_t> data_cats;
  raft::device_vector_view<int32_t> query_cats;

  inline _RAFT_HOST_DEVICE bool operator()(
    // query index
    const uint32_t query_ix,
    const uint32_t sample_ix) const
  {
    return data_cats(sample_ix) == query_cats(query_ix);
  }
};



// Fill dataset and queries with synthetic data.
void generate_dataset(raft::device_resources const& dev_resources,
                      raft::device_matrix_view<float, int64_t> dataset,
                      raft::device_matrix_view<float, int64_t> queries)
{
  auto labels = raft::make_device_vector<int64_t, int64_t>(dev_resources, dataset.extent(0));
  raft::random::make_blobs(dev_resources, dataset, labels.view());
  raft::random::RngState r(1234ULL);
  raft::random::uniform(dev_resources,
                        r,
                        raft::make_device_vector_view(queries.data_handle(), queries.size()),
                        -1.0f,
                        1.0f);
}


// Generate random categories for each data vector and query
void generate_categories(raft::device_resources const& dev_resources,
                      raft::device_vector_view<int32_t> dataset,
                      raft::device_vector_view<int32_t> queries,
		      int num_categories)
{
  raft::random::RngState r(1234ULL);
  raft::random::uniformInt(dev_resources, r, raft::make_device_vector_view(dataset.data_handle(), dataset.size()), 0,num_categories);
  raft::random::uniformInt(dev_resources, r, raft::make_device_vector_view(queries.data_handle(), queries.size()), 0,num_categories);

}

// Copy the results to host and print them
template <typename IdxT>
void print_results(raft::device_resources const& dev_resources,
                   raft::device_matrix_view<IdxT, int64_t> neighbors,
                   raft::device_matrix_view<float, int64_t> distances,
		   raft::device_vector_view<int32_t> data_cats,
		   raft::device_vector_view<int32_t> query_cats)
{
  int64_t topk        = neighbors.extent(1);
  auto neighbors_host = raft::make_host_matrix<IdxT, int64_t>(neighbors.extent(0), topk);
  auto distances_host = raft::make_host_matrix<float, int64_t>(distances.extent(0), topk);
  auto data_cats_host = raft::make_host_vector<int32_t>(data_cats.size());
  auto query_cats_host = raft::make_host_vector<int32_t>(query_cats.size());

  cudaStream_t stream = raft::resource::get_cuda_stream(dev_resources);

  raft::copy(neighbors_host.data_handle(), neighbors.data_handle(), neighbors.size(), stream);
  raft::copy(distances_host.data_handle(), distances.data_handle(), distances.size(), stream);
  raft::copy(data_cats_host.data_handle(), data_cats.data_handle(), data_cats.size(), stream);
  raft::copy(query_cats_host.data_handle(), query_cats.data_handle(), query_cats.size(), stream);


  // The calls to RAFT algorithms and  raft::copy is asynchronous.
  // We need to sync the stream before accessing the data.
  raft::resource::sync_stream(dev_resources, stream);

  for (int query_id = 0; query_id < 10; query_id++) {
    std::cout << "Query " << query_id << " (" << query_cats_host(query_id) << ")  neighbor indices: [";
    for(int i=0; i<topk; i++) {
      std::cout << neighbors_host(query_id, i) << " (" << data_cats_host(neighbors_host(query_id,i)) << "), ";
    }
    std::cout << "]" << std::endl;
  }
}


void cagra_build_search_filtered(raft::device_resources const& dev_resources,
                                  raft::device_matrix_view<const float, int64_t> dataset,
                                  raft::device_matrix_view<const float, int64_t> queries,
				  raft::device_vector_view<int32_t> data_cats,
				  raft::device_vector_view<int32_t> query_cats)
{
  using namespace raft::neighbors;

  cagra::index_params index_params;
  index_params.metric                   = raft::distance::DistanceType::L2Expanded;

  std::cout << "Building CAGRA index" << std::endl;
  auto index = cagra::build(dev_resources, index_params, dataset);

  std::cout << "CAGRA index has " << index.size() << " vectors" << std::endl;
  std::cout << "CAGRA graph has degree " << index.graph_degree() << ", graph size ["
            << index.graph().extent(0) << ", " << index.graph().extent(1) << "]" << std::endl;


  // Create output arrays.
  int64_t topk      = 10;
  int64_t n_queries = queries.extent(0);
  auto neighbors    = raft::make_device_matrix<uint32_t>(dev_resources, n_queries, topk);
  auto distances    = raft::make_device_matrix<float>(dev_resources, n_queries, topk);

  // Set search parameters.
  cagra::search_params search_params;

  // Create category filter structure and copy over metadata and cluster index lookup
  cagra_filter cat_filter;
  cat_filter.data_cats = data_cats;
  cat_filter.query_cats = query_cats;

  // Search K nearest neighbors for each of the queries.
  cagra::search_with_filtering<float, uint32_t>(
    dev_resources, search_params, index, queries, neighbors.view(), distances.view(), cat_filter);

  print_results(dev_resources, neighbors.view(), distances.view(), data_cats, query_cats);
}


int main()
{
  raft::device_resources dev_resources;

  // Set pool memory resource with 1 GiB initial pool size. All allocations use the same pool.
  rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource> pool_mr(
    rmm::mr::get_current_device_resource(), 1024 * 1024 * 1024ull);
  rmm::mr::set_current_device_resource(&pool_mr);

  // Create input arrays.
  int64_t n_samples = 10000;
  int64_t n_dim     = 3;
  int64_t n_queries = 10;
  int n_categories = 5;
  auto dataset      = raft::make_device_matrix<float, int64_t>(dev_resources, n_samples, n_dim);
  auto queries      = raft::make_device_matrix<float, int64_t>(dev_resources, n_queries, n_dim);

  generate_dataset(dev_resources, dataset.view(), queries.view());

  auto data_cats = raft::make_device_vector<int32_t>(dev_resources, n_samples);
  auto query_cats = raft::make_device_vector<int32_t>(dev_resources, n_queries);

  generate_categories(dev_resources, data_cats.view(), query_cats.view(), n_categories);

  // Simple build and search example.
  cagra_build_search_filtered(dev_resources,
                               raft::make_const_mdspan(dataset.view()),
                               raft::make_const_mdspan(queries.view()),
			       data_cats.view(),
			       query_cats.view());

}
