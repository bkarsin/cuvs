/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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
#include <raft/core/handle.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/random/make_blobs.cuh>

#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/brute_force.hpp>

#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include "common.cuh"

#define CATS 5

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
void generate_dataset_test(raft::device_resources const &dev_resources,
                      raft::device_matrix_view<float, int64_t> dataset,
                      raft::device_matrix_view<float, int64_t> queries,
                      int n_clusters,
                      int64_t seed) {
  auto labels = raft::make_device_vector<int64_t, int64_t>(dev_resources,
                                                           dataset.extent(0));
  cudaStream_t stream = raft::resource::get_cuda_stream(dev_resources);
  raft::random::make_blobs<float,int64_t>(dataset.data_handle(),
                           labels.data_handle(),
                           dataset.extent(0),
                           dataset.extent(1),
                           n_clusters,
                           stream, 
                           true,
                           nullptr,
                           nullptr,
                           float(1.0),
                           false,
                           (float)-10.0f,
                           (float)10.0f,
                           (uint64_t)seed);

  raft::random::RngState r(seed);
  raft::random::uniform(
      dev_resources, r,
      raft::make_device_vector_view(queries.data_handle(), queries.size()),
      -10.0f, 10.0f);
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

void compute_recall_unfiltered(raft::device_resources const& dev_resources, 
                    raft::device_matrix_view<const float, int64_t> dataset, 
                    raft::device_matrix_view<const float, int64_t> queries,
                    raft::device_matrix_view<uint32_t> neighbors,
                    int topk) {

  int n_queries = queries.extent(0);
  cudaStream_t stream = raft::resource::get_cuda_stream(dev_resources);

  auto bf_index = cuvs::neighbors::brute_force::build(dev_resources, dataset);
  
  auto bf_neighbors = raft::make_device_matrix<int64_t,int64_t>(dev_resources, n_queries, topk);
  auto bf_distances = raft::make_device_matrix<float,int64_t>(dev_resources, n_queries, topk);

  cuvs::neighbors::brute_force::search(dev_resources, bf_index, queries, bf_neighbors.view(), bf_distances.view(), std::nullopt);

  auto h_bf_neighbors = raft::make_host_matrix<int64_t,int64_t>(n_queries, topk);
  raft::copy(h_bf_neighbors.data_handle(), bf_neighbors.data_handle(), bf_neighbors.size(), stream);

  auto h_neighbors = raft::make_host_matrix<uint32_t>(n_queries, topk);
  raft::copy(h_neighbors.data_handle(), neighbors.data_handle(), neighbors.size(), stream);

//printf("computing unfiltered recall...\n");
  int correct=0;
  for(int i=0; i<neighbors.extent(0); i++) {
    for(int j=0; j<topk; j++) {
      for(int k=0; k<topk; k++) {
        if(h_neighbors(i,j) == h_bf_neighbors(i,k)) {
          correct++;
          break;
        }
      }
    }
  }

  printf("Recall - queries:%d, topk:%d, correct:%d, recall:%f\n", n_queries, topk, correct, (float)correct / (float)(n_queries*topk));

   
}

void cagra_build_search_subgraphs(raft::device_resources const& dev_resources,
                               std::vector<raft::device_matrix<float, int64_t>> datasets,
                               std::vector<raft::device_matrix<float, int64_t>> queries,
                               int n_categories,
                               int topk)
{
  printf("Starting fcn...\n");
  using namespace cuvs::neighbors;

  cudaStream_t stream = raft::resource::get_cuda_stream(dev_resources);

  // Create neighbors and dist output lists for each cat
  std::vector<raft::device_matrix<uint32_t, int64_t>> neighbors;
  neighbors.reserve(n_categories);
  std::vector<raft::device_matrix<float,int64_t>> distances;
  distances.reserve(n_categories);

  // use default index parameters
  cagra::index_params index_params;

  std::vector<cuvs::neighbors::cagra::index<float,uint32_t>> indexes;
  indexes.reserve(n_categories);

  for(int i=0; i<n_categories; i++) {
    neighbors.push_back(raft::make_device_matrix<uint32_t,int64_t>(dev_resources, queries[i].extent(0), topk));
    distances.push_back(raft::make_device_matrix<float,int64_t>(dev_resources, queries[i].extent(0), topk));

    printf("calling build for dataset %d...\n", i);

    indexes.push_back(cagra::build(dev_resources, index_params, raft::make_const_mdspan(datasets[i].view())));
  }

  raft::resource::sync_stream(dev_resources);

  printf("Indexes all created!\n");

  // use default search parameters
  cagra::search_params search_params;
  // search K nearest neighbors
  for(int i=0; i<n_categories; i++) {
    cagra::search(dev_resources, search_params, indexes[i], raft::make_const_mdspan(queries[i].view()), neighbors[i].view(), distances[i].view());

//    print_results(dev_resources, neighbors[i].view(), distances[i].view());
    printf("Label: %d - ", i);
    compute_recall_unfiltered(dev_resources, datasets[i].view(), queries[i].view(), neighbors[i].view(), topk);
  }

}

void print_vecs(raft::device_resources const &dev_resources, 
                  raft::device_matrix_view<float,int64_t> dataset,
                  raft::device_matrix_view<float,int64_t> queries) {
  auto h_data = raft::make_host_matrix<float,int64_t>(dataset.extent(0), dataset.extent(1));
  auto h_queries = raft::make_host_matrix<float,int64_t>(queries.extent(0), queries.extent(1));

  cudaStream_t stream = raft::resource::get_cuda_stream(dev_resources);
  raft::copy(h_data.data_handle(), dataset.data_handle(), dataset.size(), stream);
  raft::copy(h_queries.data_handle(), queries.data_handle(), queries.size(), stream);

  printf("data:\n");
  for(int i=0; i<10; i++) {
    printf("i:%d - ", i);
    for(int j=0; j<h_data.extent(1); j++) {
      printf("%f, ", h_data(i,j));
    }
    printf("\n");
  }
  printf("queries:\n");
  for(int i=0; i<10; i++) {
    printf("i:%d - ", i);
    for(int j=0; j<h_queries.extent(1); j++) {
      printf("%f, ", h_queries(i,j));
    }
    printf("\n");
  }
}

int main()
{
  raft::device_resources dev_resources;

  // Set pool memory resource with 1 GiB initial pool size. All allocations use the same pool.
  rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource> pool_mr(
    rmm::mr::get_current_device_resource(), 10*1024 * 1024 * 1024ull);
  rmm::mr::set_current_device_resource(&pool_mr);

  // Create input arrays.
  int64_t n_samples = 10000;
  int64_t n_dim     = 5;
  int64_t n_queries = 100;
  int64_t topk      = 10;
  int n_categories = CATS;
  int n_data_clusters = 10;
  std::vector<raft::device_matrix<float, int64_t>> datasets;
  datasets.reserve(n_categories);
  std::vector<raft::device_matrix<float, int64_t>> queries;
  queries.reserve(n_categories);

  for(int i=0; i<n_categories; i++) {
    datasets.push_back(raft::make_device_matrix<float, int64_t>(dev_resources, n_samples, n_dim));
    queries.push_back(raft::make_device_matrix<float, int64_t>(dev_resources, n_queries, n_dim));
    generate_dataset_test(dev_resources, datasets[i].view(), queries[i].view(), n_data_clusters, i);

  }

//  for(int i=0; i<n_categories; i++) {
//    print_vecs(dev_resources, datasets[i].view(), queries[i].view());
//  }

    // Simple build and search example.
  cagra_build_search_subgraphs(dev_resources,
                            datasets, queries, n_categories, topk);

  return 0;
}
