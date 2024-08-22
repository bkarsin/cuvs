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

//#include "../../../cpp/src/neighbors/detail/vamana/vamana_search.cuh"
//#include "../../../cpp/src/neighbors/detail/vamana/robust_prune.cuh"

#include <cstdlib>
#include <cstdint>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/random/make_blobs.cuh>
#include <raft/sparse/neighbors/cross_component_nn.cuh>

#include <cuvs/neighbors/vamana.hpp>

#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include "common.cuh"
/*
__global__ void print_graph(raft::device_matrix_view<uint32_t, int64_t> graph) {
  printf("nodes:%ld, degree:%ld\n", graph.extent(0), graph.extent(1));
  for(int i=0; i<10; i++) {
    for(int j=0; j<graph.extent(1); j++) {
      printf("%u, ", graph(i,j));
    }
    printf("\n");
  }
}
*/

//using namespace std;
void create_random_graph(raft::host_matrix_view<uint32_t, int64_t> r_graph) {

  std::srand(1);

  for(int i=0; i<r_graph.extent(0); i++) {
    for(int j=0; j<r_graph.extent(1); j++) {
      bool good = false;
      while(!good) {
        good = true;
        r_graph(i,j) = std::rand() % r_graph.extent(0);
        if(r_graph(i,j) == i) good=false;
        for(int k=0; k<j; k++) {
          if(r_graph(i,j) == r_graph(i,k)) good=false;
        }
      } 
    }
  }
}


/*
#define TEST_DEGREE 32
#define NUM_BLOCKS 128
#define BLOCK_DIM 32
template<typename T, typename accT, int Dim>
void vamana_beamSearch_test(raft::device_resources const& dev_resource,
                               raft::device_matrix_view<const T, int64_t> dataset,
                               raft::device_matrix_view<const T, int64_t> queries)
{
  using namespace cuvs::neighbors::vamana::detail;

  using IdxT = uint32_t;

  int max_batchsize = 2;
  int max_visited = 64;
  int N = dataset.extent(0);

  auto stream = raft::resource::get_cuda_stream(dev_resource);

  // create random graph
  auto r_graph = raft::make_host_matrix<IdxT, int64_t>(dataset.extent(0), TEST_DEGREE);
  create_random_graph(r_graph.view());

  auto d_graph = raft::make_device_matrix<IdxT, int64_t>(dev_resource, r_graph.extent(0), r_graph.extent(1));
  raft::copy(d_graph.data_handle(), r_graph.data_handle(), r_graph.size(), stream);

//  print_graph<<<1,1>>>(d_graph.view());
    // Temp storage about each batch of inserts being performed
  auto query_ids = raft::make_device_vector<IdxT>(dev_resource, max_batchsize);
  rmm::device_buffer query_list_ptr{
      (max_batchsize+1)*sizeof(QueryCandidates<IdxT, float>), stream, raft::resource::get_large_workspace_resource(dev_resource)};
  QueryCandidates<IdxT,accT>* query_list = static_cast<QueryCandidates<IdxT,accT>*>(query_list_ptr.data());


  // Results of each batch of inserts during build
  rmm::device_uvector<IdxT> visited_ids(
       max_batchsize*max_visited, stream, raft::resource::get_large_workspace_resource(dev_resource));
  rmm::device_uvector<accT> visited_dists(
       max_batchsize*max_visited, stream, raft::resource::get_large_workspace_resource(dev_resource));

  init_query_candidate_list<IdxT, accT><<<NUM_BLOCKS, BLOCK_DIM>>>(query_list, visited_ids.data(), visited_dists.data(), (int)max_batchsize, max_visited);

  // Create and set query id list
  std::vector<IdxT> insert_order;
  for(int j=0; j<max_batchsize; j++) {
    insert_order.push_back(j+1);
  }
  raft::copy(query_ids.data_handle(), insert_order.data(), max_batchsize, stream);
  set_query_ids<IdxT,accT><<<NUM_BLOCKS,BLOCK_DIM>>>(query_list_ptr.data(), query_ids.data_handle(), max_batchsize);

  cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Expanded;

printf("calling search kernel with %d blocks and %d threads\n", NUM_BLOCKS, BLOCK_DIM);

  GreedySearchKernel<T, accT ,Dim>
        <<<NUM_BLOCKS,BLOCK_DIM, max_visited*sizeof(DistPair<IdxT,accT>)>>>(
                         d_graph.view(),
                         raft::make_const_mdspan(dataset),
                         query_list_ptr.data(),
                         max_batchsize,
                         0,
                         TEST_DEGREE,
                         dataset.extent(0),
                         32,
                         max_visited,
                         metric,
                         max_visited);
                         
  cudaDeviceSynchronize();

  printf("Search results:\n");
  print_query_results<accT><<<1,1>>>(query_list_ptr.data(), max_batchsize);
  cudaDeviceSynchronize();

  float alpha = 1.2;

  RobustPruneKernel<T,accT,Dim>
        <<<NUM_BLOCKS,BLOCK_DIM, (TEST_DEGREE + max_visited)*sizeof(DistPair<IdxT,accT>)>>>(
                    d_graph.view(),
                    raft::make_const_mdspan(dataset),
                    query_list_ptr.data(),
                    max_batchsize,
                    TEST_DEGREE,
                    dataset.extent(0),
                    metric,
                    alpha);
  cudaDeviceSynchronize();

  printf("After RobustPrune results:\n");
  print_query_results<accT><<<1,1>>>(query_list_ptr.data(), max_batchsize);
  cudaDeviceSynchronize();

  // Write results from first prune to graph edge list
    write_graph_edges_kernel<accT, IdxT><<<NUM_BLOCKS, BLOCK_DIM>>>(
                     d_graph.view(),
                     query_list_ptr.data(),
                     TEST_DEGREE,
                     max_batchsize);
    cudaDeviceSynchronize();

// compute prefix sums of query_list sizes
   // TODO - Parallelize this prefix sums!!
   rmm::device_uvector<int> d_total_edges(
        1, stream, raft::resource::get_large_workspace_resource(dev_resource));

    prefix_sums_sizes<accT,IdxT><<<1,1>>>(query_list, max_batchsize, d_total_edges.data());
    cudaDeviceSynchronize(); // TODO -remove?

    int total_edges;
    cudaMemcpy(&total_edges, d_total_edges.data(), sizeof(IdxT), cudaMemcpyDeviceToHost);
printf("prefix sums done, total edges:%d\n", total_edges);

  rmm::device_uvector<int> edge_histogram(
       N, stream, raft::resource::get_large_workspace_resource(dev_resource));
  rmm::device_uvector<IdxT> edge_dest(
    max_batchsize*TEST_DEGREE, stream, raft::resource::get_large_workspace_resource(dev_resource));
  rmm::device_uvector<IdxT> edge_src(
    max_batchsize*TEST_DEGREE, stream, raft::resource::get_large_workspace_resource(dev_resource));

  cudaMemset(edge_histogram.data(), 0, N*sizeof(int));
  cudaDeviceSynchronize(); // TODO Remove?

// Create reverse edge list
    create_reverse_edge_list<accT,IdxT><<<NUM_BLOCKS, BLOCK_DIM>>>(
      query_list_ptr.data(), max_batchsize, TEST_DEGREE, edge_src.data(), edge_dest.data(), edge_histogram.data());
    cudaDeviceSynchronize();

  IdxT unique_dests = raft::sparse::neighbors::get_n_components(edge_dest.data(), total_edges, stream);

  // Alloc memory for sort
  size_t temp_storage_bytes = 100*max_batchsize*TEST_DEGREE*(2*sizeof(int)); // TODO - compute this size dynamically
  rmm::device_buffer temp_sort_storage{
      temp_storage_bytes, stream, raft::resource::get_large_workspace_resource(dev_resource)};

    cub::DeviceMergeSort::SortPairs(
      temp_sort_storage.data(),
      temp_storage_bytes,
      edge_dest.data(),
      edge_src.data(),
//      query_list[step_size].size,
      total_edges,
      CmpEdge<IdxT>());

    printf("After sort:%d\n", cudaDeviceSynchronize());

//  test_print_list<IdxT><<<1,1>>>(edge_src.data(), edge_dest.data(), unique_dests);

// Allocate reverse QueryCandidate list based on number of unique destinations
// TODO - If needed, perform this in batches to reduce memory needed to store rev lists
      rmm::device_buffer reverse_list_ptr{
          unique_dests*sizeof(QueryCandidates<IdxT, accT>), stream, raft::resource::get_large_workspace_resource(dev_resource)};
      rmm::device_uvector<IdxT> rev_ids(
           unique_dests*max_visited, stream, raft::resource::get_large_workspace_resource(dev_resource));
      rmm::device_uvector<accT> rev_dists(
           unique_dests*max_visited, stream, raft::resource::get_large_workspace_resource(dev_resource));

  QueryCandidates<IdxT,accT>* reverse_list = static_cast<QueryCandidates<IdxT,accT>*>(reverse_list_ptr.data());

      init_query_candidate_list<IdxT, accT><<<256, BLOCK_DIM>>>(reverse_list, rev_ids.data(), rev_dists.data(), (int)unique_dests, max_visited);

      // May need more blocks for reverse list
//      num_blocks = min(maxBlocks, unique_dests);
// Populate reverse list ids and candidate lists from edge_src and edge_dest
//      populate_reverse_list_struct<T,accT,Dim,IdxT><<<num_blocks, blockD>>>(
// TODO - Re-work these two kernels and parallelize. Maybe use sparse RAFT structs
      populate_reverse_list_struct<T,accT,Dim,IdxT><<<1,1>>>(
                     reverse_list,
        //             dataset,
                     edge_src.data(),
                     edge_dest.data(),
                     edge_histogram.data(),
                     N);

      cudaDeviceSynchronize();

      recompute_reverse_dists<T,accT,Dim,IdxT><<<NUM_BLOCKS,BLOCK_DIM>>>(
                      reverse_list,
                      dataset,
                      unique_dests);

      cudaDeviceSynchronize();


// 2nd Robust prune
// TODO - Does it need to use a separate parameter for beamsize from degree?
  RobustPruneKernel<T,accT,Dim>
        <<<NUM_BLOCKS,BLOCK_DIM, (TEST_DEGREE + max_visited)*sizeof(DistPair<IdxT,accT>)>>>(
                    d_graph.view(),
                    raft::make_const_mdspan(dataset),
                    reverse_list_ptr.data(),
                    unique_dests,
                    TEST_DEGREE,
                    dataset.extent(0),
                    metric,
                    alpha);
  cudaDeviceSynchronize();

  // Write final results out to graph
    write_graph_edges_kernel<accT, IdxT><<<NUM_BLOCKS, BLOCK_DIM>>>(
                     d_graph.view(),
                     reverse_list_ptr.data(),
                     TEST_DEGREE,
                     unique_dests);
    cudaDeviceSynchronize();

}
*/


void vamana_build_test(raft::device_resources const& dev_resources,
                               raft::device_matrix_view<const float, int64_t> dataset,
                               raft::device_matrix_view<const float, int64_t> queries)
{
  using namespace cuvs::neighbors;

  int64_t topk      = 12;
  int64_t n_queries = queries.extent(0);

  // create output arrays
  auto neighbors = raft::make_device_matrix<uint32_t>(dev_resources, n_queries, topk);
  auto distances = raft::make_device_matrix<float>(dev_resources, n_queries, topk);

  // use default index parameters
  vamana::index_params index_params;

  std::cout << "Building Vamana index (search graph)" << std::endl;
  auto index = vamana::build(dev_resources, index_params, dataset);

  std::cout << "Vamana index has " << index.size() << " vectors" << std::endl;
  std::cout << "Vamana graph has degree " << index.graph_degree() << ", graph size ["
            << index.graph().extent(0) << ", " << index.graph().extent(1) << "]" << std::endl;

  // use default search parameters
  vamana::search_params search_params;
  // search K nearest neighbors
//  cagra::search(dev_resources, search_params, index, queries, neighbors.view(), distances.view());

  // The call to cagra::search is asynchronous. Before accessing the data, sync by calling
  // raft::resource::sync_stream(dev_resources);

//  print_results(dev_resources, neighbors.view(), distances.view());
}

int main()
{
  raft::device_resources dev_resources;

  // Set pool memory resource with 1 GiB initial pool size. All allocations use the same pool.
  rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource> pool_mr(
    rmm::mr::get_current_device_resource(), 1024 * 1024 * 1024ull);
  rmm::mr::set_current_device_resource(&pool_mr);

  // Alternatively, one could define a pool allocator for temporary arrays (used within RAFT
  // algorithms). In that case only the internal arrays would use the pool, any other allocation
  // uses the default RMM memory resource. Here is how to change the workspace memory resource to
  // a pool with 2 GiB upper limit.
  // raft::resource::set_workspace_to_pool_resource(dev_resources, 2 * 1024 * 1024 * 1024ull);

  // Create input arrays.
  int64_t n_samples = 10000;
  const int64_t n_dim     = 32;
  int64_t n_queries = 10;
  auto dataset      = raft::make_device_matrix<float, int64_t>(dev_resources, n_samples, n_dim);
  auto queries      = raft::make_device_matrix<float, int64_t>(dev_resources, n_queries, n_dim);
  generate_dataset(dev_resources, dataset.view(), queries.view());

  // Simple build and search example.
  vamana_build_test(dev_resources,
                            raft::make_const_mdspan(dataset.view()),
                            raft::make_const_mdspan(queries.view()));

  printf("End build test\n");

/*
  vamana_beamSearch_test<float,float,n_dim>(dev_resources,
                            raft::make_const_mdspan(dataset.view()),
                            raft::make_const_mdspan(queries.view()));
*/
}
