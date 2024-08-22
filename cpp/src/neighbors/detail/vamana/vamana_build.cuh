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

//#include "../../vpq_dataset.cuh"
//#include "graph_core.cuh"
#include "vamana_structs.cuh"
#include "vamana_search.cuh"
#include "robust_prune.cuh"
#include <cuvs/neighbors/vamana.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/error.hpp>
#include <raft/core/host_device_accessor.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/logger-ext.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/matrix/init.cuh>
#include <raft/sparse/neighbors/cross_component_nn.cuh>


#include <cuvs/distance/distance.hpp>
//#include <cuvs/neighbors/ivf_pq.hpp>
//#include <cuvs/neighbors/refine.hpp>

#include <rmm/resource_ref.hpp>

#include <chrono>
#include <cstdio>
#include <vector>

namespace cuvs::neighbors::vamana::detail {

/* @defgroup vamana_build_detail vamana build
 * @{
 */

static const std::string RAFT_NAME = "raft";

static const int BEAMSIZE = 64;
static const int blockD = 32;
static const int maxBlocks = 10000;


// generate random permutation of inserts - TODO improve perf
void create_insert_permutation(std::vector<uint32_t>& insert_order, uint32_t N)
{
  insert_order.resize(N);
  for(uint32_t i=0; i<N; i++) {
    insert_order[i] = i;
  }
  for(uint32_t i=0; i<N; i++) {
    uint32_t temp;
    uint32_t rand_idx = rand()%N;
    temp = insert_order[i];
    insert_order[i] = insert_order[rand_idx];
    insert_order[rand_idx] = temp;
  }
}

template<typename IdxT>
__global__ void memset_graph(raft::device_matrix_view<IdxT, int64_t> graph) {
  for(int i = blockIdx.x; i<graph.extent(0); i+=gridDim.x) {
    for(int j=threadIdx.x; j<graph.extent(1); j+=blockDim.x) {
      graph(i,j) = INFTY<IdxT>();
    }
  }
}

template<typename T,
         typename accT,
         int Dim,
         typename IdxT = uint32_t,
         typename Accessor = raft::host_device_accessor<std::experimental::default_accessor<T>,
                                                         raft::memory_type::host>>
void batched_insert_vamana(
  raft::resources const& res,
  const index_params& params,
  raft::mdspan<const T, raft::matrix_extent<int64_t>, raft::row_major, Accessor> dataset,
  raft::host_matrix_view<IdxT, int64_t> graph,
  cuvs::distance::DistanceType metric)
{
  auto stream = raft::resource::get_cuda_stream(res);
  int N = dataset.extent(0);
  int degree = graph.extent(1);
  int max_batchsize = (int)(params.max_batchsize*(float)N);

  // TODO - add params
//  int insert_iters = (int)(params.num_iters);
  int insert_iters = 1;
  double base = 2;

  int max_visited = params.max_visited;
  int queue_size = BEAMSIZE; // TODO -verify this is correct

  // create gpu graph and set to all -1s
  auto d_graph = raft::make_device_matrix<IdxT, int64_t>(res, graph.extent(0), graph.extent(1));
  // TODO replace kernel below with a memset or something slick
  memset_graph<IdxT><<<256,blockD>>>(d_graph.view());

  // Temp storage about each batch of inserts being performed
// TODO - Remove these upfront conservative allocations and move to dynamic alloc/dealloc with RMM within the loop
  auto query_ids = raft::make_device_vector<IdxT>(res, max_batchsize);
  rmm::device_buffer query_list_ptr{
      (max_batchsize+1)*sizeof(QueryCandidates<IdxT, accT>), stream, raft::resource::get_large_workspace_resource(res)};
  QueryCandidates<IdxT,accT>* query_list = static_cast<QueryCandidates<IdxT,accT>*>(query_list_ptr.data());

  
  // Results of each batch of inserts during build
  rmm::device_uvector<IdxT> visited_ids(
       max_batchsize*max_visited, stream, raft::resource::get_large_workspace_resource(res));
  rmm::device_uvector<accT> visited_dists(
       max_batchsize*max_visited, stream, raft::resource::get_large_workspace_resource(res));

  init_query_candidate_list<IdxT, accT><<<256, blockD>>>(query_list, visited_ids.data(), visited_dists.data(), (int)max_batchsize, max_visited);

  rmm::device_uvector<int> edge_histogram(
       N, stream, raft::resource::get_large_workspace_resource(res));

// TODO - These especially should be dynamically created/destroyed with RMM during loop

  // Create random permutation for order of node inserts into graph
  std::vector<uint32_t> insert_order;
  create_insert_permutation(insert_order, (uint32_t)N);

  // Memory needed to sort reverse edges
  rmm::device_uvector<IdxT> edge_dest(
    max_batchsize*degree, stream, raft::resource::get_large_workspace_resource(res));
  rmm::device_uvector<IdxT> edge_src(
    max_batchsize*degree, stream, raft::resource::get_large_workspace_resource(res));

  // TODO - figure out how to avoid this memory overhead for sorting
  size_t temp_storage_bytes = max_batchsize*degree*(2*sizeof(int));
  RAFT_LOG_INFO("Temp storage needed for sorting (bytes): %lu", temp_storage_bytes);
  rmm::device_buffer temp_sort_storage{
      temp_storage_bytes, stream, raft::resource::get_large_workspace_resource(res)};
  

  // Number of passes over dataset (default 1)
  for(int iter=0; iter < insert_iters; iter++) {
  
  // Loop through batches and call the insert and prune kernels

N=4; // FOR TESTING
    int step_size=1;
    for(int start=0; start < N; ) {

//      auto step_start_t = std::chrono::system_clock::now();

      if(start+step_size > N) {
        int new_size = N - start;
        step_size = new_size;
      }

      int num_blocks = min(maxBlocks, step_size);

      // Copy ids to be inserted for this batch
      raft::copy(query_ids.data_handle(), &insert_order.data()[start], step_size, stream);
      set_query_ids<IdxT,accT><<<num_blocks,blockD>>>(query_list_ptr.data(), query_ids.data_handle(), step_size);
      cudaDeviceSynchronize();



      RAFT_LOG_INFO("start:%d, step_size:%d\n", start, step_size);
      size_t free, total;
      cudaMemGetInfo(&free, &total);
      RAFT_LOG_INFO("Device memory - free:%ld, total:%ld\n", free, total);

      int medoid_id = 0; // TODO - create medoid selector kernel/fcn

printf("Queries being inserted: [");
for(int j=0; j<step_size; j++) {
  printf("%d, ", insert_order.data()[start+j]);
}
printf("\n");

//print_graph<<<1,1>>>(d_graph.view());

      // TODO - selector for dimension or remove from template
      GreedySearchKernel<T, accT, Dim, IdxT, Accessor>
          <<<num_blocks, blockD, max_visited*sizeof(Node<accT>)>>>(
//          <<<1,blockD, max_visited*sizeof(DistPair<IdxT,accT>)>>>(
                     d_graph.view(), 
                     dataset, 
                     query_list_ptr.data(), 
                     step_size, 
                     medoid_id,
                     degree, 
                     dataset.extent(0), 
                     max_visited,
                     queue_size, //hash_table_size,
                     metric, 
                     max_visited);

      cudaDeviceSynchronize();

  float alpha = 1.2; // TODO - add param
        // Run robustPrune
      RobustPruneKernel<T,accT,Dim, IdxT>
          <<<num_blocks, blockD, (degree + max_visited)*sizeof(DistPair<IdxT,accT>)>>>(
                     d_graph.view(), 
                     dataset,
                     query_list_ptr.data(),
                      step_size,
                     degree,
                     dataset.extent(0),
                     metric,
                     alpha);

      cudaDeviceSynchronize(); // TODO Remove?

// Write results from first prune to graph edge list
      write_graph_edges_kernel<accT, IdxT><<<num_blocks, blockD>>>(
                     d_graph.view(), 
                     query_list_ptr.data(), 
                     degree, 
                     step_size);
      cudaDeviceSynchronize();

//  printf("After prune:\n");
//  print_query_results<accT><<<1,1>>>(query_list_ptr.data(), step_size);
//  printf("C - %d\n", cudaDeviceSynchronize());

// compute prefix sums of query_list sizes
   // TODO - Parallelize this prefix sums!!
      rmm::device_uvector<int> d_total_edges(
          1, stream, raft::resource::get_large_workspace_resource(res));

      prefix_sums_sizes<accT,IdxT><<<1,1>>>(query_list, step_size, d_total_edges.data());
      cudaDeviceSynchronize(); // TODO -remove?

      int total_edges;
// TODO - replace with raft::copy commands or avoid if possible
      cudaMemcpy(&total_edges, d_total_edges.data(), sizeof(IdxT), cudaMemcpyDeviceToHost);
  
      cudaMemset(edge_histogram.data(), 0, dataset.extent(0)*sizeof(int));
      cudaDeviceSynchronize(); // TODO Remove?

// Create reverse edge list
      create_reverse_edge_list<accT,IdxT><<<num_blocks,blockD>>>(
                     query_list_ptr.data(), 
                     step_size, 
                     degree, 
                     edge_src.data(), 
                     edge_dest.data(), 
                     edge_histogram.data());
      cudaDeviceSynchronize();

      cub::DeviceMergeSort::SortPairs(
                     temp_sort_storage.data(),
                     temp_storage_bytes,
                     edge_dest.data(),
                     edge_src.data(),
                     total_edges,
                     CmpEdge<IdxT>());
      cudaDeviceSynchronize();

// Get number of unique node destinations
     IdxT unique_dests = raft::sparse::neighbors::get_n_components(edge_dest.data(), total_edges, stream);

//test_print_list<IdxT><<<1,1>>>(edge_src.data(), edge_dest.data(), total_edges);
//      cudaDeviceSynchronize();

printf("reverse edges - total:%d, unique dests:%u\n", total_edges, unique_dests);

// Allocate reverse QueryCandidate list based on number of unique destinations
// TODO - If needed, perform this in batches to reduce memory needed to store rev lists
      rmm::device_buffer reverse_list_ptr{
          unique_dests*sizeof(QueryCandidates<IdxT, accT>), stream, raft::resource::get_large_workspace_resource(res)};
      rmm::device_uvector<IdxT> rev_ids(
           unique_dests*max_visited, stream, raft::resource::get_large_workspace_resource(res));
      rmm::device_uvector<accT> rev_dists(
           unique_dests*max_visited, stream, raft::resource::get_large_workspace_resource(res));

  QueryCandidates<IdxT,accT>* reverse_list = static_cast<QueryCandidates<IdxT,accT>*>(reverse_list_ptr.data());

      init_query_candidate_list<IdxT, accT><<<256, blockD>>>(reverse_list, rev_ids.data(), rev_dists.data(), (int)unique_dests, max_visited);

printf("init reverse list - %d\n", cudaDeviceSynchronize());
      // May need more blocks for reverse list
      num_blocks = min(maxBlocks, unique_dests);
// Populate reverse list ids and candidate lists from edge_src and edge_dest
//      populate_reverse_list_struct<T,accT,Dim,IdxT><<<num_blocks, blockD>>>(
// TODO - Re-work these two kernels and parallelize. Maybe use sparse RAFT structs
      populate_reverse_list_struct<T,accT,Dim,IdxT><<<1,1>>>(
                     reverse_list,
        //             dataset,
                     edge_src.data(),
                     edge_dest.data(),
                     edge_histogram.data(),
                     total_edges,
                     dataset.extent(0));
           
      printf("populate reverse list - %d\n", cudaDeviceSynchronize());
//printf("B\n");
print_query_results<accT><<<1,1>>>(reverse_list_ptr.data(), unique_dests);
      cudaDeviceSynchronize();

      recompute_reverse_dists<T,accT,Dim,IdxT><<<num_blocks, blockD>>>(
                      reverse_list,
                      dataset,
                      unique_dests);
                       
      printf("recompute reverse dists - %d\n", cudaDeviceSynchronize());
//print_query_results<accT><<<1,1>>>(reverse_list_ptr.data(), unique_dests);
  

// Call 2nd RobustPrune on reverse query_list
  // TODO - Does it need to use a separate parameter for beamsize from degree?
      RobustPruneKernel<T,accT,Dim>
//            <<<num_blocks, blockD, (degree + max_visited)*sizeof(DistPair<IdxT,accT>)>>>(
            <<<1, blockD, (degree + max_visited)*sizeof(DistPair<IdxT,accT>)>>>(
                      d_graph.view(),
                      raft::make_const_mdspan(dataset),
                      reverse_list_ptr.data(),
                      unique_dests,
                      degree,
                      dataset.extent(0),
                      metric,
                      alpha);
      printf("robustprune 2 - %d\n", cudaDeviceSynchronize());


// Write new edge lists to graph
      write_graph_edges_kernel<accT, IdxT><<<num_blocks, blockD>>>(
                      d_graph.view(),
                      reverse_list_ptr.data(),
                      degree,
                      unique_dests);
      printf("write_graph_edges - %d\n", cudaDeviceSynchronize());


      start += step_size;
      step_size *= base;
      if(step_size > max_batchsize) step_size = max_batchsize;

    } // Batch of inserts

printf("Final graph:\n");
print_graph<<<1,1>>>(d_graph.view());

  } // insert iters

}


template <typename T,
          typename IdxT     = uint32_t,
          typename Accessor = raft::host_device_accessor<std::experimental::default_accessor<T>,
                                                         raft::memory_type::host>>
index<T, IdxT> build(
  raft::resources const& res,
  const index_params& params,
  raft::mdspan<const T, raft::matrix_extent<int64_t>, raft::row_major, Accessor> dataset)
{
  size_t max_batchsize = params.max_batchsize;
  size_t graph_degree        = params.graph_degree;
  if (max_batchsize >= static_cast<size_t>(dataset.extent(0))) {
    RAFT_LOG_WARN(
      "Max batch insert size cannot be larger than dataset size, reducing it to %lu",
      dataset.extent(0));
    max_batchsize = dataset.extent(0) - 1;
  }
  // TODO - any other correctness checks for params

  if (params.compression.has_value()) {
    // Quantizaiton not yet supported for vamana
    RAFT_LOG_WARN("Quantization not yet supported for Vamana");
  }

  RAFT_LOG_INFO("Creating empty graph structure");
  auto vamana_graph = raft::make_host_matrix<IdxT, int64_t>(dataset.extent(0), graph_degree);

  RAFT_LOG_INFO("Running Vamana batched insert algorithm");

  cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Expanded;
  batched_insert_vamana<T, float, 32, IdxT, Accessor>(res, params, dataset, vamana_graph.view(), metric);

  try {
    return index<T, IdxT>(res, params.metric, dataset, raft::make_const_mdspan(vamana_graph.view()));
  } catch (std::bad_alloc& e) {
    RAFT_LOG_DEBUG("Insufficient GPU memory to construct CAGRA index with dataset on GPU");
    // We just add the graph. User is expected to update dataset separately (e.g allocating in
    // managed memory).
  } catch (raft::logic_error& e) {
    // The memory error can also manifest as logic_error.
    RAFT_LOG_DEBUG("Insufficient GPU memory to construct CAGRA index with dataset on GPU");
  }
  index<T, IdxT> idx(res, params.metric);
  RAFT_LOG_WARN("Constructor not called, returning empty index");
  return idx;
}

/**
 * @}
 */

}  // namespace cuvs::neighbors::vamana::detail
