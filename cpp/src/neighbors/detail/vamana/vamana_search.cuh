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
#include "priority_queue.cuh"
#include <cuvs/neighbors/vamana.hpp>
#include <cub/cub.cuh>

/*
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/error.hpp>
#include <raft/core/host_device_accessor.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/logger-ext.hpp>
#include <raft/core/resource/cuda_stream.hpp>
*/

#include <cuvs/distance/distance.hpp>
//#include <cuvs/neighbors/ivf_pq.hpp>
//#include <cuvs/neighbors/refine.hpp>

#include <rmm/resource_ref.hpp>

#include <chrono>
#include <cstdio>
#include <vector>

namespace cuvs::neighbors::vamana::detail {

#define CANDIDATE_QUEUE_SIZE 127


/* @defgroup vamana_search_detail vamana search
 * @{
 */



/*
  Using static size bloom filter to check whether the node is visited before or
  not.
*/

/*
template<typename IdxT, typename accT>
__inline__ __device__ bool CheckVisited(IdxT *visited_ids, accT visited_dists,
                                        int target, accT dist,
                                        const int visited_table_size,
                                        const int visited_list_size) {
  __syncthreads();
  bool ret = false;
  if (visited_cnt < visited_list_size) {
    int idx = target % visited_table_size;

    if (visited_table[idx] != target) {
      __syncthreads();
      if (threadIdx.x == 0) {
        if (visited_table[idx] == -1) {
          visited_table[idx] = target;
          visited_list[visited_cnt].idx = target;
          visited_list[visited_cnt++].dist = dist;
//          visited_cnt++;
        }
      }
    } else {
      ret = true;
    }
  }
  __syncthreads();
  return ret;
}
*/
/*
template <typename T, 
          typename accT, 
          typename IdxT = uint32_t>
__global__ void init_mem(raft::device_matrix_view<IdxT> cand_ids,
                         raft::device_matrix_view<accT> cand_dists,
                         raft::device_matrix_view<IdxT> topm_ids,
                         raft::device_matrix_view<accT> topm_dists,
                         raft::device_matrix_view<IdxT> visited) 
{
  int N = cand_ids.extent(0);
  for(int i=blockIdx.x; i<N; i+=gridDim.x) {
    for(int j=threadIdx.x; j<cand_ids.extent(1); j+=blockDim.x) {
      cand_ids(i,j) = INFTY<IdxT>();
      cand_dists(i,j) = INFTY<accT>();
    }
    for(int j=threadIdx.x; j<topm_ids.extent(1); j+=blockDim.x) {
      topm_ids(i,j) = INFTY<IdxT>();
      topm_dists(i,j) = INFTY<accT>();
    }
    for(int j=threadIdx.x; j<visited.extent(1); j+=blockDim.x) {
      visited(i,j) = INFTY<IdxT>();
    }
  }
}
*/

/*
#define BEAMSIZE 64

template <typename T, 
          typename accT, 
          int Dim, // TODO - generalize with selector fcns for Dim ranges
          typename IdxT = uint32_t,
          typename Accessor = raft::host_device_accessor<std::experimental::default_accessor<T>,
                                                          raft::memory_type::host>>
void GreedySearchSelect(
    raft::device_resource const& dev_resource,
    raft::device_matrix_view<IdxT, int64_t> graph,
    raft::mdspan<const T, raft::matrix_extent<int64_t>, raft::row_major, Accessor> dataset,
    void* query_list_ptr,
    int num_queries, int medoid_id, int degree, int n, int M,
    cuvs::distance::DistanceType metric, int max_visited) {

  // allocate memory for candidate and topM lists
  auto cand_ids = raft::make_device_matrix<IdxT>(dev_resource, num_queries, BEAMSIZE + degree);
  auto cand_dists = raft::make_device_matrix<accT>(dev_resource, num_queries, BEAMSIZE + degree);

  auto topm_ids = raft::make_device_matrix<IdxT>(dev_resource, num_queries, M);
  auto topm_dists = raft::make_device_matrix<accT>(dev_resource, num_queries, M);
  accT topm_max;

  auto visited = raft::make_device_matrix<IdxT>(dev_resource, num_queries, max_visited);

  init_mem<T,accT,IdxT><<<num_blocks, blockD>>>(cand_ids.view(), cand_dists.view(), topm_ids.view(), topm_dists.view(), visited.view());

  auto visited_cnt = raft::make_host_vector<uint32_t>(num_queries);
  for(int i=0; i<num_queries; i++) { // look for better way to 0 out
    visited_cnt(i) = 0;
  }

  

  pairwise_distance(dev_resource,
                    medoid_mtx.view(),
                    queri
                    
  
  
}
*/


/*
  GPU kernel for Graph-based ANN searching algorithm 
*/
template <typename T, 
          typename accT, 
          int Dim, // TODO - generalize with selector fcns for Dim ranges
          typename IdxT = uint32_t,
          typename Accessor = raft::host_device_accessor<std::experimental::default_accessor<T>,
                                                          raft::memory_type::host>>
__global__ void GreedySearchKernel(
    raft::device_matrix_view<IdxT, int64_t> graph,
    raft::mdspan<const T, raft::matrix_extent<int64_t>, raft::row_major, Accessor> dataset,
    void* query_list_ptr,
    int num_queries, int medoid_id, int degree, int n, int topk,
    int q_size,// raft::device_matrix_view<int, int64_t> visited_table_ptr,
//    const int visited_table_size,
    cuvs::distance::DistanceType metric, int min_visiting_node) {
    
  QueryCandidates<IdxT,accT>* query_list = static_cast<QueryCandidates<IdxT,accT>*>(query_list_ptr);
  const T* vec_ptr = dataset.data_handle();
//  QueryCandidates<T>* topk_pq = static_cast<DistPair<accT>*>(topk_pq_ptr);
//  QueryCandidates<T>* cand_pq = static_cast<DistPair<accT>*>(cand_pq_ptr);

  static __shared__ int topk_q_size;
  static __shared__ int cand_q_size;
  static __shared__ accT cur_k_max;
  static __shared__ int k_max_idx;

//  static __shared__ int visited_cnt;
  static __shared__ Point<T, accT, Dim> s_query;
  static __shared__ T s_coords[Dim];
  s_query.coords = s_coords;

  static __shared__ int neighbor_array[DEGREE];

  static __shared__ DistPair<IdxT, accT> sharememory[CANDIDATE_QUEUE_SIZE];
  PriorityQueue<IdxT, accT, Dim, DEGREE> heap_queue;

  if (threadIdx.x == 0)
    heap_queue.initialize(&((DistPair<IdxT, accT> *)sharememory)[0],
                          CANDIDATE_QUEUE_SIZE, &cand_q_size);

  static __shared__ int num_neighbors;

  extern __shared__ Node<accT> topk_pq[];
//  __shared__ Node<accT> topk_pq[64];

//  int *visited_table = visited_table_ptr.data_handle() + visited_table_size * blockIdx.x;
//  DistPair<T> *visited_list = visited_list_ptr + max_visited_list * blockIdx.x;

//  Metric metric = metric_ptr[0];

  typedef cub::BlockMergeSort<DistPair<IdxT,accT>, 32, 2> BlockMergeSort;
  __shared__ typename BlockMergeSort::TempStorage temp_storage;

  for (int i = blockIdx.x; i<num_queries; i+=gridDim.x) {
    __syncthreads();

//    bool write_flag = false;
//    int visited_node_count = 0;

    //resetting visited list
    query_list[i].reset();
//    reset_filter(visited_table, visited_table_size, -1);
//    reset_filter(query_list[i].list, query_list[i].maxSize, -1);


    //storing the current query vector into shared memory
    update_shared_point<T, accT, Dim>(&s_query, vec_ptr, query_list[i].queryId);

    if (threadIdx.x == 0) {
      topk_q_size = 0;
      cand_q_size = 0;
//      visited_cnt = 0;
      s_query.id = query_list[i].queryId;
      cur_k_max = 0;
      k_max_idx = 0;
      heap_queue.reset();
    }

    __syncthreads();

    Point<T, accT, Dim> *query_vec;

   // Just start from medoid every time, rather than multiple set_ups
    query_vec = &s_query;
//    Point<T, accT, Dim>* medoid = vec_ptr + medoid_id;
    const T* medoid = &vec_ptr[medoid_id*Dim];
//    accT medoid_dist = GetDistanceByVec<T,accT,Dim>(query_vec, medoid, metric);
    accT medoid_dist = l2<T,accT,Dim>(query_vec->coords, medoid);

    if(threadIdx.x==0) {
      heap_queue.insert_back(medoid_dist, medoid_id);
    }
    __syncthreads();

    while (cand_q_size != 0) {
      __syncthreads();

      int cand_num;
      accT cur_distance;
      if (threadIdx.x == 0) {
        Node<accT> test_cand;
        DistPair<IdxT,accT> test_cand_out = heap_queue.pop();
        test_cand.distance = test_cand_out.dist;
        test_cand.nodeid = test_cand_out.idx;
        cand_num = test_cand.nodeid;
        cur_distance = test_cand_out.dist;

      }
__syncthreads();

      cand_num = __shfl_sync(FULL_BITMASK, cand_num, 0);

      __syncthreads();

//      if (CheckVisited<accT>(visited_table, query_list[i].list, visited_cnt, cand_num, cur_distance,
//                       visited_table_size, query_list[i].maxSize)) {

      if(query_list[i].check_visited(cand_num, cur_distance)) {
        continue;
      }

      cur_distance = __shfl_sync(FULL_BITMASK, cur_distance, 0);

      //stop condidtion for the graph traversal process
      bool done = false;
      bool pass_flag = false;

      if (topk_q_size == topk) {

        //Check the current node with the worst candidate in top-k queue
        if (threadIdx.x == 0){
          if (cur_k_max <= cur_distance){
            done = true;
          }
        }

        done = __shfl_sync(FULL_BITMASK, done, 0);
        if (done) {
          if(query_list[i].size < min_visiting_node) {
//          if (visited_node_count < min_visiting_node) {
            pass_flag = true;
          }

          else if (query_list[i].size >= min_visiting_node) {
//            write_flag = true;
            break;
          }
        }
      }

      //The current node is closer to the query vector than the worst candidate in top-K queue, so 
      //enquee the current node in top-k queue
//      visited_node_count += 1;
      Node<accT> new_cand;
      new_cand.distance = cur_distance;
      new_cand.nodeid = cand_num;

      if (check_duplicate(topk_pq, topk_q_size, new_cand) == false) {
        if (!pass_flag) {
          parallel_pq_max_enqueue<accT>(topk_pq, &topk_q_size, topk,
                                           new_cand, &cur_k_max, &k_max_idx);

          __syncthreads();
        }
      } else {
        // already visited
        continue;
      }

      num_neighbors=DEGREE;
      __syncthreads();

      for (size_t j = threadIdx.x; j < DEGREE; j += blockDim.x) {
        //Load 32 neighbors from the graph array and store them in neighbor array (shared memory)
//          neighbor_array[j] = graph[(size_t)(cand_num) * (size_t)(DEGREE) + (size_t)(j)];
          neighbor_array[j] = graph(cand_num, j);
          if(neighbor_array[j] == INFTY<IdxT>()) atomicMin(&num_neighbors, (int)j); // warp-wide min to find the number of neighbors
      }

        //computing distances between the query vector and 32 neighbor vectors then sequentially enqueue in priority queue.
      enqueue_all_neighbors<T, accT, Dim,IdxT>(num_neighbors, query_vec, vec_ptr,
                                              neighbor_array,
                                              heap_queue);
      __syncthreads();

    } // End cand_q_size != 0 loop

    bool self_found=false;
    for(int j=threadIdx.x; j<query_list[i].size; j+=blockDim.x) {
      if(query_list[i].ids[j] == query_vec->id) {
        query_list[i].dists[j] = INFTY<accT>();
        query_list[i].ids[j] = INFTY<IdxT>();
        self_found = true;
      }
    }

    for(int j=query_list[i].size + threadIdx.x; j<query_list[i].maxSize; j+=blockDim.x) {
      query_list[i].ids[j] = INFTY<IdxT>();
      query_list[i].dists[j] = INFTY<accT>();
    }

    __syncthreads();
    if(self_found) query_list[i].size--;

// Logic based on fixed degree, TODO - generalize for larger degree graphs
    DistPair<IdxT, accT> tmp[2];
    tmp[0].idx = query_list[i].ids[2*threadIdx.x];
    tmp[0].dist = query_list[i].dists[2*threadIdx.x];
    tmp[1].idx = query_list[i].ids[2*threadIdx.x+1];
    tmp[1].dist = query_list[i].dists[2*threadIdx.x+1];

    __syncthreads();
    BlockMergeSort(temp_storage).Sort(tmp, CmpDist());
    __syncthreads();

    query_list[i].ids[2*threadIdx.x] = tmp[0].idx;
    query_list[i].dists[2*threadIdx.x] = tmp[0].dist;
    query_list[i].ids[2*threadIdx.x+1] = tmp[1].idx;
    query_list[i].dists[2*threadIdx.x+1] = tmp[1].dist;

//    query_list[i].size = visited_cnt;

    __syncthreads();

//    if(threadIdx.x==0) {
//      query_list[i].print_visited();
//    }
  }

  return;
  
}

/**
 * @}
 */

}  // namespace cuvs::neighbors::vamana::detail
