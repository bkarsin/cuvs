/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#include <thrust/sort.h>
#include <cub/cub.cuh>

#include "vamana_structs.cuh"

namespace cuvs::neighbors::vamana::detail {

template <typename T, typename accT, typename IdxT = uint32_t>
__inline__ __device__ void load_candidates_to_registers(
                                      DistPair<IdxT,accT>* tmp, 
                                      QueryCandidates<IdxT,accT>* query)
{

/**** Load edge list and candidates and sort them together ****/
// TODO - This whole section needs to be generalized for different degree / beamsize
    tmp[0].idx = query->ids[2*threadIdx.x];
    tmp[0].dist = query->dists[2*threadIdx.x];
    tmp[1].idx = query->ids[2*threadIdx.x + 1];
    tmp[1].dist = query->dists[2*threadIdx.x + 1];

    if(query->size > 64) query->size = 64;
    __syncthreads();
    if(2*threadIdx.x >= query->size) { 
      tmp[0].idx = INFTY<IdxT>(); 
      tmp[0].dist = INFTY<accT>(); 
    }
    if(2*threadIdx.x+1 >= query->size) { 
      tmp[1].idx = INFTY<IdxT>(); 
      tmp[1].dist = INFTY<accT>(); 
    }
}

#define TOTAL_SIZE = 96

/********************************************************************************************
  GPU kernel for RobustPrune operation for Vamana graph creation
  Input - *graph to be an edgelist of degree number of edges per vector,
  candidate_ptr should contain the list of visited nodes during beamSearch

  Output - candidate_ptr contains the new set of *degree* new neighbors that each node
           should have.
**********************************************************************************************/
template <typename T, 
          typename accT, 
          int Dim, // TODO - generalize with selector fcn
          typename IdxT = uint32_t,
          typename Accessor = raft::host_device_accessor<std::experimental::default_accessor<T>,
                                                          raft::memory_type::host>>
__global__ void RobustPruneKernel(
     raft::device_matrix_view<IdxT, int64_t> graph, 
    raft::mdspan<const T, raft::matrix_extent<int64_t>, raft::row_major, Accessor> dataset,
     void* query_list_ptr,
     int num_queries, int degree, int n,
     cuvs::distance::DistanceType metric_ptr, float alpha) {

  QueryCandidates<IdxT,accT>* query_list = static_cast<QueryCandidates<IdxT,accT>*>(query_list_ptr);
  const T* vec_ptr = dataset.data_handle();

  typedef cub::BlockMergeSort<DistPair<IdxT,accT>, 32, 3> BlockMergeSort; // TODO - generalize for degree
  __shared__ typename BlockMergeSort::TempStorage temp_storage;

  extern __shared__ DistPair<IdxT,accT> new_nbh_list[];

  static __shared__ Point<T, accT, Dim> s_query;
  static __shared__ T s_coords[Dim];
  s_query.coords = s_coords;

  // Sort combined graph edges with candidate list, place result in shared memory
  DistPair<IdxT, accT> tmp[3];

  for(int i=blockIdx.x; i<num_queries; i+=gridDim.x) {
    int queryId = query_list[i].queryId;
//    DistPair<IdxT,accT>* cand = query_list[i].list;

    update_shared_point<T, accT, Dim>(&s_query, vec_ptr, query_list[i].queryId);

    load_candidates_to_registers<T,accT,IdxT>(tmp, &query_list[i]);

    // Load new neighbors to be sorted with candidates
    new_nbh_list[threadIdx.x].idx = graph(queryId,threadIdx.x); // Fixed at 32 degree
//if(threadIdx.x < 2) printf("thread:%d, id:%u\n", threadIdx.x, new_nbh_list[threadIdx.x].idx);

    __syncthreads();
if(i==0 && threadIdx.x==0) {
  printf("before prune: ");
  for(int j=0; j<5; j++) {
    printf("%d, ", new_nbh_list[j].idx);
  }
  printf("\n");
}
    __syncthreads();
    for(int j=0; j<degree; j++) {
      if(new_nbh_list[j].idx != INFTY<IdxT>()) {
        new_nbh_list[j].dist = l2<T,accT,Dim>(s_query.coords, &vec_ptr[new_nbh_list[j].idx*Dim]);

        // Optimized warp-wide L2 function. TODO - create cosine and selector fcn
//        new_nbh_list[j].dist = L2_opt<T,T,Dim>(&s_query, vec_ptr[new_nbh_list[j].idx].coords);
      }
      else {
        new_nbh_list[j].dist = INFTY<accT>();
      }
    }
    __syncthreads();

    tmp[2] = new_nbh_list[threadIdx.x];

    __syncthreads();
    BlockMergeSort(temp_storage).Sort(tmp,CmpDist());
    __syncthreads();

  // Mark duplicates and re-sort 
    new_nbh_list[3*threadIdx.x+2].idx = tmp[2].idx;
    new_nbh_list[3*threadIdx.x+2].dist = tmp[2].dist;
    if(tmp[2].idx == tmp[1].idx) {
      new_nbh_list[3*threadIdx.x+2].idx = INFTY<IdxT>();
      new_nbh_list[3*threadIdx.x+2].dist = INFTY<accT>();
    }
    __shfl_up_sync(0xffffffff, tmp[2].idx, 1);
    __syncthreads();

    if(tmp[1].idx == tmp[0].idx) { tmp[1].idx = INFTY<IdxT>(); tmp[1].dist = INFTY<accT>(); }
    if(threadIdx.x==0) {
      if(tmp[0].idx == tmp[2].idx) { tmp[0].idx = INFTY<IdxT>(); tmp[0].dist = INFTY<accT>(); }
    }
    tmp[2].idx = new_nbh_list[3*threadIdx.x+2].idx; // copy back to tmp for re-shuffling
    tmp[2].dist = new_nbh_list[3*threadIdx.x+2].dist; // copy back to tmp for re-shuffling

    __syncthreads();
    BlockMergeSort(temp_storage).Sort(tmp,CmpDist());
    __syncthreads();

    new_nbh_list[3*threadIdx.x].idx = tmp[0].idx;
    new_nbh_list[3*threadIdx.x].dist = tmp[0].dist;
    new_nbh_list[3*threadIdx.x+1].idx = tmp[1].idx;
    new_nbh_list[3*threadIdx.x+1].dist = tmp[1].dist;
    new_nbh_list[3*threadIdx.x+2].idx = tmp[2].idx;
    new_nbh_list[3*threadIdx.x+2].dist = tmp[2].dist;

    __syncthreads();

    // If less than degree total neighbors, don't need to prune
    if(new_nbh_list[degree].idx == INFTY<IdxT>()) {
      if(threadIdx.x==0) {
        int writeId=0;
        for(; new_nbh_list[writeId].idx != INFTY<IdxT>(); writeId++) {
//          cand[writeId] = new_nbh_list[writeId];
          query_list[i].ids[writeId] = new_nbh_list[writeId].idx;
          query_list[i].dists[writeId] = new_nbh_list[writeId].dist;
        }
        query_list[i].size = writeId;
        for(; writeId < degree; writeId++) {
          query_list[i].ids[writeId] = INFTY<IdxT>();
          query_list[i].dists[writeId] = INFTY<accT>();
        }
      }
    }
    else {

  // loop through list, writing nearest to visited_list, while nulling out violating neighbors in shared memory
      if(threadIdx.x==0) {
        query_list[i].ids[0] = new_nbh_list[0].idx;
        query_list[i].dists[0] = new_nbh_list[0].dist;
//        cand[0] = new_nbh_list[0];
      }

      __threadfence_block();

      int writeId=1;
      for(int j=1; j<degree+query_list[i].size && writeId < degree; j++) {
        __syncthreads();
        if(new_nbh_list[j].idx ==queryId || new_nbh_list[j].idx == INFTY<IdxT>()) {
          continue;
        }
        __syncthreads();
        if(threadIdx.x==0) {
          query_list[i].ids[writeId] = new_nbh_list[j].idx;
          query_list[i].dists[writeId] = new_nbh_list[j].dist;
//          cand[writeId] = new_nbh_list[j];
        }
        writeId++;
        __syncthreads();

        update_shared_point<T, accT, Dim>(&s_query, vec_ptr, new_nbh_list[j].idx);

        int tot_size = degree+query_list[i].size;
        for(int k=j+1; k<tot_size; k++) {
          T* mem_ptr = const_cast<T*>(&vec_ptr[new_nbh_list[k].idx*Dim]);
          if(new_nbh_list[k].idx != INFTY<IdxT>()) {

            accT dist_starprime = l2<T,accT,Dim>(s_query.coords, mem_ptr);
//            accT dist_starprime = GetDistanceByVec<T,accT,Dim>(&s_query, &vec_ptr[new_nbh_list[k].idx], metric_ptr[0]);
              // Optimized warp-wide L2 function. TODO - create cosine and selector fcn
//              T dist_starprime = L2_opt<T,T,Dim>(&s_query, mem_ptr); 

            if(threadIdx.x==0 && alpha * dist_starprime <= new_nbh_list[j].dist) {
//            if(threadIdx.x==0 && (int32_t)(alpha * ((float)(dist_starprime))) <= new_nbh_list[j].dist) {
              new_nbh_list[k].idx = INFTY<IdxT>();
            }
          }
        }
      }
      __syncthreads();
      if(threadIdx.x==0) {query_list[i].size = writeId;}

if(i==0 && threadIdx.x==0) {
  printf("after prune: ");
  for(int j=0; j<5; j++) {
    printf("%d, ", new_nbh_list[j].idx);
  }
  printf("\n");
}

    __syncthreads();
      for(int j=writeId+threadIdx.x; j<degree; j+=blockDim.x) { // Zero out any unfilled neighbors
        query_list[i].ids[j] = INFTY<IdxT>();
        query_list[i].dists[j] = INFTY<accT>();
      }
    }
  }
}

} // namespace
