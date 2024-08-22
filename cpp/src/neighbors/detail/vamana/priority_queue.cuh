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

#ifndef _PQ_H_
#define _PQ_H_

#include <stdio.h>
#include "vamana_structs.cuh"

namespace cuvs::neighbors::vamana::detail {


// Heap object.  Implements a max-heap of DistPair<SUMTYPE> objects.  Stores a total of KVAL pairs
//   NOTE: Currently KVAL must be 2i-1 for some integer i since the heap must be complete
template<typename IdxT, typename accT, int Dim, int BLOCK_DIM>
class PriorityQueue {
  public:
    int KVAL;
    int insert_pointer;
    DistPair<IdxT,accT>* vals;
    DistPair<IdxT,accT> temp;
  
    int* q_size;
    // Enforce max-heap property on the entires
    __forceinline__ __device__ void heapify() {
      int i=0;
      int swapDest=0;

      while(2*i+2 < KVAL) {

          swapDest = 2*i;
          swapDest += (vals[i].dist > vals[2*i+1].dist && vals[2*i+2].dist >= vals[2*i+1].dist);
          swapDest += 2*(vals[i].dist > vals[2*i+2].dist && vals[2*i+1].dist > vals[2*i+2].dist);

          if(swapDest == 2*i) return;

          swap(&vals[i], &vals[swapDest]);

          i = swapDest;
        
      }

    }


    __forceinline__ __device__ void heapifyAt(int idx) {
      int i=idx;
      int swapDest=0;

      while(2*i+2 < KVAL) {
      
          swapDest = 2*i;
          swapDest += (vals[i].dist > vals[2*i+1].dist && vals[2*i+2].dist <= vals[2*i+1].dist);
          swapDest += 2*(vals[i].dist > vals[2*i+2].dist && vals[2*i+1].dist < vals[2*i+2].dist);

          if(swapDest == 2*i) return;

          swap(&vals[i], &vals[swapDest]);

          i = swapDest;

      }

    }


    __forceinline__ __device__ void heapifyReverseAt(int idx){
      int i = idx;
      int swapDest = 0;
      while(i > 0){
        swapDest = ((i-1)/2);
        if(vals[swapDest].dist <= vals[i].dist) return;
        
        swap(&vals[i], &vals[swapDest]);
        i = swapDest;
      }

    }

    __forceinline__ __device__ void MultiHeapifyReverseAt(int idx){
      int i = idx;
      int swapDest = 0;
      while(i > 0){
        swapDest = ((i-1)/2);
        if(vals[swapDest].dist <= vals[i].dist) return;
        
        swap(&vals[i], &vals[swapDest]);
        i = swapDest;
      }

    }



    __device__ void reset() {
      *q_size = 0;
      for(int i=0; i<KVAL; i++) {
        vals[i].dist =  INFTY<accT>();
        vals[i].idx = INFTY<IdxT>();
      }
    }


    __device__ void initialize(DistPair<IdxT,accT>* v, int _kval, int* _q_size) {
      vals = v;
      KVAL = _kval;
      insert_pointer = _kval / 2;
      q_size = _q_size;
      reset();
    }



    // Initialize all nodes of the heap to +infinity
    __device__ void initialize() {
      for(int i=0; i<KVAL; i++) {
        vals[i].idx = INFTY<IdxT>();
        vals[i].dist =  INFTY<accT>();
      }
    }

    __device__ void write_to_gmem(int* gmem) {
      for(int i=0; i<KVAL; i++) {
        gmem[i] = vals[i].idx;
      }
    }


    // FOR DEBUGGING: Print heap out
    __device__ void printHeap() {

      int levelSize=0;
      int offset=0;
//      for(int level=0; offset<=KVAL/2; level++) {
      for(int level=0; offset<=4; level++) {
        levelSize = pow(2,level);
        for(int i=0; i<levelSize; i++) {
          if(vals[offset+i].idx == INFTY<IdxT>()) {printf("0 (INFTY)   ");}
          else {
            printf("%d (%0.3f)   ", vals[offset+i].idx, vals[offset+i].dist);
          }
          //printf("%llu   ", vals[offset+i].dist);

        }
        offset += levelSize;
        printf("\n");
      } 
    }


    // Replace the root of the heap with new pair
    __device__ void insert(accT newDist, IdxT newIdx) {
      vals[0].dist = newDist;
      vals[0].idx = newIdx;

      heapify();
    }

    // Replace a specific element in the heap (and maintain heap properties)
    __device__ void insertAt(accT newDist, IdxT newIdx, int idx) {
      vals[idx].dist = newDist;
      vals[idx].idx = newIdx;

      heapifyAt(idx);
    }

/*
    // Load a sorted set of vectors into the heap
    __device__ void load_mem_sorted(Point<T, SUMTYPE,Dim>* data, int* mem, Point<T,SUMTYPE,Dim> query, int metric) {
      for(int i=0; i<=KVAL-1; i++) {
        vals[(KVAL-i)-1].idx = mem[i];
        if(vals[(KVAL-i)-1].idx == -1) {
          vals[(KVAL-i)-1].dist = INFTY<SUMTYPE>();
        }
        else {
          if(metric == 0) {
            vals[(KVAL-i)-1].dist = query.l2(&data[mem[i]]);
          }
          else if(metric == 1) {
            vals[(KVAL-i)-1].dist = query.cosine(&data[mem[i]]);
          }
        }
      }
    }
*/

    // Return value of the root of the heap (largest value)
    __device__ accT top() {
      return vals[0].dist;
    }

    __device__ IdxT top_node(){
      return vals[0].idx;
    }


    __device__ void insert_back(accT newDist, IdxT newIdx){
      if(newDist < vals[insert_pointer].dist){
        if(vals[insert_pointer].idx == INFTY<IdxT>())
		*q_size += 1;
	vals[insert_pointer].dist = newDist;
        vals[insert_pointer].idx = newIdx;
        heapifyReverseAt(insert_pointer);
        
      }
        insert_pointer++;

        if(insert_pointer == KVAL)  insert_pointer = KVAL/2;

    }




    __device__ DistPair<IdxT,accT> pop(){
      DistPair<IdxT,accT> result;
      result.dist = vals[0].dist;
      result.idx = vals[0].idx;
      vals[0].dist =  INFTY<accT>();
       vals[0].idx = INFTY<IdxT>();
       heapify();    
       *q_size -= 1;         
       return result;
    }

};


/*
  Reset all values in the filer to the input value
*/
/*
__device__ void reset_filter(int *filter, const int filter_size, int val) {

  for (int i = threadIdx.x; i < filter_size; i += blockDim.x) {
    filter[i] = val;
  }
}

template<typename T>
__device__ void reset_filter(DistPair<T> *filter, const int filter_size, int val) {

  for (int i = threadIdx.x; i < filter_size; i += blockDim.x) {
    filter[i].idx = val;
  }
}
*/
/*  
  Enqueing a input value into parallel queue with tracker
*/

template <typename SUMTYPE>
__inline__ __device__ void
parallel_pq_max_enqueue(Node<SUMTYPE> *pq, int *size, const int pq_size,
                        Node<SUMTYPE> input_data, SUMTYPE *cur_max_val, int* max_idx) {

  if (*size < pq_size) {
    __syncthreads();
    if (threadIdx.x == 0) {
      pq[*size].distance = input_data.distance;
      pq[*size].nodeid = input_data.nodeid;
      *size = *size + 1;
      if (input_data.distance > (*cur_max_val)){
        *cur_max_val = input_data.distance;
        *max_idx = *size - 1;
      }
    }
    __syncthreads();
    return;
  }

  else {

    if (input_data.distance >= (*cur_max_val)) {
      __syncthreads();
      return;
    }
    if(threadIdx.x == 0) {
      pq[*max_idx].distance = input_data.distance;
      pq[*max_idx].nodeid = input_data.nodeid;
    }
    int idx = 0;
    SUMTYPE max_val = pq[0].distance;

    for (int i = threadIdx.x; i < pq_size; i += 32) {
      if (pq[i].distance > max_val) {
        max_val = pq[i].distance;
        idx = i;
      }
    }

    for (int offset = 16; offset > 0; offset /= 2) {
      SUMTYPE new_max_val = __shfl_down_sync(FULL_BITMASK, max_val, offset);
      int new_idx = __shfl_down_sync(FULL_BITMASK, idx, offset);
      if (new_max_val > max_val) {
        max_val = new_max_val;
        idx = new_idx;
      }
    }

    if (threadIdx.x == 0) {
      *max_idx = idx;
      *cur_max_val = max_val;
      // if (input_data.distance < max_val) {
      //   pq[idx].distance = input_data.distance;
      //   pq[idx].nodeid = input_data.nodeid;
      // }
    }

  }

  __syncthreads();
}

/*  
  Enqueing a input value into parallel queue without tracker
*/

template <typename SUMTYPE>
__inline__ __device__ void parallel_pq_max_enque_no_track(Node<SUMTYPE> *pq, int *size,
                                                 const int pq_size,
                                                 Node<SUMTYPE> input_data) {

  if (*size < pq_size) {
    __syncthreads();
    if (threadIdx.x == 0) {
      pq[*size].distance = input_data.distance;
      pq[*size].nodeid = input_data.nodeid;
      *size = *size + 1;
    }
    __syncthreads();
    return;
  }

  else {
   int idx = 0;
    SUMTYPE max_val = pq[0].distance;

    for (int i = threadIdx.x; i < pq_size; i += 32) {
      if (pq[i].distance > max_val) {
        max_val = pq[i].distance;
        idx = i;
      }
    }

    for (int offset = 16; offset > 0; offset /= 2) {
      SUMTYPE new_max_val = __shfl_down_sync(FULL_BITMASK, max_val, offset);
      int new_idx = __shfl_down_sync(FULL_BITMASK, idx, offset);
      if (new_max_val > max_val) {
        max_val = new_max_val;
        idx = new_idx;
      }
    }

    if (threadIdx.x == 0) {
      if (input_data.distance < max_val) {
        pq[idx].distance = input_data.distance;
       pq[idx].nodeid = input_data.nodeid;
      }
    }
  }
}

/*
  Return the highest value in the queue without poping the value
*/

template <typename SUMTYPE>
__device__ bool peek_max_value(const Node<SUMTYPE> *pq, const int size,
                               SUMTYPE *output) {
  if (size == 0)
    return false;

  int idx = 0;
  SUMTYPE max_val = pq[0].distance;

  for (int i = threadIdx.x; i < size; i += 32) {
    if (pq[i].distance > max_val) {
      max_val = pq[i].distance;
      idx = i;
    }
  }

  for (int offset = 16; offset > 0; offset /= 2) {
    SUMTYPE new_max_val = __shfl_down_sync(FULL_BITMASK, max_val, offset);
    int new_idx = __shfl_down_sync(FULL_BITMASK, idx, offset);
    if (new_max_val > max_val) {
      max_val = new_max_val;
      idx = new_idx;
    }
  }

  if (threadIdx.x == 0)
    *output = max_val;

  return true;
}

template <typename T, typename SUMTYPE, int Dim, int NUM_VEC>
__device__ void GetMultiDistanceByVec(Point<T, SUMTYPE, Dim> *src_vec,
                                      Point<T, SUMTYPE, Dim> *vec_ptr,
                                      int *vec_id_array, bool *flag,
                                      SUMTYPE *out) {

  out[0] = l2<T,SUMTYPE,Dim>(src_vec, vec_ptr+(vec_id_array[0]));

  for(int i=1; i<NUM_VEC; i++) {
    if(flag[i-1]) {
      out[i] = l2<T,SUMTYPE,Dim>(src_vec, vec_ptr+(vec_id_array[i]));
    }
    else out[i] = 0;
  }
}


/*
  Compute the distances between the soruce vector and the destinaton vectors and enque them in the priority queue
*/
template <typename T, typename accT, int Dim, typename IdxT>
__device__ void
enqueue_all_neighbors(int num_neighbors, Point<T, accT, Dim> *query_vec,
                     const T *vec_ptr, int *neighbor_array,
                     PriorityQueue<IdxT, accT, Dim, DEGREE> &heap_queue) { // TODO -dynamic degree support

// MAY NEED TO CHECK FOR DUPLICATES HERE!!!

  for (int i = 0; i < num_neighbors; i++) {

//    GetMultiDistanceByVec<T, SUMTYPE, Dim, num_dist>(
//        query_vec, vec_ptr, neighbor_array + i, flags, dist_out);
    
    accT dist_out = l2<T,accT,Dim>(query_vec->coords, &vec_ptr[neighbor_array[i]*Dim]);

    __syncthreads();
    if (threadIdx.x == 0) {
      heap_queue.insert_back(dist_out, neighbor_array[i]);
    }
    __syncthreads();
  }
}


} // namespace
#endif
