#ifndef _ANN_DISTANCE_H_
#define _ANN_DISTANCE_H_

#include<cstdint>
#include<vector>
#include<climits>
#include<float.h>
#include<unordered_set>
#include<cstdio>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/error.hpp>
#include <raft/core/host_device_accessor.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/logger-ext.hpp>
#include <raft/core/resource/cuda_stream.hpp>

/***************************************************************************************
* This file contains single-threaded distance calculations not currently used,
* but is kept for testing with future optimizations.
***************************************************************************************/

namespace cuvs::neighbors::vamana::detail {

/* @defgroup vamana_structures vamana structures
 * @{
 */

#define FULL_BITMASK 0xFFFFFFFF

#define DEGREE 32 // TODO generalize!

// Object used to store id,distance combination for KNN graph
template<typename IdxT, typename accT>
struct __align__(16) DistPair {
//  public:
    accT dist;
    IdxT idx;

  __device__ __host__ DistPair& operator=( const DistPair& other ) {
    dist = other.dist;
    idx = other.idx;
    return *this;
  }

  __device__ __host__ DistPair& operator=( const volatile DistPair& other ) {
    dist = other.dist;
    idx = other.idx;
    return *this;
  }
};

// Swap the values of two DistPair<SUMTYPE> objects
template<typename IdxT, typename accT>
__device__ __host__ void swap(DistPair<IdxT, accT>* a, DistPair<IdxT, accT>* b) {
        DistPair<IdxT, accT> temp;
        temp.dist = a->dist;
        temp.idx = a->idx;
        a->dist = b->dist;
        a->idx=b->idx;
        b->dist=temp.dist;
        b->idx=temp.idx;

}


// TODO - remove and fix, redundant with DistPair
template <typename SUMTYPE> class __align__(16) Node {
public:
  SUMTYPE distance;
  int nodeid;
};

// Structure to sort by distance
struct CmpDist
{
  template<typename IdxT, typename accT>
  __device__ bool operator()(const DistPair<IdxT, accT> &lhs, const DistPair<IdxT, accT> &rhs) {
    return lhs.dist < rhs.dist;
  }
};

// Used to sort reverse edges by destination
template<typename IdxT>
struct CmpEdge
{
  __device__ bool operator()(const IdxT &lhs, const IdxT &rhs) {
    return lhs < rhs;
  }
};


/*********************************************************************
* Object representing a Dim-dimensional point, with each coordinate
* represented by a element of datatype T
* NOTE: Dim must be templated so that we can store coordinate values in registers
*********************************************************************/
template<typename T, typename SUMTYPE, int Dim>
class Point {
  public:
    int id;
//    T coords[Dim];
    T* coords;

 __host__ void load(std::vector<T> data) {
    for(int i=0; i<Dim; i++) {
      coords[i] = data[i];
    }
  }

  __host__ void loadChunk(T* data, int exact_dim) {
    for(int i=0; i<exact_dim; i++) {
      coords[i] = data[i];
    }
    for(int i=exact_dim; i<Dim; i++) {
      coords[i] = 0;
    }
  }

  __host__ __device__ Point& operator=( const Point& other ) {
    for(int i=0; i<Dim; i++) {
      coords[i] = other.coords[i];
    }
    id = other.id;
    return *this;
  }

  // Computes euclidean dist.  Uses 2 registers to increase pipeline efficiency and ILP
  __device__ __host__ SUMTYPE l2(Point<T,SUMTYPE,Dim>* other) {
    SUMTYPE total=0;
    for(int i=0; i<Dim; i++) {
      total += (coords[i]-other->coords[i])*(coords[i]-other->coords[i]);
    }
    return total;
  }

  // Computes Cosine dist.  Uses 2 registers to increase pipeline efficiency and ILP
  // Assumes coordinates are normalized so each vector is of unit length.  This lets us
  // perform a dot-product instead of the full cosine distance computation.
//  __device__ SUMTYPE cosine(Point<T,SUMTYPE,Dim>* other, bool test) {return NULL;}
  __device__ SUMTYPE cosine(Point<T,SUMTYPE,Dim>* other) {
    SUMTYPE total[2]={0,0};

    for(int i=0; i<Dim; i+=2) {
      total[0] += ((SUMTYPE)((SUMTYPE)coords[i] * (SUMTYPE)other->coords[i]));
      total[1] += ((SUMTYPE)((SUMTYPE)coords[i+1] * (SUMTYPE)other->coords[i+1]));
    }
    return (SUMTYPE)1.0 - (total[0]+total[1]);
  }

  __forceinline__ __device__ SUMTYPE dist(Point<T,SUMTYPE,Dim>* other, int metric) {
    if(metric == 0) return l2(other);
    else return cosine(other);
  }
};

// Less-than operator between two points.
template<typename T, typename SUMTYPE, int Dim>
__host__ __device__ bool operator<(const Point<T,SUMTYPE,Dim>& first, const Point<T,SUMTYPE,Dim>& other) {
  return first.id < other.id;
}



// Less-than operator between two Nodes.
template <typename SUMTYPE>
__host__ __device__ bool operator<(const Node<SUMTYPE> &first,
                                   const Node<SUMTYPE> &other) {
  return first.distance < other.distance;
}

// Less-than operator between two Nodes.
template <typename SUMTYPE>
__host__ __device__ bool operator>(const Node<SUMTYPE> &first,
                                   const Node<SUMTYPE> &other) {
  return first.distance > other.distance;
}

template <typename T, typename SUMTYPE, int Dim>
__device__ SUMTYPE l2(Point<T, SUMTYPE, Dim> *src_vec,
                      Point<T, SUMTYPE, Dim> *dst_vec) {

  SUMTYPE sum = 0;
  for(int i=threadIdx.x; i<Dim; i += blockDim.x) {
    sum += (src_vec->coords[i] - dst_vec->coords[i]) * (src_vec->coords[i] - dst_vec->coords[i]);
  }

  for (int offset = 16; offset > 0; offset /= 2) {
    sum += __shfl_down_sync(FULL_BITMASK, sum, offset);
  }

  return sum;
}

/*
template <typename T, typename SUMTYPE, int Dim>
__device__ SUMTYPE l2(Point<T, SUMTYPE, Dim> *src_vec,
                      Point<T, SUMTYPE, Dim> *dst_vec) {
//  SUMTYPE partial_sum[4] = {0,0,0,0};
//  T temp_src[4] = {0,0,0,0};
  T temp_dst[4] = {0,0,0,0};
  SUMTYPE partial_sum[4] = {0,0,0,0};
  for (int i = threadIdx.x; i < Dim; i += 4*blockDim.x) {

    temp_dst[0] = dst_vec[0].coords[i];
    temp_dst[1] = dst_vec[0].coords[i+32];
    temp_dst[2] = dst_vec[0].coords[i+64];
    temp_dst[3] = dst_vec[0].coords[i+96];

    partial_sum[0] = fmaf((src_vec[0].coords[i] - temp_dst[0]), (src_vec[0].coords[i] - temp_dst[0]), partial_sum[0]);
    partial_sum[1] = fmaf((src_vec[0].coords[i+32] - temp_dst[1]), (src_vec[0].coords[i+32] - temp_dst[1]), partial_sum[1]);
    partial_sum[2] = fmaf((src_vec[0].coords[i+64] - temp_dst[2]), (src_vec[0].coords[i+64] - temp_dst[2]), partial_sum[2]);
    partial_sum[3] = fmaf((src_vec[0].coords[i+96] - temp_dst[3]), (src_vec[0].coords[i+96] - temp_dst[3]), partial_sum[3]);
  }
  partial_sum[0] += partial_sum[1] + partial_sum[2] + partial_sum[3];


  for (int offset = 16; offset > 0; offset /= 2) {
    partial_sum[0] += __shfl_down_sync(FULL_BITMASK, partial_sum[0], offset);
  }

  return partial_sum[0];
}
*/

template<typename T, typename SUMTYPE, int Dim>
__host__ __device__ SUMTYPE l2(const T* src, const T* dest) {
  Point<T, SUMTYPE, Dim> src_p;
  src_p.coords = const_cast<T*>(src);
  Point<T, SUMTYPE, Dim> dest_p;
  dest_p.coords = const_cast<T*>(dest);

  return l2<T,SUMTYPE,Dim>(&src_p, &dest_p);
}

template <typename accT>
__device__ bool check_duplicate(const Node<accT> *pq, const int size,
                                 Node<accT> new_node) {

  bool found = false;
  for (int i = threadIdx.x; i < size; i += blockDim.x) {
    if (pq[i].nodeid == new_node.nodeid) {
      found = true;
      break;
    }
  }

  unsigned mask = __ballot_sync(FULL_BITMASK, found);

  if (mask == 0)
    return false;

  else
    return true;
}



/*
template <typename T, typename SUMTYPE, int Dim>
__device__ SUMTYPE GetDistanceByVec(Point<T, SUMTYPE, Dim> *src_vec,
                                    Point<T, SUMTYPE, Dim> *dst_vec,
                                    Metric metric) {

  SUMTYPE dist;
  // L2
  if (metric == L2)
    dist = l2<T, SUMTYPE, Dim>(src_vec, dst_vec);
  else
    dist = cosine<T, SUMTYPE, Dim>(src_vec, dst_vec);

  __syncthreads();
  return dist;
}
*/




// Templated infinity value
template<typename T> __host__ __device__ T INFTY() {}
template<> __forceinline__ __host__ __device__ int INFTY<int>() {return INT_MAX;}
template<> __forceinline__ __host__ __device__ long long int INFTY<long long int>() {return LLONG_MAX;}
template<> __forceinline__ __host__ __device__ float INFTY<float>() {return FLT_MAX;}
template<> __forceinline__ __host__ __device__ uint8_t INFTY<uint8_t>() {return 255;}
template<> __forceinline__ __host__ __device__ uint32_t INFTY<uint32_t>() {return INT_MAX;}

template<typename T> __host__ __device__ T SMALLEST() {}
template<> __forceinline__ __host__ __device__ int SMALLEST<int>() {return INT_MIN;}
template<> __forceinline__ __host__ __device__ long long int SMALLEST<long long int>() {return LLONG_MIN;}
template<> __forceinline__ __host__ __device__ float SMALLEST<float>() {return FLT_MIN;}
template<> __forceinline__ __host__ __device__ uint8_t SMALLEST<uint8_t>() {return 0;}
template<> __forceinline__ __host__ __device__ uint32_t SMALLEST<uint32_t>() {return 0;}

// Structure that holds information and results for a query
template<typename IdxT, typename accT>
struct QueryCandidates
{
  IdxT* ids;
  accT* dists;
  int queryId;
  int size;
//  DistPair<T>* list;
  int maxSize;
//  int padding; // TODO - test if padding improves perf or not...

  __device__ void reset() {
    for(int i=threadIdx.x; i<maxSize; i+=blockDim.x) {
      ids[i] = INFTY<IdxT>();
      dists[i] = INFTY<accT>();
    }
    size = 0;
  }

  __inline__ __device__ bool check_visited(IdxT target, accT dist) {
    __syncthreads();
    __shared__ bool found;
    found = false;
    __syncthreads();
   
    if(size < maxSize) {
      __syncthreads();
      for(int i=threadIdx.x; i<size; i+= blockDim.x) {
        if(ids[i] == target) {
          found = true;
        }
      }
      __syncthreads();
      if(!found && threadIdx.x==0) {
        ids[size] = target;
        dists[size] = dist;
        size++;
      }
      __syncthreads();
    }
    return found;
  }

  __inline__ __device__ void print_visited() {
    printf("queryId:%d, size:%d\n", queryId, size);
    for(int i=0; i<size; i++) {
      printf("%d (%f), ", ids[i], dists[i]);
    }
    printf("\n");
  }
};

__global__ void print_graph(raft::device_matrix_view<uint32_t, int64_t> graph) {
  printf("nodes:%ld, degree:%ld\n", graph.extent(0), graph.extent(1));
  for(int i=0; i<10; i++) {
    for(int j=0; j<graph.extent(1); j++) {
      printf("%u, ", graph(i,j));
    }
    printf("\n");
  }
}

/************************************************/
/* Kernels that work on QueryCandidates objects */
/************************************************/

template<typename accT, typename IdxT = uint32_t>
__global__ void print_query_results(void* query_list_ptr, int count) {
  QueryCandidates<IdxT,accT>* query_list = static_cast<QueryCandidates<IdxT,accT>*>(query_list_ptr);

  for(int i=0; i<count; i++) {
    query_list[i].print_visited();
  }
}

template<typename IdxT, typename accT>
//__global__ void init_query_candidate_list(void* query_list_ptr, IdxT* visited_id_ptr, accT* visited_dist_ptr, int num_queries, int maxSize) {
__global__ void init_query_candidate_list(QueryCandidates<IdxT,accT>* query_list, IdxT* visited_id_ptr, accT* visited_dist_ptr, int num_queries, int maxSize) {

//  QueryCandidates<IdxT, accT>* query_list = static_cast<QueryCandidates<IdxT, accT>*>(query_list_ptr);
  IdxT* ids_ptr = static_cast<IdxT*>(visited_id_ptr);
  accT* dist_ptr = static_cast<accT*>(visited_dist_ptr);

  for(size_t i=blockIdx.x*blockDim.x + threadIdx.x; i<num_queries; i+=blockDim.x+gridDim.x) {
    query_list[i].maxSize = maxSize;
    query_list[i].size = 0;
    query_list[i].ids = &ids_ptr[i*(size_t)(maxSize)];
    query_list[i].dists = &dist_ptr[i*(size_t)(maxSize)];

    for(int j=0; j<maxSize; j++) {
      query_list[i].ids[j] = INFTY<IdxT>();
      query_list[i].dists[j] = INFTY<accT>();
    }
  }
}

template<typename IdxT, typename accT>
__global__ void set_query_ids(void* query_list_ptr, IdxT* d_query_ids, int step_size) {

  QueryCandidates<IdxT,accT>* query_list = static_cast<QueryCandidates<IdxT,accT>*>(query_list_ptr);

  for(int i=blockIdx.x*blockDim.x + threadIdx.x; i<step_size; i+=blockDim.x*gridDim.x) {
    query_list[i].queryId = d_query_ids[i];
    query_list[i].size = 0;
  }
}


// Currently only works with 1 thread, TODO parallelize
template<typename accT, typename IdxT = uint32_t>
__global__ void prefix_sums_sizes(QueryCandidates<IdxT,accT>* query_list, int num_queries, int* total_edges) {
  if(threadIdx.x==0 && blockIdx.x==0) {
    int sum=0;
    for(int i=0; i<num_queries+1; i++) {
      sum += query_list[i].size;
      query_list[i].size = sum - query_list[i].size; // exclusive prefix sum
    }
    *total_edges = query_list[num_queries].size;
  }
}



/*
  All threads store the input point coordinates into shared ememory
*/
template <typename T, typename accT, int Dim>
__device__ void update_shared_point(Point<T, accT, Dim> *shared_point,
                                   const T* data_ptr, int id) {
  shared_point->id = id;
  for (int i = threadIdx.x; i < Dim; i += blockDim.x) {
    shared_point->coords[i] = data_ptr[id*Dim + i];
  }
}

// Update the graph from the results of the query list (or reverse edge list)
template<typename accT, typename IdxT = uint32_t>
__global__ void write_graph_edges_kernel(raft::device_matrix_view<IdxT, int64_t> graph, void* query_list_ptr, int degree, int num_queries) {
  QueryCandidates<IdxT,accT>* query_list = static_cast<QueryCandidates<IdxT,accT>*>(query_list_ptr);

  for(int i=blockIdx.x; i<num_queries; i+=gridDim.x) {
    for(int j=threadIdx.x; j<query_list[i].size; j+=blockDim.x) {
      graph(query_list[i].queryId, j) = query_list[i].ids[j];
    }
if(threadIdx.x==0 && query_list[i].queryId ==0) {
  for(int j=0; j<32; j++) {
    printf("%u, ", query_list[i].ids[j]);
  }
  printf("\n");
}
  }
}

// Create src and dest edge lists used to sort and create reverse edges
template<typename accT, typename IdxT = uint32_t>
__global__ void create_reverse_edge_list(void* query_list_ptr, int num_queries, int degree, IdxT* edge_src, IdxT* edge_dest, int* reverse_edge_counts) {

  QueryCandidates<IdxT,accT>* query_list = static_cast<QueryCandidates<IdxT,accT>*>(query_list_ptr);

  for(int i=blockIdx.x*blockDim.x + threadIdx.x; i < num_queries; i += blockDim.x*gridDim.x) {
    int read_idx = i*query_list[i].maxSize;
    int cand_count = query_list[i+1].size - query_list[i].size;

    for(int j=0; j<cand_count; j++) {
      edge_src[query_list[i].size+j] = query_list[i].queryId;
//      edge_src[query_list[i].size+j].dist = query_list[i].list[j].dist;

      edge_dest[query_list[i].size+j] = query_list[i].ids[j];
      atomicAdd(&reverse_edge_counts[query_list[i].ids[j]], 1);
    }
  }
}

template<typename IdxT>
__global__ void test_print_list(IdxT* edge_src, IdxT* edge_dest, int num_edges) {
printf("printing src and dest, total edges:%d\n", num_edges);
  for(int i=0; i<num_edges; i++) {
    printf("%u (%u), ", edge_src[i], edge_dest[i]);
  }
printf("\n");
}

template<typename T, typename accT, int Dim, typename IdxT = uint32_t>
__global__ void populate_reverse_list_struct(
     QueryCandidates<IdxT,accT>* reverse_list, 
//     raft::mdspan<const T, raft::matrix_extent<int64_t>, raft::row_major, Accessor> dataset,
     IdxT* edge_src, IdxT* edge_dest, int* edge_histogram, int total_edges, int N) {

  int write_idx=0;
  int read_idx=0;
  for(int i = 0; i<N; i++) {
    if(edge_histogram[i] > 0) {
      reverse_list[write_idx].queryId = i;
      for(int j=0; j<edge_histogram[i] && j<reverse_list[write_idx].maxSize; j++) {
        reverse_list[write_idx].ids[j] = edge_src[read_idx+j];
      }
      if(edge_histogram[i] > reverse_list[write_idx].maxSize) {
        reverse_list[write_idx].size = reverse_list[write_idx].maxSize;
      }
      else {
        reverse_list[write_idx].size = edge_histogram[i];
        for(int j=reverse_list[write_idx].size; j<reverse_list[write_idx].maxSize; j++) {
          reverse_list[write_idx].ids[j] = INFTY<IdxT>();
          reverse_list[write_idx].dists[j] = INFTY<accT>();
        }
      }

      read_idx += edge_histogram[i];
      write_idx++;
    }
  }
}

template<typename T, typename accT, int Dim, typename IdxT = uint32_t,
         typename Accessor = raft::host_device_accessor<std::experimental::default_accessor<T>,
                                                         raft::memory_type::host>>
__global__ void recompute_reverse_dists(
     QueryCandidates<IdxT,accT>* reverse_list, 
     raft::mdspan<const T, raft::matrix_extent<int64_t>, raft::row_major, Accessor> dataset,
     int unique_dests) {

  const T* vec_ptr = dataset.data_handle();

  for(int i=blockIdx.x; i<unique_dests; i+=gridDim.x) {
    for(int j=0; j<reverse_list[i].size; j++) {
      reverse_list[i].dists[j] = l2<T,accT,Dim>(&vec_ptr[reverse_list[i].queryId*Dim], &vec_ptr[reverse_list[i].ids[j]*Dim]);
    }
  }

/*
  if(threadIdx.x==0 && blockIdx.x==0) {
    for(int i=0; i<unique_dests; i++) {
      reverse_list[i].print_visited();
    }
  }
*/
}

/**
 * @}
 */



} // cuvs::neighbors::vamana::detail namespace

#endif
