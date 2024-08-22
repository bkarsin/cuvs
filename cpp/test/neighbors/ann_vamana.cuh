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

#include "../test_utils.cuh"
#include "ann_utils.cuh"
#include <raft/core/resource/cuda_stream.hpp>

#include "naive_knn.cuh"

#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/vamana.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/logger.hpp>
#include <raft/linalg/add.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/itertools.hpp>

#include <rmm/device_buffer.hpp>

#include <gtest/gtest.h>

#include <thrust/sequence.h>

#include <cstddef>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

namespace cuvs::neighbors::vamana {
namespace {


// Generate random graph (based on extents of matrix given)
RAFT_KERNEL GenerateRandomGraph_kernel(raft::device_matrix_view<uint32_t,int64_t> graph) {
  for(int i=blockIdx.x*blockDim.x + threadIdx.x; i<graph.extent(0); i+=blockDim.x*gridDim.x) {
    
  }
}

} // namespace

struct AnnVamanaInputs {
  int n_queries;
  int n_rows;
  int dim;
  int k;
  int max_queries;
  int degree;
  cuvs::distance::DistanceType metric;
  bool host_dataset;
  double min_recall;  // = std::nullopt;
};

/*
inline ::std::ostream& operator<<(::std::ostream& os, const AnnCagraInputs& p)
{
  std::vector<std::string> algo       = {"single-cta", "multi_cta", "multi_kernel", "auto"};
  std::vector<std::string> build_algo = {"IVF_PQ", "NN_DESCENT", "AUTO"};
  os << "{n_queries=" << p.n_queries << ", dataset shape=" << p.n_rows << "x" << p.dim
     << ", k=" << p.k << ", " << algo.at((int)p.algo) << ", max_queries=" << p.max_queries
     << ", itopk_size=" << p.itopk_size << ", search_width=" << p.search_width
     << ", metric=" << static_cast<int>(p.metric) << (p.host_dataset ? ", host" : ", device")
     << ", build_algo=" << build_algo.at((int)p.build_algo);
  if ((int)p.build_algo == 0 && p.ivf_pq_search_refine_ratio) {
    os << "(refine_rate=" << *p.ivf_pq_search_refine_ratio << ')';
  }
  if (p.compression.has_value()) {
    auto vpq = p.compression.value();
    os << ", pq_bits=" << vpq.pq_bits << ", pq_dim=" << vpq.pq_dim
       << ", vq_n_centers=" << vpq.vq_n_centers;
  }
  os << '}' << std::endl;
  return os;
}
*/

template <typename DistanceT, typename DataT, typename IdxT>
class AnnVamanaTest : public ::testing::TestWithParam<AnnVamanaInputs> {
 public:
  AnnVamanaTest()
    : stream_(raft::resource::get_cuda_stream(handle_)),
      ps(::testing::TestWithParam<AnnVamanaInputs>::GetParam()),
      database(0, stream_),
      search_queries(0, stream_)
  {
  }

 protected:
  void testVamana()
  {
    size_t queries_size = ps.n_queries * ps.k;
    std::vector<IdxT> indices_Vamana(queries_size);
    std::vector<IdxT> indices_naive(queries_size);
    std::vector<DistanceT> distances_Vamana(queries_size);
    std::vector<DistanceT> distances_naive(queries_size);

    {
      rmm::device_uvector<DistanceT> distances_naive_dev(queries_size, stream_);
      rmm::device_uvector<IdxT> indices_naive_dev(queries_size, stream_);

      cuvs::neighbors::naive_knn<DistanceT, DataT, IdxT>(handle_,
                                                         distances_naive_dev.data(),
                                                         indices_naive_dev.data(),
                                                         search_queries.data(),
                                                         database.data(),
                                                         ps.n_queries,
                                                         ps.n_rows,
                                                         ps.dim,
                                                         ps.k,
                                                         ps.metric);
      raft::update_host(distances_naive.data(), distances_naive_dev.data(), queries_size, stream_);
      raft::update_host(indices_naive.data(), indices_naive_dev.data(), queries_size, stream_);
      raft::resource::sync_stream(handle_);
    }

    {
      rmm::device_uvector<DistanceT> distances_dev(queries_size, stream_);
      rmm::device_uvector<IdxT> indices_dev(queries_size, stream_);

      {
        vamana::index_params index_params;
        index_params.metric = ps.metric;  
        index_params.graph_degree = ps.graph_degree;
                                          
// TODO - add search functionality
//        vamana::search_params search_params;
//        search_params.algo        = ps.algo;
//        search_params.max_queries = ps.max_queries;
//        search_params.team_size   = ps.team_size;

        auto database_view = raft::make_device_matrix_view<const DataT, int64_t>(
          (const DataT*)database.data(), ps.n_rows, ps.dim);

//        {
          vamana::index<DataT, IdxT> index(handle_);
          if (ps.host_dataset) {
            auto database_host = raft::make_host_matrix<DataT, int64_t>(ps.n_rows, ps.dim);
            raft::copy(database_host.data_handle(), database.data(), database.size(), stream_);
            auto database_host_view = raft::make_host_matrix_view<const DataT, int64_t>(
              (const DataT*)database_host.data_handle(), ps.n_rows, ps.dim);

            index = vamana::build(handle_, index_params, database_host_view);
          } else {
            index = vamana::build(handle_, index_params, database_view);
          };
/*
//          cagra::serialize(handle_, "cagra_index", index, ps.include_serialized_dataset);
//        }

//        cagra::index<DataT, IdxT> index(handle_);
//        cagra::deserialize(handle_, "cagra_index", &index);

//        if (!ps.include_serialized_dataset) { index.update_dataset(handle_, database_view); }

        auto search_queries_view = raft::make_device_matrix_view<const DataT, int64_t>(
          search_queries.data(), ps.n_queries, ps.dim);
        auto indices_out_view =
          raft::make_device_matrix_view<IdxT, int64_t>(indices_dev.data(), ps.n_queries, ps.k);
        auto dists_out_view = raft::make_device_matrix_view<DistanceT, int64_t>(
          distances_dev.data(), ps.n_queries, ps.k);

        cagra::search(
          handle_, search_params, index, search_queries_view, indices_out_view, dists_out_view);
        raft::update_host(distances_Cagra.data(), distances_dev.data(), queries_size, stream_);
        raft::update_host(indices_Cagra.data(), indices_dev.data(), queries_size, stream_);

        raft::resource::sync_stream(handle_);
      }

      // for (int i = 0; i < min(ps.n_queries, 10); i++) {
      //   //  std::cout << "query " << i << std::end;
      //   print_vector("T", indices_naive.data() + i * ps.k, ps.k, std::cout);
      //   print_vector("C", indices_Cagra.data() + i * ps.k, ps.k, std::cout);
      //   print_vector("T", distances_naive.data() + i * ps.k, ps.k, std::cout);
      //   print_vector("C", distances_Cagra.data() + i * ps.k, ps.k, std::cout);
      // }
      double min_recall = ps.min_recall;
      EXPECT_TRUE(eval_neighbours(indices_naive,
                                  indices_Cagra,
                                  distances_naive,
                                  distances_Cagra,
                                  ps.n_queries,
                                  ps.k,
                                  0.003,
                                  min_recall));
      if (!ps.compression.has_value()) {
        // Don't evaluate distances for CAGRA-Q for now as the error can be somewhat large
        EXPECT_TRUE(eval_distances(handle_,
                                   database.data(),
                                   search_queries.data(),
                                   indices_dev.data(),
                                   distances_dev.data(),
                                   ps.n_rows,
                                   ps.dim,
                                   ps.n_queries,
                                   ps.k,
                                   ps.metric,
                                   1.0e-4));
*/
      }
    }
  }

  void SetUp() override
  {
    database.resize(((size_t)ps.n_rows) * ps.dim, stream_);
    search_queries.resize(ps.n_queries * ps.dim, stream_);
    raft::random::RngState r(1234ULL);
    if constexpr (std::is_same<DataT, float>{}) {
      raft::random::normal(handle_, r, database.data(), ps.n_rows * ps.dim, DataT(0.1), DataT(2.0));
      raft::random::normal(
        handle_, r, search_queries.data(), ps.n_queries * ps.dim, DataT(0.1), DataT(2.0));
    } else {
      raft::random::uniformInt(
        handle_, r, database.data(), ps.n_rows * ps.dim, DataT(1), DataT(20));
      raft::random::uniformInt(
        handle_, r, search_queries.data(), ps.n_queries * ps.dim, DataT(1), DataT(20));
    }
    raft::resource::sync_stream(handle_);
  }

  void TearDown() override
  {
    raft::resource::sync_stream(handle_);
    database.resize(0, stream_);
    search_queries.resize(0, stream_);
  }

 private:
  raft::resources handle_;
  rmm::cuda_stream_view stream_;
  AnnVamanaInputs ps;
  rmm::device_uvector<DataT> database;
  rmm::device_uvector<DataT> search_queries;
};

inline std::vector<AnnVamanaInputs> generate_inputs()
{
  std::vector<AnnCagraInputs> inputs = raft::util::itertools::product<AnnVamanaInputs>(
    {100},
    {1000},
    {1, 8, 17},
    {1, 16},  // k
    {0, 1, 10, 100},  // query size
    {32},
    {cuvs::distance::DistanceType::L2Expanded},
    {false},
    {0.995});

  return inputs;
}

const std::vector<AnnVamanaInputs> inputs = generate_inputs();

}  // namespace cuvs::neighbors::vamana
