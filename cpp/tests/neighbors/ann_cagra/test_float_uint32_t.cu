/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include "../ann_cagra.cuh"

#include <utility>
#include <vector>

namespace cuvs::neighbors::cagra {

typedef AnnCagraTest<float, float, std::uint32_t> AnnCagraTestF_U32;
TEST_P(AnnCagraTestF_U32, AnnCagra_U32) { this->testCagra<uint32_t>(); }
TEST_P(AnnCagraTestF_U32, AnnCagra_I64) { this->testCagra<int64_t>(); }

typedef AnnCagraAddNodesTest<float, float, std::uint32_t> AnnCagraAddNodesTestF_U32;
TEST_P(AnnCagraAddNodesTestF_U32, AnnCagraAddNodes) { this->testCagra(); }

typedef AnnCagraFilterTest<float, float, std::uint32_t> AnnCagraFilterTestF_U32;
TEST_P(AnnCagraFilterTestF_U32, AnnCagra) { this->testCagra(); }

typedef AnnCagraIndexMergeTest<float, float, std::uint32_t> AnnCagraIndexMergeTestF_U32;
TEST_P(AnnCagraIndexMergeTestF_U32, AnnCagraIndexMerge_U32) { this->testCagra<uint32_t>(); }
TEST_P(AnnCagraIndexMergeTestF_U32, AnnCagraIndexMerge_I64) { this->testCagra<int64_t>(); }

typedef AnnCagraIndexFilteredMergeTest<float, float, std::uint32_t>
  AnnCagraIndexFilteredMergeTestF_U32;
TEST_P(AnnCagraIndexFilteredMergeTestF_U32, AnnCagraIndexFilteredMerge_U32)
{
  this->testCagra<uint32_t>();
}

INSTANTIATE_TEST_CASE_P(AnnCagraTest, AnnCagraTestF_U32, ::testing::ValuesIn(inputs));
INSTANTIATE_TEST_CASE_P(AnnCagraAddNodesTest,
                        AnnCagraAddNodesTestF_U32,
                        ::testing::ValuesIn(inputs_addnode));
INSTANTIATE_TEST_CASE_P(AnnCagraFilterTest,
                        AnnCagraFilterTestF_U32,
                        ::testing::ValuesIn(inputs_filtering));
INSTANTIATE_TEST_CASE_P(AnnCagraIndexMergeTest,
                        AnnCagraIndexMergeTestF_U32,
                        ::testing::ValuesIn(inputs));

INSTANTIATE_TEST_CASE_P(AnnCagraIndexFilteredMergeTest,
                        AnnCagraIndexFilteredMergeTestF_U32,
                        ::testing::ValuesIn(inputs));

typedef AnnCagraMultiPartitionTest<float, float, std::uint32_t> AnnCagraMultiPartitionTestF_U32;
TEST_P(AnnCagraMultiPartitionTestF_U32, Search) { this->testSearch(); }
TEST_P(AnnCagraMultiPartitionTestF_U32, FilteredSearch) { this->testFilteredSearch(); }

INSTANTIATE_TEST_CASE_P(AnnCagraMultiPartitionTest,
                        AnnCagraMultiPartitionTestF_U32,
                        ::testing::ValuesIn(inputs_mp));

// Builds one CAGRA index per {metric, graph_degree} spec over a shared random dataset, then asserts
// a multi-partition search over them throws. Shared by the rejection tests below, which each
// violate one "all partitions must be uniform / supported" precondition. The rejections are
// invariant to dtype / layout, so they are checked once here instead of swept across the fixture.
namespace {
void expect_multi_partition_search_throws(
  const std::vector<std::pair<cuvs::distance::DistanceType, int>>& partition_specs,
  const cagra::search_params& search_params)
{
  raft::resources handle;
  auto stream = raft::resource::get_cuda_stream(handle);

  constexpr int n_rows = 256, dim = 8, n_queries = 10, k = 4;
  const int num_partitions = static_cast<int>(partition_specs.size());
  const int part_size      = n_rows / num_partitions;

  rmm::device_uvector<float> database(static_cast<size_t>(n_rows) * dim, stream);
  rmm::device_uvector<float> queries(static_cast<size_t>(n_queries) * dim, stream);
  raft::random::RngState r(1234ULL);
  InitDataset(handle, database.data(), n_rows, dim, cuvs::distance::DistanceType::L2Expanded, r);
  InitDataset(handle, queries.data(), n_queries, dim, cuvs::distance::DistanceType::L2Expanded, r);
  raft::resource::sync_stream(handle);

  std::vector<cagra::index<float, std::uint32_t>> part_indices;
  // An index only holds a view, so any padded copy has to outlive it.
  std::vector<cuvs::neighbors::test::padded_device_matrix_for_cagra<float>> part_padded;
  part_padded.reserve(num_partitions);
  for (int i = 0; i < num_partitions; i++) {
    const auto [metric, graph_degree] = partition_specs[i];
    cagra::index_params index_params;
    index_params.metric                    = metric;
    index_params.graph_degree              = graph_degree;
    index_params.intermediate_graph_degree = graph_degree * 2;
    index_params.graph_build_params =
      graph_build_params::nn_descent_params(index_params.intermediate_graph_degree, metric);
    auto view = raft::make_device_matrix_view<const float, int64_t>(
      database.data() + static_cast<size_t>(i) * part_size * dim, part_size, dim);
    part_padded.emplace_back(handle, view);
    auto const& padded = part_padded.back().view;
    part_indices.push_back(cagra::build(handle, index_params, padded));
    auto& part_index = part_indices.back();
    part_index       = cagra::update_dataset(handle, std::move(part_index), padded);
  }
  std::vector<const cagra::index<float, std::uint32_t>*> index_ptrs;
  for (auto& idx : part_indices) {
    index_ptrs.push_back(&idx);
  }

  const size_t out_size = static_cast<size_t>(n_queries) * k;
  rmm::device_uvector<uint32_t> partition_ids(out_size, stream);
  rmm::device_uvector<uint32_t> neighbors(out_size, stream);
  rmm::device_uvector<float> distances(out_size, stream);

  auto queries_view =
    raft::make_device_matrix_view<const float, int64_t>(queries.data(), n_queries, dim);
  auto part_ids_view =
    raft::make_device_matrix_view<uint32_t, int64_t>(partition_ids.data(), n_queries, k);
  auto neighbors_view =
    raft::make_device_matrix_view<uint32_t, int64_t>(neighbors.data(), n_queries, k);
  auto dists_view = raft::make_device_matrix_view<float, int64_t>(distances.data(), n_queries, k);

  EXPECT_THROW(
    cagra::search(
      handle, search_params, index_ptrs, queries_view, part_ids_view, neighbors_view, dists_view),
    std::exception);
}
}  // namespace

// MULTI_KERNEL is intentionally unsupported in the multi-partition path; the call must fail rather
// than silently fall back.
TEST(AnnCagraMultiPartition, MultiKernelRejected)
{
  cagra::search_params search_params;
  search_params.algo = search_algo::MULTI_KERNEL;
  expect_multi_partition_search_throws({{cuvs::distance::DistanceType::L2Expanded, 16},
                                        {cuvs::distance::DistanceType::L2Expanded, 16}},
                                       search_params);
}

// The shared plan descriptor and the cross-partition select_k direction are derived from
// indices[0], so all partitions must share one metric; a mismatch must be rejected.
TEST(AnnCagraMultiPartition, MixedMetricRejected)
{
  expect_multi_partition_search_throws({{cuvs::distance::DistanceType::L2Expanded, 16},
                                        {cuvs::distance::DistanceType::InnerProduct, 16}},
                                       cagra::search_params{});
}

// The shared plan descriptor is sized from indices[0]'s graph degree, so all partitions must share
// one graph degree; a mismatch must be rejected.
TEST(AnnCagraMultiPartition, MixedGraphDegreeRejected)
{
  expect_multi_partition_search_throws({{cuvs::distance::DistanceType::L2Expanded, 16},
                                        {cuvs::distance::DistanceType::L2Expanded, 32}},
                                       cagra::search_params{});
}

// GPiPNN reaches CAGRA through two pieces of glue the parameterized sweep above cannot isolate: a
// conforming step that materializes host, padded-stride or odd-`dim` dataset views as the
// contiguous even-`dim` device matrix GPiPNN takes, and a reconciliation step that silently
// rewrites the caller's GPiPNN parameters to agree with CAGRA's. Both change results rather than
// just failing, so each is pinned here.
namespace {

constexpr int64_t kGpipnnRows    = 512;
constexpr uint32_t kGpipnnDegree = 32;

auto make_gpipnn_params(cuvs::distance::DistanceType metric =
                          cuvs::distance::DistanceType::L2Expanded) -> cagra::index_params
{
  cagra::index_params params;
  params.metric                    = metric;
  params.graph_degree              = kGpipnnDegree;
  params.intermediate_graph_degree = kGpipnnDegree * 2;
  graph_build_params::gpipnn_params gp;
  gp.graph_degree           = kGpipnnDegree;  // agrees with CAGRA's, so nothing is reconciled
  gp.metric                 = metric;
  params.graph_build_params = gp;
  return params;
}

/** Builds with the GPiPNN builder and returns the resulting graph on the host. */
template <typename DatasetViewT>
auto gpipnn_graph(raft::resources const& res,
                  cagra::index_params const& params,
                  DatasetViewT const& dataset) -> std::vector<uint32_t>
{
  auto index = cagra::build(res, params, dataset);
  std::vector<uint32_t> out(index.graph().size());
  raft::copy(
    out.data(), index.graph().data_handle(), out.size(), raft::resource::get_cuda_stream(res));
  raft::resource::sync_stream(res);
  return out;
}

/** Random rows of `dim` columns, plus the same rows widened by one zero column. */
struct gpipnn_fixture {
  raft::resources res;
  int64_t dim;
  rmm::device_uvector<float> narrow;
  rmm::device_uvector<float> wide;

  explicit gpipnn_fixture(int64_t d)
    : dim(d),
      narrow(static_cast<size_t>(kGpipnnRows * d), raft::resource::get_cuda_stream(res)),
      wide(static_cast<size_t>(kGpipnnRows * (d + 1)), raft::resource::get_cuda_stream(res))
  {
    auto stream = raft::resource::get_cuda_stream(res).get();
    raft::random::RngState r(1234ULL);
    InitDataset(res,
                narrow.data(),
                static_cast<uint32_t>(kGpipnnRows),
                static_cast<uint32_t>(dim),
                cuvs::distance::DistanceType::L2Expanded,
                r);
    RAFT_CUDA_TRY(cudaMemsetAsync(wide.data(), 0, sizeof(float) * wide.size(), stream));
    RAFT_CUDA_TRY(cudaMemcpy2DAsync(wide.data(),
                                    sizeof(float) * static_cast<size_t>(dim + 1),
                                    narrow.data(),
                                    sizeof(float) * static_cast<size_t>(dim),
                                    sizeof(float) * static_cast<size_t>(dim),
                                    static_cast<size_t>(kGpipnnRows),
                                    cudaMemcpyDefault,
                                    stream));
    raft::resource::sync_stream(res);
  }

  auto narrow_matrix() const -> raft::device_matrix_view<const float, int64_t>
  {
    return raft::make_device_matrix_view<const float, int64_t>(narrow.data(), kGpipnnRows, dim);
  }
  auto wide_matrix() const -> raft::device_matrix_view<const float, int64_t>
  {
    return raft::make_device_matrix_view<const float, int64_t>(wide.data(), kGpipnnRows, dim + 1);
  }
};

}  // namespace

// An odd `dim` is widened by one zero column, which contributes nothing to any dot product or row
// norm. So the graph must be the one the caller asked for, not an approximation of it: building
// from `dim` and from the same rows pre-widened to `dim + 1` must agree exactly.
TEST(AnnCagraGpipnn, OddDimPaddingLeavesGraphUnchanged)
{
  gpipnn_fixture f(/*dim=*/7);
  const auto from_odd =
    gpipnn_graph(f.res, make_gpipnn_params(), make_device_standard_dataset_view(f.narrow_matrix()));
  const auto from_even =
    gpipnn_graph(f.res, make_gpipnn_params(), make_device_standard_dataset_view(f.wide_matrix()));
  ASSERT_EQ(from_odd, from_even);
}

// A padded dataset carries a row stride wider than the logical `dim`, which the conforming step has
// to strip out. Same logical rows, so the graph must not move.
TEST(AnnCagraGpipnn, PaddedStrideMatchesContiguous)
{
  // dim 5 pads to a stride of 8 floats, so this exercises stride != dim and odd dim at once.
  gpipnn_fixture f(/*dim=*/5);
  const auto contiguous =
    gpipnn_graph(f.res, make_gpipnn_params(), make_device_standard_dataset_view(f.narrow_matrix()));

  auto owned_padded = make_device_padded_dataset(f.res, f.narrow_matrix());
  auto padded_view  = owned_padded->view();
  ASSERT_NE(static_cast<int64_t>(padded_view.data_const_view().stride(0)), f.dim)
    << "padded copy was not actually strided";
  const auto padded = gpipnn_graph(f.res, make_gpipnn_params(), padded_view);
  ASSERT_EQ(contiguous, padded);
}

// A host-resident dataset is staged to device by the same conforming step. Same bytes in, so the
// graph must match the device-resident build.
TEST(AnnCagraGpipnn, HostDatasetMatchesDevice)
{
  gpipnn_fixture f(/*dim=*/8);
  const auto from_device =
    gpipnn_graph(f.res, make_gpipnn_params(), make_device_standard_dataset_view(f.narrow_matrix()));

  auto host = raft::make_host_matrix<float, int64_t>(kGpipnnRows, f.dim);
  raft::copy(
    host.data_handle(), f.narrow.data(), host.size(), raft::resource::get_cuda_stream(f.res));
  raft::resource::sync_stream(f.res);
  const auto from_host =
    gpipnn_graph(f.res,
                 make_gpipnn_params(),
                 make_host_standard_dataset_view(raft::make_const_mdspan(host.view())));
  ASSERT_EQ(from_device, from_host);
}

// GPiPNN's own `graph_degree`, `max_candidates` and `metric` are overridden to agree with CAGRA's
// rather than being rejected. The override has to actually happen: `max_candidates` below
// `graph_degree` would otherwise fail GPiPNN's own validation, and a disagreeing degree or metric
// would quietly produce the wrong graph.
TEST(AnnCagraGpipnn, ParametersReconciledWithCagra)
{
  gpipnn_fixture f(/*dim=*/8);
  auto dataset = make_device_standard_dataset_view(f.narrow_matrix());

  auto params       = make_gpipnn_params();
  auto& gp          = std::get<graph_build_params::gpipnn_params>(params.graph_build_params);
  gp.graph_degree   = kGpipnnDegree / 2;
  gp.max_candidates = 4;
  gp.metric         = cuvs::distance::DistanceType::CosineExpanded;

  const auto reconciled = gpipnn_graph(f.res, params, dataset);
  ASSERT_EQ(reconciled.size(), static_cast<size_t>(kGpipnnRows) * kGpipnnDegree);

  // CAGRA's values win, so the graph must equal the one built from parameters that already agreed.
  const auto agreed = gpipnn_graph(f.res, make_gpipnn_params(), dataset);
  ASSERT_EQ(reconciled, agreed);
}

// Only L2Expanded and CosineExpanded are supported; anything else must be refused rather than
// quietly built as L2. On a GPU too old for GPiPNN the builder is swapped for NN-descent before
// this check runs, which would make the metric legal again.
TEST(AnnCagraGpipnn, UnsupportedMetricRejected)
{
  if (!cuvs::neighbors::experimental::gpipnn::is_supported()) { GTEST_SKIP(); }
  gpipnn_fixture f(/*dim=*/8);
  auto dataset = make_device_standard_dataset_view(f.narrow_matrix());
  EXPECT_THROW(
    cagra::build(f.res, make_gpipnn_params(cuvs::distance::DistanceType::InnerProduct), dataset),
    std::exception);
}

}  // namespace cuvs::neighbors::cagra
