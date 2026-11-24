/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/common.hpp>
#include <cuvs/neighbors/gpipnn.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>

#include <cuda_runtime_api.h>

#include <nvtx3/nvtx3.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

template <typename T>
struct matrix {
  int64_t rows;
  int64_t cols;
  std::vector<T> data;
};

struct options {
  std::string dataset;
  uint32_t degree   = 64;
  uint32_t runs     = 1;
  uint32_t k        = 10;
  uint32_t itopk    = 128;
  double min_recall = 0.9;
};

auto usage(std::ostream& output, const char* program) -> void
{
  output << "usage: " << program
         << " --dataset <dataset.fbin|dataset_dir> [--degree N] [--runs N]\n"
            "       [--k N] [--itopk N] [--min-recall R]\n";
}

auto parse_options(int argc, char** argv) -> options
{
  options result;
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    if (argument == "--help") {
      usage(std::cout, argv[0]);
      std::exit(0);
    }
    auto value = [&]() -> std::string {
      if (++i >= argc) { throw std::runtime_error("missing value for " + argument); }
      return argv[i];
    };
    if (argument == "--dataset") {
      result.dataset = value();
    } else if (argument == "--degree") {
      result.degree = static_cast<uint32_t>(std::stoul(value()));
    } else if (argument == "--runs") {
      result.runs = static_cast<uint32_t>(std::stoul(value()));
    } else if (argument == "--k") {
      result.k = static_cast<uint32_t>(std::stoul(value()));
    } else if (argument == "--itopk") {
      result.itopk = static_cast<uint32_t>(std::stoul(value()));
    } else if (argument == "--min-recall") {
      result.min_recall = std::stod(value());
    } else {
      throw std::runtime_error("unknown option: " + argument);
    }
  }
  if (result.dataset.empty()) { throw std::runtime_error("--dataset is required"); }
  if (result.runs == 0) { throw std::runtime_error("--runs must be positive"); }
  if (result.k == 0) { throw std::runtime_error("--k must be positive"); }
  if (result.itopk < result.k) { throw std::runtime_error("--itopk must be at least --k"); }
  if (result.min_recall < 0.0 || result.min_recall > 1.0) {
    throw std::runtime_error("--min-recall must be between 0 and 1");
  }
  return result;
}

auto cuda_check(cudaError_t status, const char* operation) -> void
{
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

auto resolve_dataset(const std::string& argument) -> std::filesystem::path
{
  const std::filesystem::path path(argument);
  if (!std::filesystem::is_directory(path)) { return path; }

  for (const char* name : {"base.fbin", "gist_base.fbin", "sift_base.fbin", "sift10m_base.fbin"}) {
    const auto candidate = path / name;
    if (std::filesystem::is_regular_file(candidate)) { return candidate; }
  }
  throw std::runtime_error("dataset directory contains no supported base .fbin");
}

auto resolve_companion(const std::filesystem::path& directory,
                       std::initializer_list<const char*> names) -> std::filesystem::path
{
  for (const char* name : names) {
    const auto candidate = directory / name;
    if (std::filesystem::is_regular_file(candidate)) { return candidate; }
  }
  throw std::runtime_error("required validation file is missing under " + directory.string());
}

template <typename T>
auto read_matrix(const std::filesystem::path& path) -> matrix<T>
{
  std::ifstream input(path, std::ios::binary);
  if (!input) { throw std::runtime_error("cannot open " + path.string()); }

  const auto extension = path.extension().string();
  if (extension == ".fvecs" || extension == ".ivecs") {
    int32_t cols = 0;
    input.read(reinterpret_cast<char*>(&cols), sizeof(cols));
    if (!input || cols <= 0) { throw std::runtime_error("invalid vecs header in " + path.string()); }
    input.seekg(0, std::ios::end);
    const std::streamoff bytes = input.tellg();
    const auto row_bytes = static_cast<std::streamoff>(sizeof(int32_t) + cols * sizeof(T));
    if (bytes <= 0 || bytes % row_bytes != 0) {
      throw std::runtime_error("invalid vecs file size in " + path.string());
    }
    matrix<T> result{bytes / row_bytes, cols, {}};
    result.data.resize(static_cast<size_t>(result.rows) * result.cols);
    input.seekg(0);
    for (int64_t row = 0; row < result.rows; ++row) {
      int32_t row_cols = 0;
      input.read(reinterpret_cast<char*>(&row_cols), sizeof(row_cols));
      if (!input || row_cols != cols) {
        throw std::runtime_error("inconsistent vecs row in " + path.string());
      }
      input.read(reinterpret_cast<char*>(result.data.data() + row * result.cols),
                 static_cast<std::streamsize>(result.cols * sizeof(T)));
      if (!input) { throw std::runtime_error("truncated vecs body in " + path.string()); }
    }
    return result;
  }

  uint32_t shape[2]{};
  input.read(reinterpret_cast<char*>(shape), sizeof(shape));
  if (!input || shape[0] == 0 || shape[1] == 0) {
    throw std::runtime_error("invalid bin header in " + path.string());
  }

  matrix<T> result{static_cast<int64_t>(shape[0]), static_cast<int64_t>(shape[1]), {}};
  result.data.resize(static_cast<size_t>(result.rows) * result.cols);
  input.read(reinterpret_cast<char*>(result.data.data()),
             static_cast<std::streamsize>(result.data.size() * sizeof(T)));
  if (!input) { throw std::runtime_error("truncated bin body in " + path.string()); }
  return result;
}

auto compute_recall(const std::vector<uint32_t>& found,
                    const matrix<int32_t>& truth,
                    uint32_t k) -> double
{
  int64_t hits = 0;
  for (int64_t row = 0; row < truth.rows; ++row) {
    for (uint32_t i = 0; i < k; ++i) {
      const auto id = found[static_cast<size_t>(row) * k + i];
      for (uint32_t j = 0; j < k; ++j) {
        if (id == static_cast<uint32_t>(truth.data[static_cast<size_t>(row) * truth.cols + j])) {
          ++hits;
          break;
        }
      }
    }
  }
  return static_cast<double>(hits) / static_cast<double>(truth.rows * k);
}

}  // namespace

auto main(int argc, char** argv) -> int
{
  try {
    namespace gpipnn = cuvs::neighbors::experimental::gpipnn;

    const auto cli          = parse_options(argc, argv);
    const auto dataset_path = resolve_dataset(cli.dataset);
    const auto directory    = dataset_path.parent_path();
    const auto query_path   = resolve_companion(
      directory, {"query.fbin", "gist_query.fbin", "query.fvecs", "gist_query.fvecs"});
    const auto truth_path = resolve_companion(directory,
                                              {"groundtruth.neighbors.ibin",
                                               "gist_groundtruth.ibin",
                                               "groundtruth.ivecs",
                                               "gist_groundtruth.ivecs"});
    auto host             = read_matrix<float>(dataset_path);
    auto queries          = read_matrix<float>(query_path);
    const auto truth      = read_matrix<int32_t>(truth_path);
    const uint32_t degree   = cli.degree;
    if (host.cols % 2 != 0) { throw std::runtime_error("GPiPNN requires an even dimension"); }
    if (queries.cols != host.cols) { throw std::runtime_error("base/query dimension mismatch"); }
    if (truth.rows != queries.rows) {
      throw std::runtime_error("ground-truth/query row-count mismatch");
    }
    if (truth.cols < cli.k) { throw std::runtime_error("ground truth has fewer than --k columns"); }
    if (degree == 0 || degree > gpipnn::kMaxCandidates ||
        degree >= static_cast<uint64_t>(host.rows)) {
      throw std::runtime_error("graph_degree must be positive, below the row count, and <= 1024");
    }
    if (!gpipnn::is_supported()) { throw std::runtime_error("GPiPNN is unsupported on this GPU"); }

    std::cout << "dataset: " << dataset_path << '\n'
              << "shape:   " << host.rows << " x " << host.cols << '\n'
              << "degree:  " << degree << '\n'
              << "runs:    " << cli.runs << '\n'
              << "validation: recall@" << cli.k << " >= " << cli.min_recall
              << " at itopk " << cli.itopk << '\n';

    raft::resources resources;
    const auto stream = raft::resource::get_cuda_stream(resources).get();
    auto dataset =
      raft::make_device_matrix<float, int64_t>(resources, host.rows, host.cols);
    auto graph = raft::make_device_matrix<uint32_t, int64_t>(resources, host.rows, degree);

    cuda_check(cudaMemcpyAsync(dataset.data_handle(),
                               host.data.data(),
                               host.data.size() * sizeof(float),
                               cudaMemcpyHostToDevice,
                               stream),
               "upload dataset");
    raft::resource::sync_stream(resources);
    host.data.clear();
    host.data.shrink_to_fit();

    gpipnn::index_params params;
    params.graph_degree  = degree;
    params.max_candidates = std::max<uint32_t>(params.max_candidates, degree);

    double total_seconds = 0.0;
    for (uint32_t run = 1; run <= cli.runs; ++run) {
      const std::string range_name = "GPiPNN build iteration " + std::to_string(run);
      nvtx3::scoped_range iteration_range{range_name};
      const auto start = std::chrono::steady_clock::now();
      gpipnn::build_graph(resources, params, dataset.view(), graph.view());
      raft::resource::sync_stream(resources);
      const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
      total_seconds += seconds;
      std::cout << "run " << run << ":   " << seconds << " s\n";
    }

    double recall = 0.0;
    {
      nvtx3::scoped_range validation_range{"GPiPNN recall validation"};
      auto padded = cuvs::neighbors::make_device_padded_dataset_view(
        resources, raft::make_const_mdspan(dataset.view()));
      auto index = cuvs::neighbors::cagra::device_padded_index<float, uint32_t>(
        resources,
        cuvs::distance::DistanceType::L2Expanded,
        padded,
        raft::make_const_mdspan(graph.view()));

      auto device_queries =
        raft::make_device_matrix<float, int64_t>(resources, queries.rows, queries.cols);
      cuda_check(cudaMemcpyAsync(device_queries.data_handle(),
                                 queries.data.data(),
                                 queries.data.size() * sizeof(float),
                                 cudaMemcpyHostToDevice,
                                 stream),
                 "upload queries");
      auto neighbors =
        raft::make_device_matrix<uint32_t, int64_t>(resources, queries.rows, cli.k);
      auto distances = raft::make_device_matrix<float, int64_t>(resources, queries.rows, cli.k);

      cuvs::neighbors::cagra::search_params search_params;
      search_params.itopk_size = cli.itopk;
      cuvs::neighbors::cagra::search(resources,
                                     search_params,
                                     index,
                                     raft::make_const_mdspan(device_queries.view()),
                                     neighbors.view(),
                                     distances.view());

      std::vector<uint32_t> found(static_cast<size_t>(queries.rows) * cli.k);
      cuda_check(cudaMemcpyAsync(found.data(),
                                 neighbors.data_handle(),
                                 found.size() * sizeof(uint32_t),
                                 cudaMemcpyDeviceToHost,
                                 stream),
                 "download search results");
      raft::resource::sync_stream(resources);
      recall = compute_recall(found, truth, cli.k);
    }

    std::vector<uint32_t> first_row(degree);
    cuda_check(cudaMemcpy(first_row.data(),
                          graph.data_handle(),
                          first_row.size() * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost),
               "download first graph row");

    std::cout << "mean:    " << total_seconds / cli.runs << " s\n"
              << "recall@" << cli.k << ": " << recall << " (threshold " << cli.min_recall << ")\n"
              << "validation: " << (recall >= cli.min_recall ? "PASSED" : "FAILED")
              << "\nfirst row:";
    for (size_t i = 0; i < std::min<size_t>(first_row.size(), 10); ++i) {
      std::cout << ' ' << first_row[i];
    }
    if (first_row.size() > 10) { std::cout << " ..."; }
    std::cout << '\n';
    if (recall < cli.min_recall) { return 1; }
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
