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

#include <cstdint>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/matrix/copy.cuh>
#include <raft/random/make_blobs.cuh>
#include <raft/random/sample_without_replacement.cuh>
#include <raft/util/cudart_utils.hpp>

#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/counting_iterator.h>

// Fill dataset and queries with synthetic data.
void generate_dataset(raft::device_resources const &dev_resources,
                      raft::device_matrix_view<float, int64_t> dataset,
                      raft::device_matrix_view<float, int64_t> queries) {
  auto labels = raft::make_device_vector<int64_t, int64_t>(dev_resources,
                                                           dataset.extent(0));
  raft::random::make_blobs(dev_resources, dataset, labels.view());
  raft::random::RngState r(time(NULL));
  raft::random::uniform(
      dev_resources, r,
      raft::make_device_vector_view(queries.data_handle(), queries.size()),
      -1.0f, 1.0f);
}

// Copy the results to host and print a few samples
template <typename IdxT>
void print_results(raft::device_resources const &dev_resources,
                   raft::device_matrix_view<IdxT, int64_t> neighbors,
                   raft::device_matrix_view<float, int64_t> distances) {
  int64_t topk = neighbors.extent(1);
  auto neighbors_host =
      raft::make_host_matrix<IdxT, int64_t>(neighbors.extent(0), topk);
  auto distances_host =
      raft::make_host_matrix<float, int64_t>(distances.extent(0), topk);

  cudaStream_t stream = raft::resource::get_cuda_stream(dev_resources);

  raft::copy(neighbors_host.data_handle(), neighbors.data_handle(),
             neighbors.size(), stream);
  raft::copy(distances_host.data_handle(), distances.data_handle(),
             distances.size(), stream);

  // The calls to RAFT algorithms and  raft::copy is asynchronous.
  // We need to sync the stream before accessing the data.
  raft::resource::sync_stream(dev_resources, stream);

  for (int query_id = 0; query_id < 2; query_id++) {
    std::cout << "Query " << query_id << " neighbor indices: ";
    raft::print_host_vector("", &neighbors_host(query_id, 0), topk, std::cout);
    std::cout << "Query " << query_id << " neighbor distances: ";
    raft::print_host_vector("", &distances_host(query_id, 0), topk, std::cout);
  }
}

/** Subsample the dataset to create a training set*/
raft::device_matrix<float, int64_t>
subsample(raft::device_resources const &dev_resources,
          raft::device_matrix_view<const float, int64_t> dataset,
          raft::device_vector_view<const int64_t, int64_t> data_indices,
          float fraction) {
  int64_t n_samples = dataset.extent(0);
  int64_t n_dim = dataset.extent(1);
  int64_t n_train = n_samples * fraction;
  auto trainset =
      raft::make_device_matrix<float, int64_t>(dev_resources, n_train, n_dim);

  int seed = 137;
  raft::random::RngState rng(seed);
  auto train_indices =
      raft::make_device_vector<int64_t>(dev_resources, n_train);

  raft::random::sample_without_replacement(dev_resources, rng, data_indices,
                                           std::nullopt, train_indices.view(),
                                           std::nullopt);

  raft::matrix::copy_rows(dev_resources, dataset, trainset.view(),
                          raft::make_const_mdspan(train_indices.view()));

  return trainset;
}


template<typename T, typename idxT>
void read_labeled_data(std::string data_fname, 
                       std::string data_label_fname, 
                       std::string query_fname, 
                       std::string query_label_fname, 
                       std::vector<T>* data, 
                       std::vector<std::vector<int>>* data_labels, 
                       std::vector<T>* queries, 
                       std::vector<std::vector<int>>* query_labels, 
                       std::vector<int>* cat_freq, 
                       std::vector<int>* query_freq, 
                       int max_N) {

  // Read datafile in
  std::ifstream datafile(data_fname, std::ifstream::binary);
  uint32_t N;
  uint32_t dim;
  datafile.read((char*)&N, sizeof(uint32_t));
  datafile.read((char*)&dim, sizeof(uint32_t));

  if(N > max_N) N = max_N;
  printf("N:%u, dim:%u\n", N, dim);

  data->resize(N*dim);
  datafile.read(reinterpret_cast<char*>(data->data()), N*dim);
  datafile.close();


  // read query data in
  std::ifstream queryfile(query_fname, std::ifstream::binary);
  
  uint32_t q_N;
  uint32_t q_dim;
  queryfile.read((char*)&q_N, sizeof(uint32_t));
  queryfile.read((char*)&q_dim, sizeof(uint32_t));
printf("qN:%u, qdim:%u\n", q_N, q_dim);
  if(q_dim != dim) {
    printf("query dim and data dim don't match!\n");
    exit(1);
  }
  queries->resize(q_N*dim);
  queryfile.read(reinterpret_cast<char*>(queries->data()), q_N*dim);
  queryfile.close();

  // read labels for data vectors
  data_labels->reserve(N);
  std::ifstream labelfile(data_label_fname);
  std::string line, token;
  std::string label_str;
  int cnt=0;
//  while(std::getline(labelfile, line)) {
  while(cnt < N && std::getline(labelfile, line)) {
    std::vector<int> label_list;
    std::istringstream ss(line);
    for(int i; ss >> i;) {
      if(i > cat_freq->size()) { cat_freq->resize(i, 0); printf("new size:%d\n", cat_freq->size());}
      (*cat_freq)[i]++;
      label_list.push_back(i);
      if(ss.peek() == ',')
         ss.ignore();
    }
    data_labels->push_back(label_list); 

    cnt++;
  }
  labelfile.close();

  // read labels for queries
  query_freq->resize(cat_freq->size(), 0);
  query_labels->reserve(q_N);
  std::ifstream qlabelfile(query_label_fname);
  cnt=0;
//  while(std::getline(labelfile, line)) {
  while(cnt < q_N && std::getline(qlabelfile, line)) {
    std::vector<int> label_list;
    std::istringstream ss(line);
    for(int i; ss >> i;) {
//      if(i > query_freq->size()) { query_freq->resize(i, 0); }
      (*query_freq)[i]++;
      label_list.push_back(i);
      if(ss.peek() == ',')
         ss.ignore();
    }
    query_labels->push_back(label_list); 

    cnt++;
  }
  qlabelfile.close();

  printf("queries read. size:%d\n", queries->size());

}
