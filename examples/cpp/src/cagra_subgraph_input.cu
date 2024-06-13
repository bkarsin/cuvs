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

#include <cstdint>
#include <raft/core/handle.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/random/make_blobs.cuh>

#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/brute_force.hpp>

#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include "common.cuh"


void print_vecs(raft::device_resources const &dev_resources, 
                  raft::device_matrix_view<float,int64_t> dataset,
                  raft::device_matrix_view<float,int64_t> queries) {
  auto h_data = raft::make_host_matrix<float,int64_t>(dataset.extent(0), dataset.extent(1));
  auto h_queries = raft::make_host_matrix<float,int64_t>(queries.extent(0), queries.extent(1));

  cudaStream_t stream = raft::resource::get_cuda_stream(dev_resources);
  raft::copy(h_data.data_handle(), dataset.data_handle(), dataset.size(), stream);
  raft::copy(h_queries.data_handle(), queries.data_handle(), queries.size(), stream);

  printf("data:\n");
  for(int i=0; i<10; i++) {
    printf("i:%d - ", i);
    for(int j=0; j<h_data.extent(1); j++) {
      printf("%f, ", h_data(i,j));
    }
    printf("\n");
  }
  printf("queries:\n");
  for(int i=0; i<10; i++) {
    printf("i:%d - ", i);
    for(int j=0; j<h_queries.extent(1); j++) {
      printf("%f, ", h_queries(i,j));
    }
    printf("\n");
  }
}

__global__ void copy_int8_to_float(raft::device_matrix_view<const uint8_t, int64_t> old_data, raft::device_matrix_view<float, int64_t> new_data) {
  for(int i=blockIdx.x*blockDim.x + threadIdx.x; i<old_data.extent(0); i+=blockDim.x*gridDim.x) {
    for(int j=0; j<old_data.extent(1); j++) {
      new_data(i,j) = (float)old_data(i,j);
    }
  }
}

void compute_recall_unfiltered(raft::device_resources const& dev_resources, 
                    raft::device_matrix_view<const uint8_t, int64_t> dataset, 
                    raft::device_matrix_view<const uint8_t, int64_t> queries,
                    raft::device_matrix_view<uint32_t> neighbors,
                    int topk) {

  int n_queries = queries.extent(0);
  cudaStream_t stream = raft::resource::get_cuda_stream(dev_resources);

  auto temp_dataset = raft::make_device_matrix<float, int64_t>(dev_resources, dataset.extent(0), dataset.extent(1));
  copy_int8_to_float<<<32,32>>>(dataset, temp_dataset.view());
  
  auto temp_queries = raft::make_device_matrix<float, int64_t>(dev_resources, queries.extent(0), queries.extent(1));
  copy_int8_to_float<<<32,32>>>(queries, temp_queries.view());
  cudaDeviceSynchronize();

  auto bf_index = cuvs::neighbors::brute_force::build(dev_resources, temp_dataset.view());
  
  auto bf_neighbors = raft::make_device_matrix<int64_t,int64_t>(dev_resources, n_queries, topk);
  auto bf_distances = raft::make_device_matrix<float,int64_t>(dev_resources, n_queries, topk);

  cuvs::neighbors::brute_force::search(dev_resources, bf_index, temp_queries.view(), bf_neighbors.view(), bf_distances.view(), std::nullopt);

  auto h_bf_neighbors = raft::make_host_matrix<int64_t,int64_t>(n_queries, topk);
  raft::copy(h_bf_neighbors.data_handle(), bf_neighbors.data_handle(), bf_neighbors.size(), stream);

  auto h_neighbors = raft::make_host_matrix<uint32_t>(n_queries, topk);
  raft::copy(h_neighbors.data_handle(), neighbors.data_handle(), neighbors.size(), stream);

//printf("computing unfiltered recall...\n");
  int correct=0;
  for(int i=0; i<neighbors.extent(0); i++) {
    for(int j=0; j<topk; j++) {
      for(int k=0; k<topk; k++) {
        if(h_neighbors(i,j) == h_bf_neighbors(i,k)) {
          correct++;
          break;
        }
      }
    }
  }

/*
  for(int i=0; i<queries.extent(0); i++) {
    for(int j=0; j<topk; j++) {
      printf("%d, ", h_neighbors(i,j));
    }
    printf("\n");
    for(int j=0; j<topk; j++) {
      printf("%d, ", h_bf_neighbors(i,j));
    }
    printf("\n");
  }
*/

  printf("Recall - vectors:%d, queries:%d, topk:%d, correct:%d, recall:%f\n", dataset.extent(0), n_queries, topk, correct, (float)correct / (float)(n_queries*topk));

}
   


void cagra_build_search_subgraphs(raft::device_resources const& dev_resources,
                               std::vector<raft::device_matrix_view<uint8_t, int64_t>> datasets,
                               std::vector<raft::device_matrix_view<uint8_t, int64_t>> queries,
                               int n_categories,
                               int topk, int itopk)
{
  printf("Starting fcn...\n");
  using namespace cuvs::neighbors;

  cudaStream_t stream = raft::resource::get_cuda_stream(dev_resources);

  // Create neighbors and dist output lists for each cat
  std::vector<raft::device_matrix<uint32_t, int64_t>> neighbors;
  neighbors.reserve(n_categories);
  std::vector<raft::device_matrix<float,int64_t>> distances;
  distances.reserve(n_categories);

  // use default index parameters
  cagra::index_params index_params;

  std::vector<cuvs::neighbors::cagra::index<uint8_t,uint32_t>> indexes;
  indexes.reserve(n_categories);

  for(int i=0; i<n_categories; i++) {
    neighbors.push_back(raft::make_device_matrix<uint32_t,int64_t>(dev_resources, queries[i].extent(0), topk));
    distances.push_back(raft::make_device_matrix<float,int64_t>(dev_resources, queries[i].extent(0), topk));


    if(queries[i].extent(0) > 0 && datasets[i].extent(0) > 0) {
      printf("calling build for dataset %d...\n", i);
      indexes.push_back(cagra::build(dev_resources, index_params, raft::make_const_mdspan(datasets[i])));
    }
    else {
      printf("skipping build for label %d\n", i);
      indexes.push_back(cagra::index<uint8_t,uint32_t>(dev_resources));
    }
  }

  raft::resource::sync_stream(dev_resources);

  printf("Indexes all created!\n");
  size_t free_t, total_t;
  cudaMemGetInfo(&free_t, &total_t);
  printf("total mem:%lld, free mem:%lld\n", total_t/1000000, free_t/1000000);

  // use default search parameters
  cagra::search_params search_params;
  search_params.algo = cuvs::neighbors::cagra::search_algo::SINGLE_CTA;
  search_params.itopk_size = itopk;
  search_params.search_width = 2;

  // search K nearest neighbors
  for(int i=0; i<n_categories; i++) {
    if(queries[i].extent(0) > 0 && datasets[i].extent(0) > 0) {
      cagra::search(dev_resources, search_params, indexes[i], raft::make_const_mdspan(queries[i]), neighbors[i].view(), distances[i].view());

//    print_results(dev_resources, neighbors[i].view(), distances[i].view());
      printf("Label: %d - ", i);
      compute_recall_unfiltered(dev_resources, datasets[i], queries[i], neighbors[i].view(), topk);
    }
  }

}


template<typename T, typename IdxT>
void get_labeled_lists(raft::device_resources const& dev_resources, std::vector<uint8_t> data, std::vector<std::vector<int>> data_labels, std::vector<raft::device_matrix_view<T, IdxT>> datasets, std::vector<int> cat_freq)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(dev_resources);
  int N = data_labels.size();
  int dim = data.size() / N;
  printf("N:%d, data.size:%d, dim:%d\n", N, data.size(), dim);

  // Find largest cat dataset
  int max_cat_size=0;
  for(int cat=0; cat < cat_freq.size(); cat++) {
    if(cat_freq[cat] > max_cat_size) max_cat_size = cat_freq[cat];
  }

  auto vec_list = raft::make_host_matrix<uint8_t, int64_t>(max_cat_size, dim);

  for(int cat=0; cat < cat_freq.size(); cat++) {
//    auto vec_list = raft::make_host_matrix<int8_t, int64_t>(cat_freq[cat], dim);
    int idx=0;
    for(int i=0; i<data_labels.size(); i++) {
//printf("i:%d, label size:%d\n", i, data_labels[i].size());
      for(int j=0; j<data_labels[i].size(); j++) {
        if(data_labels[i][j] == cat) {
//          printf("j:%d matches label!, adding to vec_list at idx:%d\n", j, idx);
          // add vector to dataset
          for(int k=0; k<dim; k++) {
            vec_list(idx, k) = data[i*dim+k];
          }
          idx++;
          break;
        }
      }
    }
    // copy vec_list to corresponding dataset
    raft::copy(datasets[cat].data_handle(), vec_list.data_handle(), idx*dim, stream);
  }

}

int main(int argc, char* argv[])
{
  raft::device_resources dev_resources;

  // Set pool memory resource with 1 GiB initial pool size. All allocations use the same pool.
  rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource> pool_mr(
    rmm::mr::get_current_device_resource(), 10*1024 * 1024 * 1024ull);
  rmm::mr::set_current_device_resource(&pool_mr);

  std::string data_fname = (std::string)argv[1];
  std::string data_label_fname = (std::string)argv[2];
  std::string query_fname = (std::string)argv[3];
  std::string query_label_fname = (std::string)argv[4];
  size_t max_N = atoi(argv[5]);
  int topk = atoi(argv[6]);
  int itopk = atoi(argv[7]);

  std::vector<uint8_t> h_data;
  std::vector<uint8_t> h_queries;
  std::vector<std::vector<int>> data_labels;
  std::vector<std::vector<int>> query_labels;
  
  std::vector<int> cat_freq(0);
  std::vector<int> query_freq(0);

  read_labeled_data<uint8_t, int64_t>(data_fname, data_label_fname, query_fname, query_label_fname,
                    &h_data, &data_labels, &h_queries, &query_labels, &cat_freq, &query_freq, max_N);

  int N = data_labels.size();
  int dim = h_data.size() / N;
  printf("N:%d, dim:%d\n", N, dim);

  std::vector<raft::device_matrix<uint8_t, int64_t>> labeled_datasets;
  labeled_datasets.reserve(cat_freq.size());
  std::vector<raft::device_matrix_view<uint8_t, int64_t>> dataset_views;
  dataset_views.reserve(cat_freq.size());

  std::vector<raft::device_matrix<uint8_t, int64_t>> labeled_queries;
  labeled_datasets.reserve(query_freq.size());
  std::vector<raft::device_matrix_view<uint8_t, int64_t>> query_views;
  dataset_views.reserve(query_freq.size());

  printf("Allocating device datasets for num categories:%d\n", cat_freq.size());
  for(int i=0; i<cat_freq.size(); i++) {
printf("cat:%d, data vecs:%d, queries:%d, totoal size:%d\n", i, cat_freq[i], query_freq[i], (cat_freq[i]+query_freq[i])*dim);
    if(cat_freq[i] > 0 && query_freq[i] > 0) {
      labeled_datasets.push_back(raft::make_device_matrix<uint8_t, int64_t>(dev_resources, cat_freq[i], dim));
      dataset_views.push_back(labeled_datasets[i].view());
      labeled_queries.push_back(raft::make_device_matrix<uint8_t, int64_t>(dev_resources, query_freq[i], dim));
      query_views.push_back(labeled_queries[i].view());
    }
    else {
      labeled_datasets.push_back(raft::make_device_matrix<uint8_t, int64_t>(dev_resources, 1, dim));
      dataset_views.push_back(labeled_datasets[i].view());
      labeled_queries.push_back(raft::make_device_matrix<uint8_t, int64_t>(dev_resources, 0, dim));
      query_views.push_back(labeled_queries[i].view());
      cat_freq[i]=0;
      query_freq[i]=0;
    }
  }

  get_labeled_lists<uint8_t, int64_t>(dev_resources, h_data, data_labels, dataset_views, cat_freq);
  printf("created separate datasets for each label\n");
  get_labeled_lists<uint8_t, int64_t>(dev_resources, h_queries, query_labels, query_views, query_freq);
  printf("created separate query lists\n");

/*
  // Create input arrays.
  int64_t n_samples = 10000;
  int64_t n_dim     = 5;
  int64_t n_queries = 100;
  int64_t topk      = 10;
  int n_categories = CATS;
  int n_data_clusters = 10;
  std::vector<raft::device_matrix<float, int64_t>> datasets;
  datasets.reserve(n_categories);
  std::vector<raft::device_matrix<float, int64_t>> queries;
  queries.reserve(n_categories);

  std::vector<raft::device_matrix_view<float, int64_t>> dataset_views;
  dataset_views.reserve(n_categories);
  std::vector<raft::device_matrix_view<float, int64_t>> query_views;
  query_views.reserve(n_categories);

  for(int i=0; i<n_categories; i++) {
    datasets.push_back(raft::make_device_matrix<float, int64_t>(dev_resources, n_samples, n_dim));
    queries.push_back(raft::make_device_matrix<float, int64_t>(dev_resources, n_queries, n_dim));

    dataset_views.push_back(datasets[i].view());
    query_views.push_back(queries[i].view());

  }
  generate_labeled_datasets(dev_resources, dataset_views, query_views, n_categories, n_data_clusters, 1234ULL);


*/
    // Simple build and search example.
//  int topk = 10;
  cagra_build_search_subgraphs(dev_resources,
                            dataset_views, query_views, cat_freq.size(), topk, itopk);

  printf("finished tests!\n");
  return 0;
}
