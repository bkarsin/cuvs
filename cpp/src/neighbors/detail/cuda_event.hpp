/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/util/cudart_utils.hpp>

#include <cuda_runtime_api.h>

#include <functional>
#include <memory>
#include <utility>

namespace cuvs::neighbors::detail {

/**
 * Move-only owner of a CUDA event.
 */
class cuda_event {
public:
  cuda_event(cuda_event&&)            = default;
  auto operator=(cuda_event&&) -> cuda_event& = default;
  ~cuda_event()                       = default;
  cuda_event(cuda_event const&)       = delete;  // Copying disallowed: one event one owner
  auto operator=(cuda_event const&) -> cuda_event& = delete;

  cuda_event()
    : event_{[]() {
              auto* e = new cudaEvent_t;
              RAFT_CUDA_TRY(cudaEventCreateWithFlags(e, cudaEventDisableTiming));
              return e;
            }(),
            [](cudaEvent_t* e) {
              RAFT_CUDA_TRY_NO_THROW(cudaEventDestroy(*e));
              delete e;
            }}
  {
  }

  [[nodiscard]] auto value() const -> cudaEvent_t { return *event_; }

private:
  std::unique_ptr<cudaEvent_t, std::function<void(cudaEvent_t*)>> event_;
};

}  // namespace cuvs::neighbors::detail
