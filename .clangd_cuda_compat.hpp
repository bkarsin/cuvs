#pragma once

// Clang's CUDA parser still lowers <<<...>>> syntax through this legacy runtime entry point.
// CUDA 12.8+ no longer declares it, so provide the declaration only to clang-based tooling.
#if defined(__clang__)
#include <source_location>
#include <cuda_runtime_api.h>

// CUDA's header only marks these built-ins as device variables under NVRTC. Clang also needs
// that annotation when parsing ordinary CUDA translation units.
#define __STORAGE__ extern const __device__
#include <device_launch_parameters.h>

extern __host__ cudaError_t CUDARTAPI cudaConfigureCall(
  dim3 grid_dim, dim3 block_dim, size_t shared_mem = 0, cudaStream_t stream = nullptr);
#endif
