#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../portable/gpipnn_env.sh
source "${HERE}/gpipnn_env.sh"
gpipnn_check_env || exit 1

BUILD_DIR="${GPIPNN_BUILD_DIR}"
DEPS="${BUILD_DIR}/_deps"
CONDA_TARGETS="${CONDA_PREFIX}/targets/x86_64-linux/include"
OUTPUT="${BUILD_DIR}/gpipnn_standalone"

DEP_INCLUDES=()
for path in \
  "${DEPS}/rmm-src/cpp/include" \
  "${DEPS}/rmm-build/include" \
  "${DEPS}/raft-src/cpp/include" \
  "${DEPS}/raft-build/include"; do
  [[ -d "${path}" ]] && DEP_INCLUDES+=(-I"${path}")
done
DEP_INCLUDES+=(-isystem "${CONDA_PREFIX}/include")

if [[ -d "${DEPS}/rmm-build" ]]; then
  RMM_LIB_DIR="${DEPS}/rmm-build"
else
  RMM_LIB_DIR="${CONDA_PREFIX}/lib"
fi

echo "--- building GPiPNN standalone for ${GPIPNN_ARCH#--generate-code=}"

nvcc -forward-unknown-to-host-compiler \
  -DCCCL_DISABLE_PDL -DCUB_DISABLE_NAMESPACE_MAGIC -DCUB_IGNORE_NAMESPACE_MAGIC_ERROR \
  -DCUTLASS_NAMESPACE=cuvs_cutlass -DNVTX_ENABLED \
  -DRAFT_LOG_ACTIVE_LEVEL=RAPIDS_LOGGER_LOG_LEVEL_INFO \
  -DRAFT_SYSTEM_LITTLE_ENDIAN=1 -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA \
  -DTHRUST_DISABLE_ABI_NAMESPACE -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP \
  -DTHRUST_IGNORE_ABI_NAMESPACE_ERROR -DRAFT_HIDE_DEPRECATION_WARNINGS \
  -DCUDA_API_PER_THREAD_DEFAULT_STREAM \
  "${DEP_INCLUDES[@]}" \
  -I"${DEPS}/cccl-src/thrust" \
  -I"${DEPS}/cccl-src/libcudacxx/include" \
  -I"${DEPS}/cccl-src/cub" \
  -I"${GPIPNN_REPO}/cpp/include" \
  -I"${BUILD_DIR}/include" \
  -isystem "${CONDA_TARGETS}" \
  -isystem "${CONDA_TARGETS}/cccl" \
  -O3 -DNDEBUG -std=c++20 \
  ${GPIPNN_ARCH} \
  --expt-extended-lambda --expt-relaxed-constexpr \
  -Xcompiler=-Wno-deprecated-declarations -Xcompiler=-fopenmp -Xcompiler=-pthread \
  -x cu "${HERE}/standalone.cu" \
  -L"${BUILD_DIR}" -lcuvs \
  -L"${RMM_LIB_DIR}" -lrmm \
  -L"${CONDA_PREFIX}/lib" -lcublas -lcublasLt -lrapids_logger \
  -Wl,-rpath,"${BUILD_DIR}" -Wl,-rpath,"${RMM_LIB_DIR}" -Wl,-rpath,"${CONDA_PREFIX}/lib" \
  -o "${OUTPUT}"

echo "    built ${OUTPUT}"
