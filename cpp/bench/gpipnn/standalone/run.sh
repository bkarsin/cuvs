#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../portable/gpipnn_env.sh
source "${HERE}/gpipnn_env.sh"
gpipnn_check_env || exit 1

BIN="${GPIPNN_BUILD_DIR}/gpipnn_standalone"
[[ -x "${BIN}" ]] || {
  echo "missing ${BIN}; run ${HERE}/build.sh first" >&2
  exit 2
}

usage() {
  cat <<EOF
usage: $0 --dataset <dataset.fbin|dataset_dir> [--degree N] [--runs N]
          [--k N] [--itopk N] [--min-recall R]
          [--nsys] [--nsys-output PREFIX] [--gpu-metrics-frequency HZ]

  --nsys                     profile with CUDA tracing, GPU metrics, and GPU memory usage
  --nsys-output PREFIX       report prefix (default: ./gpipnn_standalone_nsys)
  --gpu-metrics-frequency HZ GPU metric sampling rate (default: 10000)
EOF
}

USE_NSYS=0
NSYS_OUTPUT="${PWD}/gpipnn_standalone_nsys"
GPU_METRICS_FREQUENCY=10000
PROGRAM_ARGS=()
while (( $# > 0 )); do
  case "$1" in
    --nsys)
      USE_NSYS=1
      shift
      ;;
    --nsys-output)
      (( $# >= 2 )) || { echo "--nsys-output requires a value" >&2; exit 2; }
      NSYS_OUTPUT="$2"
      shift 2
      ;;
    --gpu-metrics-frequency)
      (( $# >= 2 )) || { echo "--gpu-metrics-frequency requires a value" >&2; exit 2; }
      GPU_METRICS_FREQUENCY="$2"
      shift 2
      ;;
    --help)
      usage
      exit 0
      ;;
    *)
      PROGRAM_ARGS+=("$1")
      shift
      ;;
  esac
done

if (( USE_NSYS == 1 )); then
  command -v nsys >/dev/null 2>&1 || {
    echo "nsys is unavailable; install or load Nsight Systems" >&2
    exit 2
  }
  exec nsys profile \
    --trace=cuda,nvtx,osrt \
    --cuda-memory-usage=true \
    --sample=none \
    --cpuctxsw=none \
    --gpu-metrics-devices=cuda-visible \
    --gpu-metrics-frequency="${GPU_METRICS_FREQUENCY}" \
    --force-overwrite=true \
    --output="${NSYS_OUTPUT}" \
    "${BIN}" "${PROGRAM_ARGS[@]}"
fi

exec "${BIN}" "${PROGRAM_ARGS[@]}"
