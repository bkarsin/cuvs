#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
# Shared environment resolution for the portable GPiPNN bench scripts. Source it; do not run it
# as part of a pipeline. Running it directly prints what it resolved, which is the fastest way to
# diagnose a machine it does not fit:
#
#   cpp/bench/gpipnn/portable/gpipnn_env.sh
#
# The originals in the parent directory hardcode one workstation -- repo at
# /home/bkarsin/PRs/gpipnn/cuvs, conda at /media/data/miniforge3, env `gpipnn_poc`, datasets under
# /media/data/datasets, and `compute_120a`. Every one of those is resolved or overridable here.
#
# Resolved automatically:
#   GPIPNN_REPO       repo root, from this script's own location
#   GPIPNN_BUILD_DIR  $GPIPNN_REPO/cpp/build
#   GPIPNN_ARCH       the --generate-code flag, read from the build's own compile_commands.json
#   GPIPNN_DATASETS   /scratch (so GIST resolves to /scratch/gist)
#
# Override any of them in the environment. Per-dataset directory overrides also work and win over
# GPIPNN_DATASETS, e.g. GPIPNN_DIR_SIFT10M=/scratch/sift10m.
#
# The conda environment is NOT activated here. Activate it yourself before calling these scripts;
# this only checks that CONDA_PREFIX points somewhere usable, because guessing an env name is how
# the originals became unportable in the first place.

# shellcheck disable=SC2034

GPIPNN_ENV_SELF="${BASH_SOURCE[0]}"
GPIPNN_REPO="${GPIPNN_REPO:-$(cd "$(dirname "${GPIPNN_ENV_SELF}")/../../../.." && pwd)}"
GPIPNN_BUILD_DIR="${GPIPNN_BUILD_DIR:-${GPIPNN_REPO}/cpp/build}"
GPIPNN_BENCH_DIR="${GPIPNN_BENCH_DIR:-${GPIPNN_REPO}/cpp/bench/gpipnn}"
GPIPNN_DATASETS="${GPIPNN_DATASETS:-/scratch}"

gpipnn_die() {
  echo "gpipnn_env: $*" >&2
  return 1
}

# The architecture libcuvs was actually compiled for, not a guess from the installed card.
#
# Read from the build's compile_commands.json, scoped to the GPiPNN translation unit, so the bench
# binary is generated for exactly the architecture the library it links against was. A mismatch
# here is not a compile error -- it surfaces much later as a kernel that will not launch.
#
# Falls back to the card's own compute capability when there is no compile database. `a` is
# appended from 9.0 on, matching what RAPIDS CMake selects for those families; CUTLASS paths in
# cuVS need the arch-accelerated variant rather than the portable one.
gpipnn_detect_arch() {
  local db="${GPIPNN_BUILD_DIR}/compile_commands.json"
  local flag=""

  if [[ -f "${db}" ]] && command -v python3 >/dev/null 2>&1; then
    flag="$(python3 - "${db}" <<'PY'
import json, re, sys, collections
try:
    entries = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(0)
pat = re.compile(r'--generate-code=arch=[^\s"]+')
# Prefer the flag on a GPiPNN source; fall back to the most common real (non-lto) flag.
counts = collections.Counter()
for e in entries:
    cmd = e.get('command') or ' '.join(e.get('arguments', []))
    found = [m for m in pat.findall(cmd) if 'lto_' not in m]
    if not found:
        continue
    if 'gpipnn' in (e.get('file') or ''):
        print(found[0]); sys.exit(0)
    for m in found:
        counts[m] += 1
if counts:
    print(counts.most_common(1)[0][0])
PY
)"
  fi

  if [[ -z "${flag}" ]] && command -v nvidia-smi >/dev/null 2>&1; then
    local cap major minor suffix
    cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')"
    if [[ "${cap}" =~ ^([0-9]+)\.([0-9]+)$ ]]; then
      major="${BASH_REMATCH[1]}"; minor="${BASH_REMATCH[2]}"
      suffix=""
      (( major >= 9 )) && suffix="a"
      flag="--generate-code=arch=compute_${major}${minor}${suffix},code=[sm_${major}${minor}${suffix}]"
      echo "gpipnn_env: no compile_commands.json; inferred arch from the installed card" >&2
    fi
  fi

  [[ -n "${flag}" ]] || return 1
  printf '%s\n' "${flag}"
}

GPIPNN_ARCH="${GPIPNN_ARCH:-$(gpipnn_detect_arch || true)}"

# Dataset directory for a manifest's dataset column.
#
# sift_bench.cu finds its files through cuVS-Bench naming with generic fallbacks, so a directory
# only has to hold base/query/groundtruth under one of the names it searches.
gpipnn_dataset_dir() {
  local ds="$1" var
  var="GPIPNN_DIR_${ds^^}"
  if [[ -n "${!var:-}" ]]; then printf '%s\n' "${!var%/}/"; return 0; fi
  case "${ds}" in
    sift)    printf '%s/sift/\n' "${GPIPNN_DATASETS%/}" ;;
    gist)    printf '%s/gist/\n' "${GPIPNN_DATASETS%/}" ;;
    sift10m) printf '%s/sift10m/\n' "${GPIPNN_DATASETS%/}" ;;
    cohere)  printf '%s/cohere_768/\n' "${GPIPNN_DATASETS%/}" ;;
    *) return 2 ;;
  esac
}

# Everything the bench needs before it can run at all.
gpipnn_check_env() {
  local ok=0
  [[ -n "${CONDA_PREFIX:-}" ]] || { gpipnn_die "CONDA_PREFIX is unset -- activate the cuVS build environment first"; ok=1; }
  [[ -d "${GPIPNN_REPO}/cpp/include/cuvs" ]] || { gpipnn_die "no cuVS headers under ${GPIPNN_REPO}/cpp/include -- set GPIPNN_REPO"; ok=1; }
  [[ -f "${GPIPNN_BUILD_DIR}/libcuvs.so" ]] || { gpipnn_die "no libcuvs.so in ${GPIPNN_BUILD_DIR} -- build cuVS first, or set GPIPNN_BUILD_DIR"; ok=1; }
  if [[ ! -d "${GPIPNN_BUILD_DIR}/_deps/rmm-build" &&
        ! -f "${CONDA_PREFIX:-}/lib/librmm.so" ]]; then
    gpipnn_die "RMM is unavailable -- expected _deps/rmm-build or \$CONDA_PREFIX/lib/librmm.so"
    ok=1
  fi
  [[ -n "${GPIPNN_ARCH}" ]] || { gpipnn_die "could not determine a GPU architecture -- set GPIPNN_ARCH, e.g. --generate-code=arch=compute_80,code=[sm_80]"; ok=1; }
  [[ -f "${GPIPNN_REPO}/cpp/src/neighbors/detail/gpipnn/gpipnn_build.cuh" ]] || { gpipnn_die "no GPiPNN sources -- is this checkout on the PiPNN PR branch?"; ok=1; }
  return "${ok}"
}

gpipnn_print_env() {
  echo "repo:      ${GPIPNN_REPO}"
  echo "build:     ${GPIPNN_BUILD_DIR}"
  echo "bench:     ${GPIPNN_BENCH_DIR}"
  echo "datasets:  ${GPIPNN_DATASETS}"
  echo "arch:      ${GPIPNN_ARCH:-<undetermined>}"
  echo "conda:     ${CONDA_PREFIX:-<unset>}"
  echo "libcuvs:   ${GPIPNN_BUILD_DIR}/libcuvs.so $( [[ -f "${GPIPNN_BUILD_DIR}/libcuvs.so" ]] && echo present || echo MISSING )"
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "gpu:       $(nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader | head -1)"
  fi
}

# Executed rather than sourced: report and validate.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  gpipnn_print_env
  echo
  if gpipnn_check_env; then echo "environment OK"; else echo "environment NOT usable (see above)" >&2; exit 1; fi
fi
