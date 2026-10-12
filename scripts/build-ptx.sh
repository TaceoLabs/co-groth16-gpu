#!/usr/bin/env bash
# Regenerates the checked-in PTX of the CUDA kernels (all, or those given, e.g. `ntt`).
# compute_75 PTX is compiled by the driver for any newer GPU at load time. ICICLE_SRC is the
# `icicle` directory of the icicle-snark revision in Cargo.lock.
set -euo pipefail
cd "$(dirname "$0")/.."
ICICLE_SRC=${ICICLE_SRC:-$(ls -d ~/.cargo/git/checkouts/icicle-snark-*/bf00385/icicle | head -1)}
for k in ${*:-spmv msm_sort msm_g1 msm_g2}; do
  nvcc -ptx -O3 -arch=compute_75 -std=c++17 --expt-relaxed-constexpr \
    -I"$ICICLE_SRC/include" -I"$ICICLE_SRC/backend/cuda/include" -DCURVE_ID=1 -DFIELD_ID=1 \
    "src/cuda/$k.cu" -o "src/cuda/$k.ptx"
done
