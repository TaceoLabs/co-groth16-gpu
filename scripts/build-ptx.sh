#!/usr/bin/env bash
# Regenerates the checked-in PTX of the CUDA kernels; run after changing src/cuda/*.cu.
# compute_75 PTX is compiled by the driver for any newer GPU at load time.
set -euo pipefail
cd "$(dirname "$0")/.."
nvcc -ptx -O3 -arch=compute_75 src/cuda/spmv.cu -o src/cuda/spmv.ptx
