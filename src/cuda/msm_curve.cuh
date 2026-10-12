// The MSM kernels of a curve, for the types P (projective), A (affine) and F (coordinates).
#pragma once
#include "msm.cuh"

// x = X / ZZ, y = Y / ZZZ; infinity is ZZ = 0. Faster than Icicle's formulas in G2, not in G1.
struct Xyzz {
  F x, y, zz, zzz;
};

// Icicle's affine infinity (pads the l query).
__device__ __forceinline__ bool is_infinity(const A& q) { return q.x == F::zero() && q.y == F::zero(); }

// madd-2008-s
__device__ __forceinline__ void xyzz_add_affine(Xyzz& p, const A& q)
{
  if (is_infinity(q)) return;
  if (p.zz == F::zero()) {
    p.x = q.x;
    p.y = q.y;
    p.zz = F::one();
    p.zzz = F::one();
    return;
  }
  const F u2 = q.x * p.zz;
  const F s2 = q.y * p.zzz;
  const F P_ = u2 - p.x;
  const F R = s2 - p.y;
  if (P_ == F::zero()) {
    if (R == F::zero()) { // p == q: mdbl-2008-s
      const F U = q.y + q.y;
      const F V = F::sqr(U);
      const F W = U * V;
      const F Sv = q.x * V;
      const F xx = F::sqr(q.x);
      const F M = xx + xx + xx;
      const F X3 = F::sqr(M) - (Sv + Sv);
      p.y = M * (Sv - X3) - W * q.y;
      p.x = X3;
      p.zz = V;
      p.zzz = W;
    } else { // p == -q
      p.zz = F::zero();
      p.zzz = F::zero();
    }
    return;
  }
  const F PP = F::sqr(P_);
  const F PPP = P_ * PP;
  const F Q = p.x * PP;
  const F X3 = F::sqr(R) - PPP - (Q + Q);
  p.y = R * (Q - X3) - p.y * PPP;
  p.x = X3;
  p.zz = p.zz * PP;
  p.zzz = p.zzz * PPP;
}

// Sum of the bases of pairs [start, start + len), loading the next base while adding one.
__device__ __forceinline__ P
sum_bases(const A* __restrict__ bases, const uint32_t* __restrict__ pairs, uint32_t start, uint32_t len)
{
  const uint32_t end = start + len;
#if XYZZ
  Xyzz acc = {F::zero(), F::zero(), F::zero(), F::zero()};
#else
  P acc = P::zero();
#endif
  if (len == 0) return P::zero();
  A q = bases[pairs[start]];
  uint32_t next = start + 1 < end ? pairs[start + 1] : 0;
  for (uint32_t j = start; j < end; j++) {
    const A current = q;
    if (j + 1 < end) q = bases[next];
    if (j + 2 < end) next = pairs[j + 2];
#if XYZZ
    xyzz_add_affine(acc, current);
#else
    if (!is_infinity(current)) acc = acc + current;
#endif
  }
#if XYZZ
  if (acc.zz == F::zero()) return P::zero();
  return P{acc.x * acc.zzz, acc.y * acc.zz, acc.zz * acc.zzz};
#else
  return acc;
#endif
}

// One thread per work item, MSM blockIdx.y.
#ifndef ACC_MIN_BLOCKS
#define ACC_MIN_BLOCKS 1
#endif
extern "C" __global__ void __launch_bounds__(128, ACC_MIN_BLOCKS) msm_accumulate(
  const A* __restrict__ bases,
  uint64_t bases_stride,
  const uint32_t* __restrict__ pairs,
  const uint4* __restrict__ work,
  const uint32_t* __restrict__ info,
  P* __restrict__ sums,
  uint32_t buckets,
  P* __restrict__ partials,
  uint32_t partials_stride)
{
  const uint32_t k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= info[INFO_WORK]) return;
  const uint32_t m = blockIdx.y;
  const uint4 w = work[k];
  const P sum = sum_bases(bases + m * bases_stride, pairs, w.y, w.z);
  if (k < info[INFO_LARGE_TASKS])
    partials[(uint64_t)m * partials_stride + k] = sum;
  else
    sums[(uint64_t)m * buckets + w.x] = sum;
}

#if XYZZ
// Inlined in G2, the compiler interleaves additions and spills.
__device__ __noinline__ P padd(P a, P b) { return a + b; }
__device__ __noinline__ P pdbl(P a) { return P::dbl(a); }
#else
__device__ __forceinline__ P padd(const P& a, const P& b) { return a + b; }
__device__ __forceinline__ P pdbl(const P& a) { return P::dbl(a); }
#endif

// The sums of the large buckets, a block per bucket.
#define LARGE_THREADS 128
extern "C" __global__ void msm_large_reduce(
  const uint32_t* __restrict__ large_task_offsets,
  const uint32_t* __restrict__ large_list,
  const uint32_t* __restrict__ info,
  const P* __restrict__ partials,
  uint32_t partials_stride,
  P* __restrict__ sums,
  uint32_t buckets)
{
  __shared__ P sh[LARGE_THREADS];
  const uint32_t m = blockIdx.y, t = threadIdx.x;
  partials += (uint64_t)m * partials_stride;
  for (uint32_t k = blockIdx.x; k < info[INFO_LARGE_BUCKETS]; k += gridDim.x) {
    const uint32_t b = large_list[k];
    P acc = P::zero();
    for (uint32_t i = large_task_offsets[b] + t; i < large_task_offsets[b + 1]; i += LARGE_THREADS)
      acc = padd(acc, partials[i]);
    sh[t] = acc;
    __syncthreads();
    for (uint32_t step = LARGE_THREADS / 2; step > 0; step >>= 1) {
      if (t < step) sh[t] = padd(sh[t], sh[t + step]);
      __syncthreads();
    }
    if (t == 0) sums[(uint64_t)m * buckets + b] = sh[0];
    __syncthreads();
  }
}

// sum_b b * B_b in levels, keeping result = sum_i Z_i + sum_i i * T_i (T = bucket sums, Z = 0 at
// first): over segments s of 2^seg_log items, T'_s = 2^seg_log * sum_j T_i and
// Z'_s = sum_j Z_i + sum_j j * T_i, until one item is left.


// The first level, over the bucket sums.
extern "C" __global__ void msm_reduce_first(
  const P* __restrict__ T,
  const uint32_t* __restrict__ offsets,
  uint32_t n_out,
  uint32_t seg_log,
  P* __restrict__ T_out,
  P* __restrict__ Z_out)
{
  const uint32_t s = blockIdx.x * blockDim.x + threadIdx.x;
  if (s >= n_out) return;
  const uint32_t m = blockIdx.y;
  T += (uint64_t)m * ((uint64_t)n_out << seg_log);
  const uint32_t lo = s << seg_log;
  P running = P::zero(), acc = P::zero();
  for (uint32_t j = (1u << seg_log) - 1;; j--) {
    const uint32_t i = lo + j;
    if (offsets[i + 1] != offsets[i]) running = padd(running, T[i]);
    if (j == 0) break;
    acc = padd(acc, running);
  }
  Z_out[(uint64_t)m * n_out + s] = acc;
  if (T_out) {
    for (uint32_t k = 0; k < seg_log; k++)
      running = pdbl(running);
    T_out[(uint64_t)m * n_out + s] = running;
  }
}

// A later level, over pairs.
extern "C" __global__ void msm_reduce_pairs(
  const P* __restrict__ T, const P* __restrict__ Z, uint32_t n_out, P* __restrict__ T_out, P* __restrict__ Z_out)
{
  const uint32_t s = blockIdx.x * blockDim.x + threadIdx.x;
  if (s >= n_out) return;
  const uint64_t i = (uint64_t)blockIdx.y * 2 * n_out + 2 * s, o = (uint64_t)blockIdx.y * n_out + s;
  const P t1 = T[i + 1];
  Z_out[o] = padd(padd(Z[i], Z[i + 1]), t1);
  if (T_out) T_out[o] = pdbl(padd(T[i], t1));
}
