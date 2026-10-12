// The witness map NTTs (CircomReduction): inverse NTT by decimation in frequency (bit-reversed
// out), coset scaling, forward NTT by decimation in time (bit-reversed in). An NTT runs in passes
// over at most TILE_LOG bits of the index (four-step FFT): pass [lo, hi) runs sub-NTTs in shared
// memory and multiplies element (L = bits below lo, p) by w_{2^hi}^(L * bitrev(p)), after them
// in the inverse NTT and before them in the forward one.

#include <cstdint>
#include "icicle/curves/curve_config.h"

using namespace curve_config;

typedef scalar_t S;

#define TILE_LOG 10
#define TILE (1 << TILE_LOG)
#define THREADS 256
// w_N^e = lo[e % 2^TW_LOG] * hi[e / 2^TW_LOG]
#define TW_LOG 12

__device__ __forceinline__ uint32_t bitrev(uint32_t x, uint32_t bits) { return bits ? __brev(x) >> (32 - bits) : 0; }

__device__ __forceinline__ S twiddle(const S* __restrict__ lo, const S* __restrict__ hi, uint32_t e)
{
  return lo[e & ((1 << TW_LOG) - 1)] * hi[e >> TW_LOG];
}

__device__ __forceinline__ S* poly(S* p0, S* p1, S* p2) { return blockIdx.y == 0 ? p0 : blockIdx.y == 1 ? p1 : p2; }

// A tile holds 1 << log_g groups of 2^r elements, interleaved if STRIDED.
template <bool STRIDED>
__device__ __forceinline__ uint32_t tile_index(uint32_t p, uint32_t g, uint32_t r, uint32_t log_g)
{
  return STRIDED ? (p << log_g | g) : (g << r | p);
}

// The twiddle of level s (pairs 2^s apart) at position j.
__device__ __forceinline__ S level_twiddle(const S* __restrict__ sub, uint32_t s, uint32_t j)
{
  return sub[j << (TILE_LOG - 1 - s)];
}

// Decimation in frequency over the r bits of every group, two levels at a time.
template <bool STRIDED>
__device__ __forceinline__ void dif(S* tile, uint32_t r, uint32_t log_g, const S* __restrict__ sub)
{
  int s = r - 1;
  // Levels s and s - 1.
  for (; s >= 1; s -= 2) {
    __syncthreads();
    const uint32_t h = 1u << (s - 1);
    for (uint32_t b = threadIdx.x; b < 1u << (r - 2 + log_g); b += THREADS) {
      const uint32_t g = STRIDED ? b & ((1u << log_g) - 1) : b >> (r - 2);
      const uint32_t q = STRIDED ? b >> log_g : b & ((1u << (r - 2)) - 1);
      const uint32_t j = q & (h - 1);
      const uint32_t p0 = (q >> (s - 1) << (s + 1)) | j;
      const uint32_t i0 = tile_index<STRIDED>(p0, g, r, log_g);
      const uint32_t i1 = tile_index<STRIDED>(p0 + h, g, r, log_g);
      const uint32_t i2 = tile_index<STRIDED>(p0 + 2 * h, g, r, log_g);
      const uint32_t i3 = tile_index<STRIDED>(p0 + 3 * h, g, r, log_g);
      const S x0 = tile[i0], x1 = tile[i1], x2 = tile[i2], x3 = tile[i3];
      const S a0 = x0 + x2, a1 = x1 + x3;
      const S a2 = (x0 - x2) * level_twiddle(sub, s, j);
      const S a3 = (x1 - x3) * level_twiddle(sub, s, j + h);
      tile[i0] = a0 + a1;
      tile[i2] = a2 + a3;
      if (s == 1) {
        tile[i1] = a0 - a1;
        tile[i3] = a2 - a3;
      } else {
        const S w = level_twiddle(sub, s - 1, j);
        tile[i1] = (a0 - a1) * w;
        tile[i3] = (a2 - a3) * w;
      }
    }
  }
  if (s == 0) {
    __syncthreads();
    for (uint32_t b = threadIdx.x; b < 1u << (r - 1 + log_g); b += THREADS) {
      const uint32_t g = STRIDED ? b & ((1u << log_g) - 1) : b >> (r - 1);
      const uint32_t q = STRIDED ? b >> log_g : b & ((1u << (r - 1)) - 1);
      const uint32_t i0 = tile_index<STRIDED>(2 * q, g, r, log_g);
      const uint32_t i1 = tile_index<STRIDED>(2 * q + 1, g, r, log_g);
      const S u = tile[i0], v = tile[i1];
      tile[i0] = u + v;
      tile[i1] = u - v;
    }
  }
  __syncthreads();
}

// Decimation in time, the transpose of `dif`.
template <bool STRIDED>
__device__ __forceinline__ void dit(S* tile, uint32_t r, uint32_t log_g, const S* __restrict__ sub)
{
  uint32_t s = 1;
  if (r % 2) {
    __syncthreads();
    for (uint32_t b = threadIdx.x; b < 1u << (r - 1 + log_g); b += THREADS) {
      const uint32_t g = STRIDED ? b & ((1u << log_g) - 1) : b >> (r - 1);
      const uint32_t q = STRIDED ? b >> log_g : b & ((1u << (r - 1)) - 1);
      const uint32_t i0 = tile_index<STRIDED>(2 * q, g, r, log_g);
      const uint32_t i1 = tile_index<STRIDED>(2 * q + 1, g, r, log_g);
      const S u = tile[i0], v = tile[i1];
      tile[i0] = u + v;
      tile[i1] = u - v;
    }
    s = 2;
  }
  // Levels s - 1 and s.
  for (; s < r; s += 2) {
    __syncthreads();
    const uint32_t h = 1u << (s - 1);
    for (uint32_t b = threadIdx.x; b < 1u << (r - 2 + log_g); b += THREADS) {
      const uint32_t g = STRIDED ? b & ((1u << log_g) - 1) : b >> (r - 2);
      const uint32_t q = STRIDED ? b >> log_g : b & ((1u << (r - 2)) - 1);
      const uint32_t j = q & (h - 1);
      const uint32_t p0 = (q >> (s - 1) << (s + 1)) | j;
      const uint32_t i0 = tile_index<STRIDED>(p0, g, r, log_g);
      const uint32_t i1 = tile_index<STRIDED>(p0 + h, g, r, log_g);
      const uint32_t i2 = tile_index<STRIDED>(p0 + 2 * h, g, r, log_g);
      const uint32_t i3 = tile_index<STRIDED>(p0 + 3 * h, g, r, log_g);
      const S x0 = tile[i0], x2 = tile[i2];
      S x1 = tile[i1], x3 = tile[i3];
      if (s > 1) {
        const S w = level_twiddle(sub, s - 1, j);
        x1 = x1 * w;
        x3 = x3 * w;
      }
      const S a0 = x0 + x1, a1 = x0 - x1, a2 = x2 + x3, a3 = x2 - x3;
      const S b2 = a2 * level_twiddle(sub, s, j);
      const S b3 = a3 * level_twiddle(sub, s, j + h);
      tile[i0] = a0 + b2;
      tile[i2] = a0 - b2;
      tile[i1] = a1 + b3;
      tile[i3] = a1 - b3;
    }
  }
  __syncthreads();
}

// A pass over bits [lo, lo + r), lo > 0, of polynomial blockIdx.y.
template <bool INVERSE>
__device__ __forceinline__ void strided_pass(
  S* data,
  uint32_t log_n,
  uint32_t lo,
  uint32_t r,
  const S* __restrict__ sub,
  const S* __restrict__ tw_lo,
  const S* __restrict__ tw_hi)
{
  __shared__ S tile[TILE];
  const uint32_t hi = lo + r;
  const uint32_t log_g = min(TILE_LOG - r, lo);
  const uint32_t size = 1u << (r + log_g);
  const uint32_t tile_lo = blockIdx.x & ((1u << (lo - log_g)) - 1);
  const uint32_t tile_hi = blockIdx.x >> (lo - log_g);
  const uint32_t l0 = tile_lo << log_g;
  S* base = data + ((uint64_t)tile_hi << hi) + l0;
  const uint32_t shift = log_n - hi;

  for (uint32_t i = threadIdx.x; i < size; i += THREADS) {
    const uint32_t p = i >> log_g, g = i & ((1u << log_g) - 1);
    S x = base[((uint64_t)p << lo) + g];
    if (!INVERSE) x = x * twiddle(tw_lo, tw_hi, ((l0 + g) * bitrev(p, r)) << shift);
    tile[i] = x;
  }
  if (INVERSE)
    dif<true>(tile, r, log_g, sub);
  else
    dit<true>(tile, r, log_g, sub);
  for (uint32_t i = threadIdx.x; i < size; i += THREADS) {
    const uint32_t p = i >> log_g, g = i & ((1u << log_g) - 1);
    S x = tile[i];
    if (INVERSE) x = x * twiddle(tw_lo, tw_hi, ((l0 + g) * bitrev(p, r)) << shift);
    base[((uint64_t)p << lo) + g] = x;
  }
}

extern "C" __global__ void ntt_inverse_pass(
  S* a,
  S* b,
  S* c,
  uint32_t log_n,
  uint32_t lo,
  uint32_t r,
  const S* __restrict__ sub,
  const S* __restrict__ tw_lo,
  const S* __restrict__ tw_hi)
{
  strided_pass<true>(poly(a, b, c), log_n, lo, r, sub, tw_lo, tw_hi);
}

extern "C" __global__ void ntt_forward_pass(
  S* a,
  S* b,
  S* c,
  uint32_t log_n,
  uint32_t lo,
  uint32_t r,
  const S* __restrict__ sub,
  const S* __restrict__ tw_lo,
  const S* __restrict__ tw_hi)
{
  strided_pass<false>(poly(a, b, c), log_n, lo, r, sub, tw_lo, tw_hi);
}

// The pass over bits [0, r): the end of the inverse NTT, the scaling by g^m / N and the start of
// the forward NTT.
extern "C" __global__ void ntt_coset_pass(
  S* a,
  S* b,
  S* c,
  uint32_t log_n,
  uint32_t r,
  const S* __restrict__ sub_inv,
  const S* __restrict__ sub_fwd,
  const S* __restrict__ coset_lo,
  const S* __restrict__ coset_hi)
{
  __shared__ S tile[TILE];
  const uint32_t log_size = min(TILE_LOG, log_n);
  const uint32_t size = 1u << log_size;
  const uint32_t log_g = log_size - r;
  S* base = poly(a, b, c) + ((uint64_t)blockIdx.x << log_size);

  for (uint32_t i = threadIdx.x; i < size; i += THREADS)
    tile[i] = base[i];
  dif<false>(tile, r, log_g, sub_inv);
  for (uint32_t i = threadIdx.x; i < size; i += THREADS)
    tile[i] = tile[i] * twiddle(coset_lo, coset_hi, bitrev((blockIdx.x << log_size) | i, log_n));
  dit<false>(tile, r, log_g, sub_fwd);
  for (uint32_t i = threadIdx.x; i < size; i += THREADS)
    base[i] = tile[i];
}

// c = a * b
extern "C" __global__ void ntt_mul(const S* __restrict__ a, const S* __restrict__ b, S* __restrict__ c, uint32_t n)
{
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] * b[i];
}

// h = a * b - c
extern "C" __global__ void
ntt_mul_sub(const S* __restrict__ a, const S* __restrict__ b, const S* __restrict__ c, S* __restrict__ h, uint32_t n)
{
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) h[i] = a[i] * b[i] - c[i];
}
