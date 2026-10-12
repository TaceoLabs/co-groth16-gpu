// Counting sort of the (digit, base) pairs of an MSM by digit; see src/msm.rs.

#include "msm.cuh"

typedef scalar_t S;

extern "C" __global__ void msm_zero(uint32_t* p, uint32_t n)
{
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] = 0;
}

// counters[key]++ for the lanes with key != 0, aggregated over the warp; returns the lane's slot.
// The whole warp must call it.
__device__ __forceinline__ uint32_t warp_inc(uint32_t* counters, uint32_t key)
{
  const uint32_t peers = __match_any_sync(0xffffffffu, key);
  const uint32_t lane = threadIdx.x & 31;
  const uint32_t leader = __ffs(peers) - 1;
  uint32_t base = 0;
  if (key != 0 && lane == leader) base = atomicAdd(&counters[key], __popc(peers));
  base = __shfl_sync(0xffffffffu, base, leader);
  return base + __popc(peers & ((1u << lane) - 1));
}

// counts[d] = number of nonzero digits d.
extern "C" __global__ void
msm_count(const S* __restrict__ scalars, uint32_t n, uint32_t c, uint32_t digits, uint32_t* __restrict__ counts)
{
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  const S s = i < n ? scalars[i] : S::zero();
  for (uint32_t k = 0; k < digits; k++)
    warp_inc(counts, s.get_scalar_digit(k, c));
}

// Scan of the bucket sizes over tiles: bucket offsets, the first task of every large bucket (of
// more than `large` pairs), where the buckets of every smaller size start in the work list, and
// the counts reset to the offsets, as the cursors of msm_scatter.
#define SCAN_TILE 1024
#define SCAN_THREADS 1024

__device__ __forceinline__ uint32_t tasks_of(uint32_t count, uint32_t large)
{
  return count > large ? (count + large - 1) / large : 0;
}

// Inclusive scan of x over the block (SCAN_THREADS threads).
__device__ __forceinline__ uint32_t block_scan(uint32_t x, uint32_t* sh)
{
  const uint32_t t = threadIdx.x;
  sh[t] = x;
  __syncthreads();
  for (uint32_t step = 1; step < SCAN_THREADS; step <<= 1) {
    const uint32_t y = t >= step ? sh[t - step] : 0;
    __syncthreads();
    sh[t] += y;
    __syncthreads();
  }
  const uint32_t r = sh[t];
  __syncthreads();
  return r;
}

// tile_sums[tile] = (pairs, large tasks); hist[s] += non-empty buckets of size s <= large.
extern "C" __global__ void msm_scan_tiles(
  const uint32_t* __restrict__ counts, uint32_t large, uint2* __restrict__ tile_sums, uint32_t* __restrict__ hist)
{
  __shared__ uint32_t sh_hist[SCAN_THREADS], sh[SCAN_THREADS];
  const uint32_t t = threadIdx.x;
  sh_hist[t] = 0;
  __syncthreads();
  const uint32_t count = counts[blockIdx.x * SCAN_TILE + t];
  if (count && count <= large) atomicAdd(&sh_hist[count], 1);
  const uint32_t pairs = block_scan(count, sh);
  const uint32_t tasks = block_scan(tasks_of(count, large), sh);
  if (t == SCAN_THREADS - 1) tile_sums[blockIdx.x] = make_uint2(pairs, tasks);
  if (sh_hist[t]) atomicAdd(&hist[t], sh_hist[t]);
}

// In one block: the scan of the tile sums, the class cursors and the info.
extern "C" __global__ void msm_scan_top(
  uint2* __restrict__ tile_sums,
  uint32_t tiles,
  uint32_t large,
  const uint32_t* __restrict__ hist,
  uint32_t* __restrict__ class_cursors,
  uint32_t* __restrict__ info)
{
  __shared__ uint32_t sh[SCAN_THREADS];
  const uint32_t t = threadIdx.x;
  uint32_t pairs = 0, tasks = 0;
  for (uint32_t base = 0; base < tiles; base += SCAN_THREADS) {
    const uint2 x = base + t < tiles ? tile_sums[base + t] : make_uint2(0, 0);
    const uint32_t p = block_scan(x.x, sh), q = block_scan(x.y, sh);
    if (base + t < tiles) tile_sums[base + t] = make_uint2(pairs + p - x.x, tasks + q - x.y);
    // The totals of this chunk: the last thread's inclusive values.
    if (t == SCAN_THREADS - 1) {
      sh[0] = p;
      sh[1] = q;
    }
    __syncthreads();
    pairs += sh[0];
    tasks += sh[1];
    __syncthreads();
  }
  // Suffix sums of the size histogram, as a scan over the reversed sizes.
  const uint32_t h = hist[SCAN_THREADS - 1 - t];
  const uint32_t suffix = block_scan(h, sh); // sum over sizes >= SCAN_THREADS - 1 - t
  const uint32_t size = SCAN_THREADS - 1 - t;
  if (size <= large) class_cursors[size] = tasks + suffix - h;
  if (size == 1) info[INFO_WORK] = tasks + suffix;
  if (t == 0) {
    info[INFO_PAIRS] = pairs;
    info[INFO_LARGE_TASKS] = tasks;
    info[INFO_LARGE_BUCKETS] = 0;
  }
}

// The scans within the tiles.
extern "C" __global__ void msm_scan_apply(
  uint32_t* __restrict__ counts,
  uint32_t large,
  const uint2* __restrict__ tile_sums,
  uint32_t* __restrict__ offsets,
  uint32_t* __restrict__ large_task_offsets,
  uint32_t buckets)
{
  __shared__ uint32_t sh[SCAN_THREADS];
  const uint32_t b = blockIdx.x * SCAN_TILE + threadIdx.x;
  const uint32_t count = counts[b];
  const uint32_t tasks = tasks_of(count, large);
  const uint2 base = tile_sums[blockIdx.x];
  const uint32_t offset = base.x + block_scan(count, sh) - count;
  const uint32_t task_offset = base.y + block_scan(tasks, sh) - tasks;
  offsets[b] = offset;
  large_task_offsets[b] = task_offset;
  counts[b] = offset;
  if (b == buckets - 1) {
    offsets[buckets] = offset + count;
    large_task_offsets[buckets] = task_offset + tasks;
  }
}

// pairs[cursor[d]++] = i * pf + k for every nonzero digit d = digit k of scalar i.
extern "C" __global__ void msm_scatter(
  const S* __restrict__ scalars,
  uint32_t n,
  uint32_t c,
  uint32_t digits,
  uint32_t pf,
  uint32_t* __restrict__ cursors,
  uint32_t* __restrict__ pairs)
{
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  const S s = i < n ? scalars[i] : S::zero();
  for (uint32_t k = 0; k < digits; k++) {
    const uint32_t d = s.get_scalar_digit(k, c);
    const uint32_t slot = warp_inc(cursors, d);
    if (d) pairs[slot] = i * pf + k;
  }
}

// The work list: the large buckets' tasks, then the other buckets by decreasing size, so that
// the threads of a warp get about as many pairs.
extern "C" __global__ void msm_order(
  const uint32_t* __restrict__ offsets,
  uint32_t buckets,
  uint32_t large,
  const uint32_t* __restrict__ large_task_offsets,
  uint32_t* __restrict__ class_cursors,
  uint4* __restrict__ work,
  uint32_t* __restrict__ large_list,
  uint32_t* __restrict__ info)
{
  const uint32_t b = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t start = b < buckets ? offsets[b] : 0;
  const uint32_t count = b < buckets ? offsets[b + 1] - start : 0;
  const uint32_t slot = warp_inc(class_cursors, count <= large ? count : 0);
  if (count == 0) return;
  if (count <= large) {
    work[slot] = make_uint4(b, start, count, 0);
    return;
  }
  const uint32_t first = large_task_offsets[b];
  for (uint32_t j = 0; j * large < count; j++)
    work[first + j] = make_uint4(b, start + j * large, min(large, count - j * large), 0);
  large_list[atomicAdd(&info[INFO_LARGE_BUCKETS], 1)] = b;
}
