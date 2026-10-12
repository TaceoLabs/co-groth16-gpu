// Sparse matrix-vector product over a 4x64-bit-limb prime field, used to evaluate the R1CS
// constraint matrices on the device.
//
// The crate loads the PTX compiled from this file (`spmv.ptx`) at runtime; regenerate it with
// `scripts/build-ptx.sh` after changing this file.
//
// Field elements are little-endian limbs. The vector is in standard form (as Icicle keeps
// scalars on the device) and the matrix coefficients are in Montgomery form (as arkworks keeps
// them), so a Montgomery multiplication of the two yields the product in standard form.

#include <cstdint>

struct Fe {
  uint64_t l[4];
};

struct FieldParams {
  uint64_t p[4];
  // -p^-1 mod 2^64
  uint64_t inv;
};

// a * b + c + carry; returns the low word and leaves the high word in `carry`.
__device__ __forceinline__ uint64_t mac(uint64_t a, uint64_t b, uint64_t c, uint64_t& carry)
{
  uint64_t lo = a * b;
  uint64_t hi = __umul64hi(a, b);
  lo += c;
  hi += lo < c;
  lo += carry;
  hi += lo < carry;
  carry = hi;
  return lo;
}

// a + b + carry; returns the sum and leaves the carry (0 or 1) in `carry`.
__device__ __forceinline__ uint64_t adc(uint64_t a, uint64_t b, uint64_t& carry)
{
  uint64_t s = a + b;
  uint64_t c = s < a;
  uint64_t r = s + carry;
  carry = c + (r < s);
  return r;
}

__device__ __forceinline__ bool geq_p(const uint64_t* t, const FieldParams& f)
{
  for (int i = 3; i >= 0; i--) {
    if (t[i] != f.p[i]) return t[i] > f.p[i];
  }
  return true;
}

__device__ __forceinline__ void sub_p(uint64_t* t, const FieldParams& f)
{
  uint64_t borrow = 0;
  for (int i = 0; i < 4; i++) {
    uint64_t d = t[i] - f.p[i];
    uint64_t b = t[i] < f.p[i];
    uint64_t r = d - borrow;
    b += d < borrow;
    t[i] = r;
    borrow = b;
  }
}

// Montgomery multiplication (CIOS): a * b * 2^-256 mod p.
__device__ __forceinline__ Fe mont_mul(const Fe& a, const Fe& b, const FieldParams& f)
{
  uint64_t t[6] = {0, 0, 0, 0, 0, 0};
#pragma unroll
  for (int i = 0; i < 4; i++) {
    uint64_t carry = 0;
#pragma unroll
    for (int j = 0; j < 4; j++)
      t[j] = mac(a.l[j], b.l[i], t[j], carry);
    uint64_t c = 0;
    t[4] = adc(t[4], carry, c);
    t[5] = c;

    uint64_t m = t[0] * f.inv;
    carry = 0;
    mac(m, f.p[0], t[0], carry);
#pragma unroll
    for (int j = 1; j < 4; j++)
      t[j - 1] = mac(m, f.p[j], t[j], carry);
    c = 0;
    t[3] = adc(t[4], carry, c);
    t[4] = t[5] + c;
  }
  if (t[4] != 0 || geq_p(t, f)) sub_p(t, f);
  return Fe{{t[0], t[1], t[2], t[3]}};
}

// a + b mod p, for a, b < p < 2^255.
__device__ __forceinline__ Fe add_mod(const Fe& a, const Fe& b, const FieldParams& f)
{
  uint64_t t[4];
  uint64_t carry = 0;
#pragma unroll
  for (int i = 0; i < 4; i++)
    t[i] = adc(a.l[i], b.l[i], carry);
  if (geq_p(t, f)) sub_p(t, f);
  return Fe{{t[0], t[1], t[2], t[3]}};
}

// out[row] = sum_k table[coeff_idx[k]] * z[cols[k]] for the rows in `row_list`, z being the
// public inputs followed by the witness, L lanes per row.
template <int L>
__device__ __forceinline__ void spmv_rows(
  const uint32_t* __restrict__ row_list,
  uint32_t num_rows,
  const uint32_t* __restrict__ row_ptr,
  const uint32_t* __restrict__ cols,
  const uint32_t* __restrict__ coeff_idx,
  const Fe* __restrict__ table,
  const Fe* __restrict__ public_inputs,
  uint32_t num_public,
  const Fe* __restrict__ witness,
  Fe* __restrict__ out,
  const FieldParams& f)
{
  const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t i = gid / L;
  const uint32_t lane = gid % L;
  const uint32_t row = i < num_rows ? row_list[i] : 0;
  Fe acc = {{0, 0, 0, 0}};
  if (i < num_rows) {
    const uint32_t end = row_ptr[row + 1];
    for (uint32_t k = row_ptr[row] + lane; k < end; k += L) {
      const uint32_t col = cols[k];
      const Fe v = col < num_public ? public_inputs[col] : witness[col - num_public];
      acc = add_mod(acc, mont_mul(v, table[coeff_idx[k]], f), f);
    }
  }
#pragma unroll
  for (int offset = L / 2; offset > 0; offset /= 2) {
    Fe other;
#pragma unroll
    for (int j = 0; j < 4; j++)
      other.l[j] = __shfl_down_sync(0xffffffff, acc.l[j], offset, L);
    acc = add_mod(acc, other, f);
  }
  if (lane == 0 && i < num_rows) out[row] = acc;
}

#define SPMV_KERNEL(L)                                                                             \
  extern "C" __global__ void spmv_l##L(                                                            \
    const uint32_t* __restrict__ row_list, uint32_t num_rows, const uint32_t* __restrict__ row_ptr, \
    const uint32_t* __restrict__ cols, const uint32_t* __restrict__ coeff_idx,                     \
    const Fe* __restrict__ table, const Fe* __restrict__ public_inputs, uint32_t num_public,       \
    const Fe* __restrict__ witness, Fe* __restrict__ out, FieldParams f)                           \
  {                                                                                                \
    spmv_rows<L>(                                                                                  \
      row_list, num_rows, row_ptr, cols, coeff_idx, table, public_inputs, num_public, witness, out, f); \
  }

SPMV_KERNEL(1)
SPMV_KERNEL(2)
SPMV_KERNEL(4)
SPMV_KERNEL(8)
SPMV_KERNEL(16)
SPMV_KERNEL(32)
