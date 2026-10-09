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

// out[row] = sum_k coeffs[k] * z[cols[k]] over row_ptr[row]..row_ptr[row + 1], where the
// assignment z is the public inputs followed by the witness.
extern "C" __global__ void spmv_kernel(
  const uint32_t* __restrict__ row_ptr,
  const uint32_t* __restrict__ cols,
  const Fe* __restrict__ coeffs,
  const Fe* __restrict__ public_inputs,
  uint32_t num_public,
  const Fe* __restrict__ witness,
  uint32_t rows,
  Fe* __restrict__ out,
  FieldParams f)
{
  uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) return;
  Fe acc = {{0, 0, 0, 0}};
  for (uint32_t k = row_ptr[row]; k < row_ptr[row + 1]; k++) {
    uint32_t col = cols[k];
    Fe v = col < num_public ? public_inputs[col] : witness[col - num_public];
    acc = add_mod(acc, mont_mul(v, coeffs[k], f), f);
  }
  out[row] = acc;
}
