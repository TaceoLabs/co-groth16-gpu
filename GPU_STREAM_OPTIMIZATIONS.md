# GPU stream/pipelining optimizations

Findings from a review of the ICICLE GPU prover pipeline (`src/mpc.rs`, `src/mpc/shamir.rs`,
`src/mpc/rep3.rs`, `src/groth16_gpu.rs`, `src/groth16_gpu/reduction.rs`, `src/bridges.rs`,
`src/gpu_utils.rs`). These are performance-only observations — the underlying MPC protocol logic
is correct (verified separately against the CPU `co-groth16` reference). Ordered by expected
impact; A and B are believed to be actively undoing multi-stream parallelism the code already
sets up elsewhere.

## A. Drop the internal `stream.synchronize()` in `local_mul_vec`

**Where:** [`src/mpc/shamir.rs:169-186`](src/mpc/shamir.rs#L169), identically in
[`src/mpc/rep3.rs:188-227`](src/mpc/rep3.rs#L188).

```rust
mul_scalars(a, b, result.as_mut_slice(), &cfg).unwrap();
stream
    .synchronize()
    .expect("Failed to synchronize local_mul_vec stream");
result
```

**Problem:** `witness_map_from_r1cs_eval` ([`src/groth16_gpu/reduction.rs`](src/groth16_gpu/reduction.rs))
allocates three named streams (`ReductionStreams { a, b, c }`, lines 20-46) specifically so the
`a`, `b`, and `c` computations can run concurrently. But the pipeline's first op is:

```rust
let mut c = T::local_mul_vec::<B>(eval_a, eval_b, state, stream_c);   // line 112 — blocks here
T::ifft_in_place(eval_a, stream_a, None);                              // line 115 — can't be issued until c unblocks
```

The internal synchronize forces the host to block before it can even *enqueue* the `a`/`b` IFFTs,
so streams `a` and `b` never overlap with `c`. It's also unnecessary for correctness: every
subsequent op on the result runs on the same stream, so CUDA's in-order stream semantics already
guarantee correctness without a host-side wait. The caller already places real syncs where they're
needed (e.g. `stream_b.synchronize()` before computing `ab`, reduction.rs:129).

**Fix:** remove the `stream.synchronize()` call from both `local_mul_vec` implementations. Let the
existing caller-side synchronization points (already correct) be the only sync points.

**Risk:** low. No correctness dependency on the removed sync; only touches two functions.

## B. `ark_to_icicle_scalars` hardcodes the CUDA null stream

**Where:** [`src/bridges.rs:54-66`](src/bridges.rs#L54)

```rust
pub(crate) fn ark_to_icicle_scalars<T, I>(ark_scalars: DeviceVec<T>) -> eyre::Result<DeviceVec<I>>
where
    T: PrimeField,
    I: FieldImpl + MontgomeryConvertible,
{
    let mut icicle_scalars = unsafe { transmute::<DeviceVec<T>, DeviceVec<I>>(ark_scalars) };
    I::from_mont(&mut icicle_scalars, &IcicleStream::default());   // <-- always the null stream
    Ok(icicle_scalars)
}
```

**Problem:** `IcicleStream::default()` has `handle: null` (icicle-runtime `stream.rs`) — the CUDA
legacy default stream. Under legacy stream semantics, any op on the null stream implicitly
synchronizes against *every other stream on the device* (waits for their prior work, blocks their
subsequent work). `MontgomeryConvertible::from_mont` already accepts a real stream
(`fn from_mont(values, stream: &IcicleStream)`); it's just not being forwarded.

This function is called from:
- `shares_to_device` / `half_shares_to_device` (both drivers, during `setup()`)
- the public-input conversion in `setup()` ([`src/groth16_gpu.rs:132`](src/groth16_gpu.rs#L132))
- Rep3's `local_mul_vec` masking-element conversion — **mid-pipeline**, on `stream_c`
  ([`src/mpc/rep3.rs:194-200`](src/mpc/rep3.rs#L194))

Each call is a hidden whole-device barrier.

**Fix:** add a `stream: &IcicleStream` parameter to `ark_to_icicle_scalars` and thread the caller's
actual stream through instead of `IcicleStream::default()`. Update all call sites.

**Risk:** low-medium — mechanical signature change, touches several call sites in `shamir.rs`,
`rep3.rs`, `groth16_gpu.rs`.

## C. Synchronous host→device transfers in `setup()`

**Where:** [`src/groth16_gpu.rs:99-144`](src/groth16_gpu.rs#L99) (`CoGroth16Icicle::setup`)

`evaluate_constraints` → `shares_to_device` (witness) → public-input conversion run strictly back
to back, each timed separately (`eval_timer`, `witness_timer`, `public_timer`). `shares_to_device`
/ `half_shares_to_device` use the blocking `from_host_slice` helper
([`src/gpu_utils.rs:75-82`](src/gpu_utils.rs#L75): plain `device_malloc` + `copy_from_host`), even
though an async counterpart already exists and is used elsewhere (`from_host_slice_async`, see
`upload_points_async`).

**Problem:** these three uploads are logically independent and could run concurrently on separate
streams, overlapping with each other and with the tail of the rayon-based constraint evaluation.

**Fix:** switch `shares_to_device`/`half_shares_to_device`/the public-input path to
`from_host_slice_async` with per-purpose streams. **Do this together with fix B** — without B, the
Montgomery conversion inside these calls would still force a serialization point regardless of the
transfer being async.

**Risk:** low-medium. Needs care with stream lifetime/ownership at the `setup()` call site.

## D. MSMs are gated behind `h` even though most don't depend on it

**Where:** [`src/groth16_gpu.rs:149-189`](src/groth16_gpu.rs#L149) (`prove_inner`) and
[`src/groth16_gpu.rs:209-406`](src/groth16_gpu.rs#L209) (`create_proof_with_assignment`)

**Problem:** `prove_inner` computes `h` (FFT-bound, via `witness_map_from_r1cs_eval`) to completion
*before* calling `create_proof_with_assignment`, which then launches all 8 MSMs. Only `h_acc`'s MSM
(line 308) actually needs `h` — the other 7 (`a_query_pub/priv`, `b_g1_query_pub/priv`,
`b_g2_query_pub/priv`, `l_query`) only need the public inputs / private witness, both available
right after `setup()`. Today the FFT-bound and MSM-bound phases run strictly sequentially.

**Fix:** restructure so the 7 independent MSMs are dispatched (async, on their own streams)
immediately after the witness/public-input upload, running concurrently with the witness-map FFTs;
only gate `h_acc`'s MSM behind `h`'s availability, then join everything before assembling the proof.

**Risk:** medium-high — genuine control-flow restructuring across `prove_inner` /
`create_proof_with_assignment`, not a local tweak. Recommend profiling A/B/C first to see how much
headroom is actually left before investing here.

## E. Per-proof stream creation/destruction

**Where:** [`src/groth16_gpu.rs:251`](src/groth16_gpu.rs#L251) (`ProofStreams::new()`), used in
`create_proof_with_assignment`

**Problem:** `ProofStreams { g1, g2 }` is created fresh on every single call and destroyed via
`Drop` at the end — i.e. every proof pays CUDA stream create/destroy cost. `reduction.rs` already
avoids this for its own streams via a `thread_local!` cache
([`src/groth16_gpu/reduction.rs:44-46`](src/groth16_gpu/reduction.rs#L44)).

**Fix:** cache `ProofStreams` the same way (`thread_local!`), matching the existing
`REDUCTION_STREAMS` pattern.

**Risk:** low. Same pattern already proven elsewhere in the codebase.

## F. Six independent G1 MSMs share one stream (lower priority)

**Where:** [`src/groth16_gpu.rs:257-308`](src/groth16_gpu.rs#L257)

All 6 G1 MSMs are issued on the same `stream_g1`, so CUDA runs them in strict issue order — only
the G1-vs-G2 split (2 streams total) gives any concurrency today. Splitting across a couple more
streams could let the small "pub" MSMs (bounded by `num_instance_variables`, usually tiny) overlap
with the large "priv"/`l_query`/`h_query` ones.

**Fix:** experiment with 2-3 streams for the G1 MSMs; **profile before committing** — benefit is
capped by how much spare GPU capacity remains once the large MSMs are running.

---

## Suggested order of work

1. **A** — remove redundant sync (isolated, no signature changes)
2. **B** — thread real streams through `ark_to_icicle_scalars` (mechanical, several call sites)
3. **C** — async transfers in `setup()` (depends on B to pay off fully)
4. **E** — cache `ProofStreams` (isolated, low risk)
5. Re-profile. Only then consider **D** (structural MSM/FFT overlap) and **F** (extra G1 streams),
   since their payoff depends on how much serialization A-C actually remove.
