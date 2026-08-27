# GPU stream/pipelining optimizations

Findings from a review of the ICICLE GPU prover pipeline (`src/mpc.rs`, `src/mpc/shamir.rs`,
`src/mpc/rep3.rs`, `src/groth16_gpu.rs`, `src/groth16_gpu/reduction.rs`, `src/bridges.rs`,
`src/gpu_utils.rs`). These are performance-only observations — the underlying MPC protocol logic
is correct (verified separately against the CPU `co-groth16` reference). Ordered by expected
impact; A and B are believed to be actively undoing multi-stream parallelism the code already
sets up elsewhere.

**Status: B, C, D, E implemented. A was implemented, then reverted — it was wrong, see below.** F
(splitting the 6 G1 MSMs in `dispatch_independent_msms` across more streams) intentionally left
out — needs profiling data before committing to a stream layout, see the note in its section. All
changes compile clean (`cargo check --all-targets`, `cargo clippy --all-targets`,
`cargo test --no-run`) but could not be exercised at runtime in this environment (no GPU/CUDA
device available) — re-run the existing CPU-vs-GPU proof tests on GPU hardware before merging.

## A. ~~Drop the internal `stream.synchronize()` in `local_mul_vec`~~ — ❌ wrong, reverted

**This finding was incorrect and the change was reverted.** The sync is load-bearing, not
redundant. Kept below for the record, with the correction.

**Where:** [`src/mpc/shamir.rs:169-186`](src/mpc/shamir.rs#L169), identically in
[`src/mpc/rep3.rs:188-227`](src/mpc/rep3.rs#L188) and `src/mpc/plain.rs`.

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
so streams `a` and `b` never overlap with `c`. **This reasoning missed the actual reason the sync
exists.** `local_mul_vec` *reads* `eval_a`/`eval_b` on `stream_c` — but the very next thing the
caller does is `ifft_in_place(eval_a, stream_a, ...)` / `ifft_in_place(eval_b, stream_b, ...)`,
which mutate those *same buffers in place*, on *different streams*, with no synchronize issued in
between. CUDA gives no ordering guarantee across streams — only same-stream ops are FIFO-ordered —
so without a host-side block after the read, the in-place write can be issued to the GPU before
the read has actually finished executing: a genuine data race that silently corrupts `eval_a`/
`eval_b`. The `stream_b.synchronize()` / `stream_a.synchronize()` calls later in the same function
protect *different* pairs of ops (the second `local_mul_vec` call and the final `sub_scalars`) —
they do not reach back and protect this first read.

Confirmed against this repo's own git history: `14f5ac7 "fix: missing synchronization
checkpoint"` added exactly this sync to `rep3.rs`/`plain.rs`, at a point where `reduction.rs`
already had the same stream-a/b/c structure it has today — i.e. this was a real, previously-fixed
bug, not defensive leftover code. `shamir.rs` (added later) already included the sync from the
file's first commit, consistent with the team having already learned this the hard way in the
other two drivers.

**Fix:** ~~remove the `stream.synchronize()` call~~ — **do not remove it.** If the overlap in A is
still wanted, the correct fix is to replace the *host-blocking* sync with a GPU-side ordering
primitive — record a CUDA event after `local_mul_vec`'s kernels on `stream_c` and have `stream_a`/
`stream_b` wait on that event before their `ifft_in_place` calls (if `icicle_runtime` exposes
stream-wait-event bindings) — so the *host* doesn't block, but the GPU still enforces the
same-buffer ordering. Not attempted here; would need its own careful review.

**Risk:** ~~low~~ **high — this was a correctness bug, not a safe optimization.**

**Implemented as:** attempted the removal, then reverted after re-checking against the caller's
actual data-dependency graph and this repo's git history. The sync is back in all three drivers
(`shamir.rs`, `rep3.rs`, `plain.rs`), now with a comment explaining why it's required.

## B. `ark_to_icicle_scalars` hardcodes the CUDA null stream — ✅ implemented

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

**Implemented as:** `ark_to_icicle_scalars` now takes `stream: &IcicleStream` and passes it to
`from_mont` instead of `IcicleStream::default()`. Updated every call site (`shamir.rs`, `rep3.rs`,
`plain.rs`'s `shares_to_device`/`half_shares_to_device`, Rep3's `local_mul_vec` masking-element
conversion, and `setup()`'s public-input conversion) to pass the caller's real stream.

## C. Synchronous host→device transfers in `setup()` — ✅ implemented

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

**Implemented as:** added a `SetupStreams` thread-local (5 streams: `eval_a`, `eval_b`, `eval_c`,
`witness`, `public`, cached the same way as `ReductionStreams`/`ProofStreams`). `shares_to_device`
and `half_shares_to_device` gained a `stream: &IcicleStream` parameter (trait-level change, see B);
`evaluate_constraints` gained `stream_a`/`stream_b`/`stream_c` parameters. `setup()` dispatches all
five uploads on their own streams — interleaved with the CPU-bound rayon evaluation of the next
constraint matrix — then synchronizes all five once at the end, before handing the results off to
code that uses different streams (`reduction.rs`, the MSM stage).

## D. MSMs are gated behind `h` even though most don't depend on it — ✅ implemented

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

**Implemented as:** split `create_proof_with_assignment` into `dispatch_independent_msms` (the 7
MSMs that only need `public_inputs`/the private witness — dispatched async, unsynchronized) and
`finish_proof_with_assignment` (dispatches `h_acc`'s MSM once `h` is ready, synchronizes both
streams, then does the existing coefficient assembly + opening). `prove_inner` now calls
`dispatch_independent_msms` *before* `witness_map_from_r1cs_eval`, so the 7 MSMs run on `stream_g1`
/`stream_g2` concurrently with the witness-map FFTs on `stream_a`/`stream_b`/`stream_c`.

## E. Per-proof stream creation/destruction — ✅ implemented

**Where:** [`src/groth16_gpu.rs:251`](src/groth16_gpu.rs#L251) (`ProofStreams::new()`), used in
`create_proof_with_assignment`

**Problem:** `ProofStreams { g1, g2 }` is created fresh on every single call and destroyed via
`Drop` at the end — i.e. every proof pays CUDA stream create/destroy cost. `reduction.rs` already
avoids this for its own streams via a `thread_local!` cache
([`src/groth16_gpu/reduction.rs:44-46`](src/groth16_gpu/reduction.rs#L44)).

**Fix:** cache `ProofStreams` the same way (`thread_local!`), matching the existing
`REDUCTION_STREAMS` pattern.

**Risk:** low. Same pattern already proven elsewhere in the codebase.

**Implemented as:** added a `PROOF_STREAMS` thread-local (same `RefCell<ProofStreams>` pattern as
`SETUP_STREAMS`); `create_proof_with_assignment`'s (now `dispatch_independent_msms` /
`finish_proof_with_assignment`'s) `ProofStreams::new()` call was replaced with borrowing from the
cache.

## F. Six independent G1 MSMs share one stream (lower priority) — ⏭️ skipped (by request)

**Where:** [`src/groth16_gpu.rs:257-308`](src/groth16_gpu.rs#L257)

All 6 G1 MSMs are issued on the same `stream_g1`, so CUDA runs them in strict issue order — only
the G1-vs-G2 split (2 streams total) gives any concurrency today. Splitting across a couple more
streams could let the small "pub" MSMs (bounded by `num_instance_variables`, usually tiny) overlap
with the large "priv"/`l_query`/`h_query` ones.

**Fix:** experiment with 2-3 streams for the G1 MSMs; **profile before committing** — benefit is
capped by how much spare GPU capacity remains once the large MSMs are running.

---

## Suggested order of work

~~1. **A** — remove redundant sync (isolated, no signature changes)~~ — retracted, see above.

1. **B** — thread real streams through `ark_to_icicle_scalars` (mechanical, several call sites)
2. **C** — async transfers in `setup()` (depends on B to pay off fully)
3. **E** — cache `ProofStreams` (isolated, low risk)
4. Re-profile. Only then consider **D** (structural MSM/FFT overlap) and **F** (extra G1 streams),
   since their payoff depends on how much serialization B/C actually remove.

## Verification performed

- `cargo check --all-targets` — clean, no errors.
- `cargo clippy --all-targets` — no new warnings (3 pre-existing warnings, unrelated to this
  change, unchanged before/after — confirmed via `git stash`).
- `cargo test --no-run` — full test binary (including dev-dependencies and the CPU/GPU comparison
  tests in `src/lib.rs`) links successfully.
- **A was implemented, then found to be a correctness bug by re-reading this repo's own git
  history (see the A section) and reverted before landing anything else.** B, C, D, and E were
  re-checked afterwards, specifically for the same bug class (a buffer read on one stream that is
  later mutated in place, on a different stream, with no synchronize between the two) — none of
  them exhibit it: B/C only touch freshly-allocated, non-aliased buffers per stream and join with
  an explicit joint synchronize before handoff; D's two functions only ever touch two streams
  (`stream_g1`/`stream_g2`) and explicitly synchronize both before any host-side read (`get_first`).
  That said, compilation success proves nothing about this class of bug — it is entirely a runtime
  data race, invisible to the type checker.
- **Not run: the actual test suite / any real proof generation.** This environment has no GPU/CUDA
  device (`nvidia-smi` unavailable), so none of this could be exercised at runtime. Given that A
  was a real, shipped-then-caught mistake in this exact review, **do not treat the hand-verification
  above as sufficient — run the existing CPU-vs-GPU test suite on real GPU hardware before
  merging any of this**, ideally under `compute-sanitizer --tool racecheck` (or equivalent) to
  catch exactly this class of cross-stream hazard directly rather than relying on manual
  data-flow tracing.
