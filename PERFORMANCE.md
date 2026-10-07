# Performance blockers in the Shamir GPU prover

This review covers the Shamir proving path only:

- `ShamirCoGroth16Prover::prove` ([src/groth16_gpu.rs](src/groth16_gpu.rs))
- the internal `CoGroth16Icicle` prover it wraps
- `ShamirGroth16Driver` ([src/mpc/shamir.rs](src/mpc/shamir.rs))
- the witness reductions in [src/groth16_gpu/reduction.rs](src/groth16_gpu/reduction.rs)

> **Status:** these findings come from reading the code. Nothing has been profiled or benchmarked yet, so the impact ranking is an estimate. An `nsys` profile of one Shamir proof, together with timings of the preprocessing step, would confirm or reorder it.
>
> Finding 1 is resolved and finding 2 is partially resolved; their sections describe what was changed and what remains. The original analysis of each is kept for reference. Run with `RUST_LOG=debug` to get the preprocessing time and the per-upload copy/`from_mont` split.

## Summary

| # | Issue | Where | Impact |
|---|-------|-------|--------|
| 1 | ~~Shamir preprocessing runs over the network on every proof, before any other work~~ **Resolved:** overlapped with host work | [groth16_gpu.rs:903-915](src/groth16_gpu.rs#L903-L915) | High (scales with network latency) |
| 2 | Witness upload and Montgomery conversion are serial and synchronous. **Partially resolved:** uploads now overlap CPU evaluation; still synchronous and unpinned | [groth16_gpu.rs:216-283](src/groth16_gpu.rs#L216-L283), [bridges.rs:69-90](src/bridges.rs#L69-L90) | High |
| 3 | `local_mul_vec` host sync stops the reduction streams from overlapping | [shamir.rs:183-198](src/mpc/shamir.rs#L183-L198), [reduction.rs:142](src/groth16_gpu/reduction.rs#L142) | Medium |
| 4 | `evaluate_constraint` reallocates on `resize` | [utils.rs:73-79](src/utils.rs#L73-L79) | Medium |
| 5 | `promote_to_trivial_shares` allocates and copies twice | [shamir.rs:80-88](src/mpc/shamir.rs#L80-L88), [reduction.rs:130-131](src/groth16_gpu/reduction.rs#L130-L131) | Medium |
| 6 | Smaller items: MSM stream balance, host syncs | various | Low |

**Suggested order:**
1. ~~Overlap or amortize the preprocessing (finding 1).~~ Done (overlap).
2. Profile the upload/conversion phase, then add the required Icicle pinned-memory support before attempting a fully asynchronous upload pipeline (finding 2). The CPU/transfer overlap part is done.
3. Fix findings 3 and 5 together, since both touch the start of the reduction.

## High impact

### 1. Shamir preprocessing runs over the network on every proof — resolved

[groth16_gpu.rs:903-915](src/groth16_gpu.rs#L903-L915), [groth16_gpu.rs:173-199](src/groth16_gpu.rs#L173-L199)

**What changed:** the preprocessing is now overlapped with the host work. `CoGroth16Icicle::prove_with_state_init` runs the state creation (`ShamirPreprocessing::new` + `ShamirState::from`) on a separate thread while the host evaluates and uploads the constraints, and joins it before the witness map. `Rep3CoGroth16Prover` uses the same path for `Rep3State::new`, which also has a network round. The plain prover has no state to set up and keeps the direct `prove` path. The protocol's network messages are unchanged, so the desynchronization risk of the amortize option does not apply. Amortizing (persistent `ShamirState`) remains possible as a follow-up if the preprocessing turns out to be longer than the host work it now hides behind.

**Original analysis:** `ShamirCoGroth16Prover::prove` builds a fresh `ShamirPreprocessing` for every proof, only to get the two random shares `r` and `s`. In `mpc-core`, `ShamirPreprocessing::new` does the following:

- Exchanges seeds with the other parties (`ShamirRng::new`). This is a network round.
- Runs `buffer_triples` to create random double shares. This involves a further `send_many`/`recv_many` exchange.
- In `ShamirState::from`, recomputes the Lagrange coefficients, which only depend on `num_parties`, `threshold` and the party ID.

All of this happens before constraint evaluation starts and before any GPU work is launched. So every proof pays the network round-trip time up front. The randomness is not needed until `prove_inner` calls `T::rand` after the witness map ([groth16_gpu.rs:308](src/groth16_gpu.rs#L308)).

**Fix (pick one or combine):**
- **Overlap it.** Run the preprocessing on a separate thread (`std::thread::scope` or `rayon::join`) while the host evaluates the constraints and uploads them. That host work doesn't use `net`. Join before `T::rand`. This needs `N: Sync`.
- **Amortize it.** Keep a `ShamirState` in `ShamirCoGroth16Prover` across proofs and refill its random-pair buffer in larger batches. This also removes the per-proof seed exchange and the Lagrange recomputation. `get_pair` already refills on its own when the buffer is empty.

Persistent state changes the sequence of network operations across proofs. Every party must enable it consistently and consume/refill correlated randomness in the same order, or the protocol can desynchronize.

### 2. Witness upload and Montgomery conversion are serial and synchronous — partially resolved

[groth16_gpu.rs:216-283](src/groth16_gpu.rs#L216-L283), [bridges.rs:69-90](src/bridges.rs#L69-L90)

**What changed:**
- `upload_inputs` runs each upload on its own scoped thread, so transfers overlap the host-side evaluation. The witness and public inputs upload while `A` is evaluated, `A` uploads while `B` is evaluated, and `B` uploads while `C` (LibSnark only) is evaluated. Each thread selects the caller's Icicle device first, since the active device is thread-local.
- `ark_scalars_to_device_into` logs the copy and `from_mont` times separately at debug level, for the profiling step below.

**Still open:** the pinned-memory, async-stream and witness-during-reduction items below. Each upload is still a synchronous copy from ordinary memory followed by a synchronous `from_mont`; it just no longer blocks the CPU evaluation. The threads are spawned per proof (measured at ~0.2 ms for four threads under WSL), which is small next to a proof but could be replaced by reusable workers if it shows up for small circuits.

**Original analysis:** the steps ran strictly one after another: CPU evaluates `eval_a`, blocking upload, CPU evaluates `eval_b`, blocking upload. The same happens for `eval_c` (LibSnark only), the witness and the public inputs. CPU work never overlapped a transfer.

The cost sits inside `ark_scalars_to_device_into`:

- **Synchronous copy from ordinary Rust memory.** It uses `copy_from_host`, so the host waits for every transfer to finish. The performance difference from pinned memory is hardware- and transfer-size-dependent and needs measurement.
- **Synchronous Montgomery conversion.** `from_mont` receives `IcicleStream::default()`. In this Icicle revision a null stream sets `is_async = false`, so the conversion completes before the call returns.

At present these operations do not stall already-running reduction or MSM streams: `prove_inner` starts the reduction only after every upload has returned, and the MSMs start after the reduction. The problem is the resulting lack of CPU/GPU overlap. Default-stream ordering would become an additional concern only after the surrounding work is made concurrent.

**Fix:**
- First profile copies and Montgomery conversion separately to establish how much of this phase is worth optimizing.
- Add or upgrade to Icicle support for pinned host allocation. The pinned Icicle revision currently exposes `HostSlice` over ordinary memory and its CUDA backend reports `supports_pinned_memory = false`; it does not provide the reusable pinned buffers required for reliable transfer/CPU overlap.
- Once pinned memory is available, keep reusable pinned staging buffers in `CoGroth16Icicle`. Upload with `copy_from_host_async` on a dedicated stream and run `from_mont` on that stream. Keep each staging buffer alive and unchanged until its stream has completed.
- Without pinned-memory support, asynchronous copies from ordinary memory may be synchronously staged by CUDA. They can be benchmarked, but overlap should not be assumed.
- ~~Start evaluating the next matrix on the CPU while the previous one uploads.~~ Done.
- Upload the public inputs before starting the reduction, which consumes them immediately. The private-witness upload is not consumed until the later MSM phase, so it can potentially overlap with the reduction if an explicit stream dependency makes it complete before those MSMs begin.

## Medium impact

### 3. `local_mul_vec` host sync stops the reduction streams from overlapping

[shamir.rs:183-198](src/mpc/shamir.rs#L183-L198), [reduction.rs:142](src/groth16_gpu/reduction.rs#L142)

`ShamirGroth16Driver::local_mul_vec` ends with `stream.synchronize()`. In `CircomReduction`, the `c = a·b` product is launched on stream `c` before any FFT work for `a` and `b`. The host therefore waits for that kernel to finish, plus one host round trip, before streams `a` and `b` get any work. The product is a single `mul_scalars`, so the stall is short, but it is still serialization on the critical path that the three streams were meant to avoid.

In `LibSnarkReduction` ([reduction.rs:250-252](src/groth16_gpu/reduction.rs#L250-L252)), stream `a` is synchronized twice in a row, once inside `local_mul_vec` and again right after it.

**Caution:** in `CircomReduction` the sync is currently needed for correctness. Stream `c` reads `eval_a`/`eval_b` while the in-place iFFT on streams `a`/`b` overwrites them. Removing the sync without adding another dependency introduces a data race.

**Fix:**
- Remove the sync from `local_mul_vec` and let callers synchronize. The Shamir version has no temporaries that would be freed early, so this is safe at the driver level.
- In `CircomReduction`, record an event on stream `c` after the product, and make streams `a`/`b` wait on that event before their iFFT. This orders the work on the GPU instead of blocking the host. The pinned Icicle Rust runtime does not currently expose event record/wait operations, so this requires upgrading or extending its runtime bindings. Until then, retain the host synchronization or restructure the buffers/work so the operations no longer race.

### 4. `evaluate_constraint` reallocates on `resize`

[utils.rs:73-79](src/utils.rs#L73-L79)

`collect()` sizes the Vec to exactly `num_constraints`. The following `resize(domain_size)` then reallocates and copies the whole vector. `evaluate_constraint_half_share`, used for `eval_c` in LibSnark, has the same problem. A fresh domain-sized Vec is also allocated on every proof.

**Fix:** allocate with `Vec::with_capacity(domain_size)` and fill it with `par_extend`. Better still, evaluate directly into the reusable pinned staging buffers from finding 2.

### 5. `promote_to_trivial_shares` allocates and copies twice

[shamir.rs:80-88](src/mpc/shamir.rs#L80-L88), [reduction.rs:130-131](src/groth16_gpu/reduction.rs#L130-L131), [reduction.rs:215-216](src/groth16_gpu/reduction.rs#L215-L216)

For Shamir, a public value is already a valid trivial share, so promotion is just a copy. The current code does all of this on every proof:

- a blocking `device_malloc`
- a copy on the default stream
- a second copy into `eval_a`
- a free when the temporary is dropped

Both `cudaMalloc` and `cudaFree` synchronize the whole device.

**Fix:** copy `public_inputs` directly into `eval_a[num_constraints..]` with an async copy on stream `a`. Ideally, upload the public inputs straight into that slice during the upload phase.

## Low impact

- **Unbalanced MSM streams** ([groth16_gpu.rs:373-432](src/groth16_gpu.rs#L373-L432)). Seven G1 MSMs are serialized on `stream_g1`, including two tiny public-input MSMs, while `stream_g2` has only two. Afterwards, `get_first` makes eight separate blocking device-to-host copies. `h_acc`/`l_acc` could run on a third stream.
- **Host syncs instead of events in `LibSnarkReduction`** ([reduction.rs:247-252](src/groth16_gpu/reduction.rs#L247-L252)). `stream_b.synchronize()` before the `a·b` product could be a cross-stream event wait.
