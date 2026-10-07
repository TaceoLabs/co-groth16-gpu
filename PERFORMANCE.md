# Performance blockers in the Shamir GPU prover

This review covers the Shamir proving path only:

- `ShamirCoGroth16Prover::prove` ([src/groth16_gpu.rs](src/groth16_gpu.rs)), both the default path (fresh state per proof) and the opt-in persistent-state path
- the internal `CoGroth16Icicle` prover it wraps
- `ShamirGroth16Driver` ([src/mpc/shamir.rs](src/mpc/shamir.rs))
- the witness reductions in [src/groth16_gpu/reduction.rs](src/groth16_gpu/reduction.rs)
- the Icicle CUDA backend behavior these depend on (`icicle-snark` rev `bf00385`, paths below are relative to `icicle/backend/cuda/src/` in that checkout)

> **Status:** these findings come from reading this crate and the Icicle CUDA backend source. Nothing has been profiled yet, so the impact ranking is an estimate. An `nsys` profile of one Shamir proof would confirm or reorder it; the log lines named below give a first check without one. Run with `RUST_LOG=debug` to also get the Shamir preprocessing time and the per-upload copy/`from_mont` split.
>
> This is the second pass. The first pass's findings that have since been fixed are summarized under [Resolved](#resolved); the open ones are carried over with updated line references.

## Summary

| # | Issue | Where | Impact |
|---|-------|-------|--------|
| 1 | "Async" MSM launches block the host, so the reduction is not issued until most MSMs have finished | [groth16_gpu.rs:376-491](src/groth16_gpu.rs#L376-L491), `msm/cuda_msm.cuh:590-597` | High |
| 2 | Synchronous default-stream copies at the start of the reduction wait for every in-flight MSM | [shamir.rs:162-171](src/mpc/shamir.rs#L162-L171), [reduction.rs:166](src/groth16_gpu/reduction.rs#L166), `cuda_device_api.cu:75,101` | High |
| 3 | Uploads are synchronous copies from unpinned memory (partially resolved) | [groth16_gpu.rs:290-368](src/groth16_gpu.rs#L290-L368), [bridges.rs:84-115](src/bridges.rs#L84-L115) | Medium |
| 4 | `local_mul_vec` host sync stops the reduction streams from overlapping | [shamir.rs:279-294](src/mpc/shamir.rs#L279-L294), [reduction.rs:177](src/groth16_gpu/reduction.rs#L177) | Medium |
| 5 | Persistent Shamir state is topped up two pairs at a time, serially, before the upload | [groth16_gpu.rs:1025-1037](src/groth16_gpu.rs#L1025-L1037), [groth16_gpu.rs:1113](src/groth16_gpu.rs#L1113) | Low–medium (only with more than 3 parties) |
| 6 | Smaller items: host buffer reuse, public-segment copy, Icicle per-MSM overhead, LibSnark syncs | various | Low |

Findings 1 and 2 have to be fixed together. Each one alone is enough to serialize the witness-independent MSMs and the reduction, which is the overlap `prove_inner` is designed around.

**Suggested order:**
1. Confirm finding 1 from the existing logs (see below), then fix findings 1 and 2 together.
2. Fix finding 4 once the reduction actually runs alongside the MSMs; its stall matters more then.
3. Pinned-memory uploads (finding 3) need an Icicle upgrade or patch first.

## High impact

### 1. "Async" MSM launches block the host

[groth16_gpu.rs:376-414](src/groth16_gpu.rs#L376-L414) (`prove_inner`), [groth16_gpu.rs:436-491](src/groth16_gpu.rs#L436-L491) (`launch_witness_independent_msms`), [groth16_gpu.rs:419-430](src/groth16_gpu.rs#L419-L430) (`launch_h_msm`)

`prove_inner` launches the four witness-independent MSMs (`a`, `b_g1`, `l` on `streams.g1`; `b_g2` on `streams.g2`) and then issues the witness-map reduction, expecting the host to return from the launches immediately. It doesn't.

In the Icicle CUDA MSM, partway through each call (after the bucket sort), the host reads two bucket counts back with `cudaMemcpyAsync(&h_nof_buckets_to_compute, ..., cudaMemcpyDeviceToHost, stream)` into stack variables (`msm/cuda_msm.cuh:590-597`, and `:660` for the large-bucket count), then uses them to size the next kernels. A device-to-host copy into pageable memory is synchronous for the host. So each `msm_into` call returns only after:

- all earlier work on its stream has finished, and
- its own sort phase has finished.

With three G1 MSMs queued on one stream, `launch_witness_independent_msms` returns only after `a` and `b_g1` have completed and `l` is through its sort. Only then is the reduction issued. In practice the reduction runs after the G1 MSMs instead of alongside them. `launch_h_msm` has the same problem: it blocks until `a`, `b_g1` and `l` have drained from `streams.g1`.

**How to confirm without a profiler:** the info log `Launching witness-independent MSMs took N ms` should be close to zero if the launches were async. A value close to the time of two to three G1 MSMs confirms this finding.

**Fix:**
- Issue each MSM stream's launches from its own host thread (the `on_device` helper from the upload phase already handles the thread-local device), while the main thread issues the reduction. Join the MSM threads before `finish_proof_with_assignment` reads the results.
- Give the `h` MSM its own stream (or put it on `streams.g2`, which carries only `b_g2`), so its launch does not wait for the other G1 MSMs to drain.
- The GPU's compute is shared, so concurrent MSMs and NTTs won't add up linearly. The gain is in removing the host-side gaps and letting the reduction's smaller kernels fill the GPU between MSM phases.
- The upstream fix is for Icicle to read those counters through pinned host memory. That would make the launches truly async, but needs an Icicle patch.

### 2. Synchronous default-stream copies wait for every in-flight MSM

[shamir.rs:162-171](src/mpc/shamir.rs#L162-L171), [reduction.rs:166](src/groth16_gpu/reduction.rs#L166) (Circom), [reduction.rs:249](src/groth16_gpu/reduction.rs#L249) (LibSnark)

The Icicle CUDA backend creates every stream with plain `cudaStreamCreate` (`cuda_device_api.cu:101`), so all of them are *blocking* streams. Its synchronous `copy` and `memset` use `cudaMemcpy`/`cudaMemset` on the legacy default stream (`cuda_device_api.cu:59,75`). The backend is not built with per-thread default streams. Under these semantics, a legacy-default-stream operation waits for all earlier work on every blocking stream, and later work on those streams waits for it.

The first thing each reduction does is `T::write_trivial_shares_into`. For Shamir this is a synchronous device-to-device `copy` of the public inputs into `eval_a[num_constraints..]`. It runs right after the MSM launches, so it waits for all four witness-independent MSMs to finish before the reduction can start. This holds even after finding 1 is fixed. The REP3 and plain drivers use the same synchronous `copy`/`memset` calls.

**Fix:**
- Make `write_trivial_shares_into` asynchronous: `copy_async`/`memset_async` on the reduction's stream `a` (pass the stream in), so it is ordered before the iFFT on `a` and doesn't touch the default stream.
- Better for Shamir: write the public inputs into `eval_a[num_constraints..]` from the host during the upload phase. Every Shamir party writes the same values, so no device-to-device copy is needed at all. (REP3 needs the id-dependent split.)
- As a rule, nothing between the first MSM launch and the final `streams.g1/g2.synchronize()` may use a synchronous Icicle copy, memset, `device_malloc` or drop of a `DeviceVec`, or a null-stream Icicle call. Each of these is an implicit device-wide barrier here.
- Making Icicle create non-blocking streams (`cudaStreamCreateWithFlags(..., cudaStreamNonBlocking)`) would remove the implicit barriers entirely, but needs an Icicle patch.

## Medium impact

### 3. Uploads are synchronous copies from unpinned memory — partially resolved

[groth16_gpu.rs:290-368](src/groth16_gpu.rs#L290-L368), [bridges.rs:84-115](src/bridges.rs#L84-L115)

**Already done:** `upload_inputs` runs each upload on its own scoped thread, so transfers overlap the host-side constraint evaluation: the witness (straight into `combined_scalars[pub_len..]`) and public inputs upload while `A` is evaluated, `A` uploads while `B` is evaluated, and `B` uploads while `C` (LibSnark only) is evaluated. Only `num_constraints` entries are uploaded per matrix; the domain padding after them is zeroed with a device memset in the same upload thread, on every proof. `ark_scalars_to_device_into_at` logs the copy and `from_mont` times separately at debug level.

**Still open:**
- Each upload is still a synchronous `cudaMemcpy` from pageable memory, followed by a `from_mont` on the null stream, which Icicle runs synchronously. That's fine for the overlap above, but it caps transfer bandwidth. The pinned Icicle revision reports `supports_pinned_memory = false` (`cuda_device_api.cu:116`) and has no pinned host allocation API.
- Once pinned memory is available: keep reusable pinned staging buffers in `CoGroth16Icicle`, evaluate the constraints straight into them, and upload with `copy_from_host_async` plus `from_mont` on a dedicated stream. Keep each staging buffer unchanged until its stream has completed.
- The threads are spawned per proof (measured at ~0.2 ms for four threads under WSL). This is small next to a proof, but could be replaced by reusable workers if it shows up for small circuits.

### 4. `local_mul_vec` host sync stops the reduction streams from overlapping

[shamir.rs:279-294](src/mpc/shamir.rs#L279-L294), [reduction.rs:177](src/groth16_gpu/reduction.rs#L177), [reduction.rs:194-199](src/groth16_gpu/reduction.rs#L194-L199)

`ShamirGroth16Driver::local_mul_vec` ends with `stream.synchronize()`. In `CircomReduction`, the `c = a·b` product is launched on stream `c` before any FFT work for `a` and `b`, so the host waits for that kernel plus a host round trip before streams `a` and `b` get any work. The second call (`h = a·b` on stream `a`) is preceded by `stream_b.synchronize()` and then syncs stream `a`, so the host waits there too before issuing the final subtraction on stream `c`.

In `LibSnarkReduction` ([reduction.rs:280-288](src/groth16_gpu/reduction.rs#L280-L288)) the same pattern applies: a host sync on stream `b`, then the product with its internal sync on stream `a`.

**Caution:** in `CircomReduction` the first sync is needed for correctness. Stream `c` reads `eval_a`/`eval_b` while the in-place iFFT on streams `a`/`b` overwrites them. Removing the sync without adding another dependency introduces a data race.

**Fix:**
- Remove the sync from `local_mul_vec` and let callers synchronize. The Shamir version has no temporaries that would be freed early, so this is safe at the driver level.
- Order the streams on the GPU instead of on the host: record an event on stream `c` after the product and make streams `a`/`b` wait on it before their iFFT; likewise for the `b → a` and `a → c` handoffs. The pinned Icicle Rust runtime does not expose event record/wait, so this requires extending its bindings.
- Without events: issue the first product on stream `a` itself (before its iFFT) and have stream `b`'s iFFT wait via one host sync of stream `a`, which removes one of the round trips. Any restructuring must keep the `eval_a`/`eval_b` read-before-overwrite ordering.

## Low–medium impact

### 5. Persistent Shamir state is topped up two pairs at a time, serially

[groth16_gpu.rs:1025-1037](src/groth16_gpu.rs#L1025-L1037) (`preprocess`), [groth16_gpu.rs:1113](src/groth16_gpu.rs#L1113)

On the persistent path, every `prove` calls `self.preprocess(net, 1)` before the upload. `ShamirState::buffer_triples` is a no-op when enough pairs are buffered. But unless the caller pre-filled the buffer with a large `preprocess(net, n)`, the buffer holds exactly the two pairs the previous proof used up, so every proof generates two fresh pairs. For 3 parties this needs no communication. For more parties it is a `random_double_share` network round on every proof, on the critical path, before any host or GPU work. This is the cost the overlap on the default path was added to hide.

**Fix:**
- When the buffer is short, top up geometrically, as `ShamirState::get_pair` does internally, so the round trip is amortized over many proofs.
- Or overlap the top-up with the upload, as `prove_with_state_init` does for the default path. The state lives in `ShamirCoGroth16Prover::state`, separate from `inner`, so a scoped thread can borrow it mutably while `inner` uploads.
- All parties must top up in the same pattern, or the network desyncs (see the struct docs).

## Low impact

- **Per-proof host allocations for the evaluations** ([utils.rs:67-98](src/utils.rs#L67-L98)). `evaluate_constraint` allocates a fresh `num_constraints`-sized `Vec` per matrix per proof, so every proof pays for page faults on freshly mapped memory. Reusable host buffers in `CoGroth16Icicle` (cleared and `par_extend`ed) avoid this, and become the pinned staging buffers of finding 3 later.
- **Public segment of `combined_scalars` via a device-to-device copy** ([groth16_gpu.rs:453-461](src/groth16_gpu.rs#L453-L461)). `write_combined_public_segment` does a synchronous default-stream copy before the MSMs on every proof. It is small and happens before the MSM launches, so it is not a barrier against them. For Shamir the segment could instead be written from the host during the upload phase.
- **Icicle per-MSM overhead.** When an MSM takes the large-bucket path, Icicle creates a CUDA stream and event and, in async mode, a detached cleanup `std::thread` per call (`msm/cuda_msm.cuh:686`, `:1123`). This is up to five per proof and can only be changed in Icicle.
- **Host syncs in `LibSnarkReduction`** ([reduction.rs:280](src/groth16_gpu/reduction.rs#L280)). `stream_b.synchronize()` before the `a·b` product could be a cross-stream event wait (needs the event bindings from finding 4).

## Resolved

Findings from the first pass that are fixed on this branch:

- **Shamir preprocessing ran over the network before any other work.** On the default path, `prove_with_state_init` ([groth16_gpu.rs:242-268](src/groth16_gpu.rs#L242-L268)) now creates the state (`ShamirPreprocessing::new` + `ShamirState::from`) on a separate thread while the host evaluates and uploads the constraints, and joins before the reduction. REP3 uses the same path for `Rep3State::new`. The opt-in persistent state ([groth16_gpu.rs:1082-1129](src/groth16_gpu.rs#L1082-L1129)) adds the amortized alternative, which removes the per-proof seed exchange and Lagrange recomputation (see finding 5 for its remaining cost).
- **Upload and CPU evaluation were fully serial.** Now overlapped; see finding 3 for what remains.
- **`evaluate_constraint` reallocated on `resize`.** The host now produces exactly `num_constraints` entries with no padding, and `upload_inputs` zeroes the domain padding with a device memset on every proof ([groth16_gpu.rs:325-366](src/groth16_gpu.rs#L325-L366)). (An earlier version zeroed it only once, in `CoGroth16Icicle::new`. That produced invalid proofs from the second `prove` on a reused prover onwards, because the reduction's in-place (i)FFTs leave the padding dirty. The test macro now verifies every proof so this can't pass silently again.)
- **`promote_to_trivial_shares` allocated, copied twice and freed per proof.** Replaced by `write_trivial_shares_into`, which copies straight into `eval_a[num_constraints..]` with no temporary buffer. Its remaining cost is the default-stream barrier in finding 2.
- **Eight MSMs and eight blocking result read-backs.** The public and private MSMs are merged over `combined_scalars` (five MSMs per proof), results land in long-lived slots, and one device-to-host copy per curve reads them back ([groth16_gpu.rs:515-545](src/groth16_gpu.rs#L515-L545)). The witness-independent MSMs are launched before the reduction (but see findings 1 and 2).
