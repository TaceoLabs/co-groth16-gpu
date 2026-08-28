# Performance analysis: Shamir GPU Groth16 prover

Scope: the collaborative Groth16 prover instantiated with `ShamirGroth16Driver`
([src/mpc/shamir.rs](../src/mpc/shamir.rs)), driven through
[src/groth16_gpu.rs](../src/groth16_gpu.rs) /
[src/groth16_gpu/reduction.rs](../src/groth16_gpu/reduction.rs), plus the
pinned `co-snarks` Shamir MPC core (`mpc-core/src/protocols/shamir/{arithmetic,network,rngs}.rs`,
rev `b95c341`) and the `icicle-runtime` stream API.

Line numbers in this repo are as of commit `e2a71d0` ("fix: msms slices need to
be of equal length") and have been verified against it. See
[Working-tree divergence](#working-tree-divergence) for uncommitted changes that
invalidate some of the evidence below.

Note on `co-snarks` paths: `co-groth16` and `co-plonk` live under
`co-circom/` in that repo (`co-circom/co-groth16/src/groth16.rs`,
`co-circom/co-plonk/src/...`). Paths in this document are given in full.

Note on `icicle`: it is **not** vendored in-tree. It is a cargo git dependency
resolved to `~/.cargo/git/checkouts/icicle-snark-4985b69852b5d20e/bf00385/`,
rev `bf00385d`. It is also **not rev-pinned** in
[Cargo.toml:25-28](../Cargo.toml#L25-L28) — only the lockfile pins it, unlike
the `co-snarks` deps on lines 22-23. Since [P4](#p4--local_mul_vecs-internal-stream-sync-two-of-three-call-sites-are-redundant-one-is-a-real-hazard)
rests entirely on the current icicle API surface, a `cargo update` can
invalidate it silently; pinning it is worth doing regardless.

## Summary

The protocol design is already close to optimal on the metric that usually
dominates collaborative-SNARK cost: network rounds and bytes. The online
phase does exactly **two** broadcast rounds, each moving a handful of curve
points, regardless of circuit size — see [Baseline](#baseline-already-optimal-no-changes-recommended)
below. The remaining opportunities are:

1. Per-proof rebuild of the *proving key* and of the *global NTT domain*
   (P0 — by far the largest fixed per-proof cost, and a correctness hazard
   under concurrency).
2. Per-proof MPC session setup overhead (matters for throughput/service
   workloads, not single-proof latency).
3. GPU/CPU pipelining inside a single proof — several places do sequential,
   blocking work. The codebase's own async-stream pattern is used in key
   setup but not on the per-proof hot path; note that extending it is
   *blocked* on a default-stream problem in the ark→icicle bridge (see P1).

None of this is "broken" — the prover works and the round complexity is
already good. These are throughput/latency optimizations on top of a sound
design, plus one concurrency hazard.

## Baseline: already optimal, no changes recommended

Stated explicitly so it doesn't get "fixed" by accident:

- **Two communication rounds total** in the online phase
  ([groth16_gpu.rs:389](../src/groth16_gpu.rs#L389),
  [:403-404](../src/groth16_gpu.rs#L403-L404)), each opening only 2-3 curve
  points via a single `broadcast_next` round trip.
- **The half-share trick** — `to_half_share` / `to_half_share_vec`
  ([mpc.rs:76-79](../src/mpc.rs#L76-L79)), `local_mul_vec`
  ([mpc.rs:180-185](../src/mpc.rs#L180-L185)), `open_device_half_shares`
  ([mpc.rs:228-232](../src/mpc.rs#L228-L232)) — avoids ever running DN07
  degree-reduction (`degree_reduce_many`,
  `mpc-core/src/protocols/shamir/network.rs:150` in the pinned `co-snarks`
  dependency) on domain-sized vectors. `AB` and the `H` polynomial are
  carried as unreduced degree-`2t` shares and only ever combined *linearly*
  into MSMs; only the resulting single point is opened. This is the single
  biggest possible communication-volume win for this workload and it's
  already implemented.
- `broadcast_next` messages only the minimal `t+1` / `2t+1`-party subset, not
  all `n` parties, and `set_nodelay(true)` is set on every `mpc-net` TCP
  path (verified: all 8 call sites across `tcp.rs`, `tcp_session.rs`,
  `tcp_session_blocking.rs`, `tls.rs` cover both connect and accept
  branches), so small point broadcasts aren't taxed by Nagle-induced latency.
- Key prep (`ProvingKey::from_ark`,
  [gpu_utils.rs:219-350](../src/gpu_utils.rs#L219-L350)) pipelines
  8 independent uploads across 8 streams and parallelizes the ark→icicle
  conversion with `rayon_join_5!`. **Caveat:** this is only *one-time* if the
  caller passes a cached key — see [P0](#p0--proving-key-and-ntt-domain-are-rebuilt-per-proof-unless-the-caller-caches).

## Findings and recommended changes

### P0 — Proving key and NTT domain are rebuilt per proof unless the caller caches

**Where:** [groth16_gpu.rs:774-780](../src/groth16_gpu.rs#L774-L780) (Shamir),
[:654-660](../src/groth16_gpu.rs#L654-L660) (Rep3),
[:511-517](../src/groth16_gpu.rs#L511-L517) (plain bn254),
[:576-580](../src/groth16_gpu.rs#L576-L580) (plain bls12_377);
`ProvingKey::from_ark` [gpu_utils.rs:219-350](../src/gpu_utils.rs#L219-L350);
`initialize_domain` [gpu_utils.rs:353-367](../src/gpu_utils.rs#L353-L367)

**Problem, part 1 — key rebuild.** Every `prove` entry point takes
`prepared_bn254_key: Option<Arc<Bn254PreparedKey>>` and does:

```rust
let prepared_key = prepared_bn254_key.unwrap_or_else(|| {
    Arc::new(prepare_bn254_key::<R>(key, ..., ...))
});
```

When the caller passes `None`, the *entire* proving key is re-converted
(5 rayon ark→icicle conversions over the full query vectors) and re-uploaded
to the device with 8× precompute — **on every proof**. The bls12_377 path in
`plain_prove` ([:576-580](../src/groth16_gpu.rs#L576-L580)) has no caching
option at all and always rebuilds. For the multi-proof service workload that
[P1](#p1--shamircogroth16prove-re-runs-full-session-setup-on-every-call)
targets, this dwarfs the Shamir seed handshake by orders of magnitude.

**Problem, part 2 — global NTT domain churn.** `from_ark` calls
`initialize_domain::<F>(domain_size)` ([gpu_utils.rs:310](../src/gpu_utils.rs#L310)),
which does `ntt::release_domain::<F>()` followed by `ntt::initialize_domain(...)`
([gpu_utils.rs:353-367](../src/gpu_utils.rs#L353-L367)). Two consequences:

1. Combined with part 1, every uncached proof pays a full twiddle-factor
   domain teardown and rebuild.
2. It is **global mutable device state**, mutated from a function that reads
   as pure key preparation. Two provers running concurrently on the same
   field clobber each other's NTT domain. Given `REDUCTION_STREAMS` is a
   `thread_local!` ([reduction.rs:44-46](../src/groth16_gpu/reduction.rs#L44-L46)),
   multi-threaded use is clearly contemplated. CI already pins
   `--test-threads=1` (`.github/workflows/rust_test.yml:128`), attributing it
   to GPU contention — but this would force serialization regardless.

**Impact:** highest of anything in this document for any workload issuing
more than one proof. Part 2 is a correctness hazard, not just a perf issue.

**Recommended change:**
- Make the cached-key path the documented default: have callers hold the
  `Arc<Bn254PreparedKey>` for the lifetime of the service, and give the
  bls12_377 path the same `Option<Arc<...>>` parameter the bn254 paths have.
  Consider making the parameter non-optional so the cost can't be incurred
  by accident.
- Move `initialize_domain` out of `from_ark` into an explicit, idempotent
  one-time init keyed on `(F, domain_size)`, so it is not re-run per key
  build and cannot be triggered concurrently. At minimum, document that
  `from_ark` mutates global device state and must not run concurrently with
  any prover on the same field.

**Risk/effort:** low risk for the key-caching half (API addition plus
documentation); medium for the NTT-domain half, since it changes where global
init happens. Both are small in code size.

---

### P1 — `ShamirCoGroth16::prove` re-runs full session setup on every call

**Where:** [groth16_gpu.rs:757-760](../src/groth16_gpu.rs#L757-L760)

```rust
// we need 2 corr rand pairs for the two rand calls
let num_pairs = 2;
let preprocessing = ShamirPreprocessing::new(num_parties, threshold, num_pairs, net)?;
let mut state = ShamirState::from(preprocessing);
```

**Problem:** `ShamirPreprocessing::new` (`mpc-core/src/protocols/shamir.rs:35-63`)
does two things from scratch on *every* call: `ShamirRng::new`'s pairwise-seed
handshake (`get_shared_rngs`, `rngs.rs:98-139` — one round of
`send_to`/`recv_from` per peer, O(n) messages per party) and one DN07
`buffer_triples` round (`random_double_share`, `rngs.rs:334-395`). The
pairwise seeds never need to change between proofs — they're a property of
the party set, not of any individual proof. For a single ad-hoc proof this is
invisible; for a proving service issuing many proofs over a live session,
this fixed per-call overhead is pure waste.

**Important caveat for `n = 3`:** the DN07 triple round is **entirely
network-free** in the 3-party case. `send_share_of_randomness` /
`receive_share_of_randomness` both early-return with zero messages when
`num_parties - seeded - 1 == 0` (`rngs.rs:297-300`, `:319-322`), and for
n=3/t=1 that expression is 0; there is an in-tree comment at
`shamir.rs:130-133` saying as much. So for 3 parties — which is exactly what
`Rep3CoGroth16::prove_with_shamir_bridge`
([groth16_gpu.rs:704-705](../src/groth16_gpu.rs#L704-L705)) uses — the only
per-proof network cost here is the seed handshake, not a full DN07 round.
The finding still holds; its magnitude is smaller than it first appears for
the configuration this repo has a dedicated entry point for.

**Impact:** medium for throughput/service workloads with n > 3, low for n = 3,
zero for a one-off single proof. Scales with proof *count*, not circuit size.
Strictly smaller than [P0](#p0--proving-key-and-ntt-domain-are-rebuilt-per-proof-unless-the-caller-caches).

**Recommended change:** add a `ShamirCoGroth16::prove_with_state` (or
similar) that takes an existing, long-lived `&mut ShamirState<P::ScalarField>`
built once per session (handshake done once), and per proof calls
`parent_state.fork(2)` (public API via the `MpcState` trait,
`mpc-core/src/protocols/shamir.rs:173-186`) to hand out a cheap,
communication-free child state pre-loaded with exactly the 2 pairs that
proof's `rand()` calls need. `fork` takes no `&N` parameter and neither does
its helper `ShamirRng::fork_with_pairs` (`rngs.rs:74-96`), so it is
structurally incapable of communicating; it re-derives the child's RNGs
deterministically from the parent's. This is the established idiom — see
[Tricks from `co-snarks`](#tricks-found-in-the-co-snarks-dependency) below.
Keep the current `prove()` as a convenience wrapper for the single-shot case.
`prove_inner` already takes `state: &mut T::State`, so the plumbing exists.

**Sizing the buffer is a correctness requirement, not an optimization.**
If the parent has fewer than `amount` pairs buffered, `fork_with_pairs`
**panics** — `panic!("not enough corr rand pairs")` at `rngs.rs:80-83`, with
an in-tree `// TODO return err? pass in net and generate more?`. It does
*not* return `Err` (the Shamir `fork` impl always returns `Ok`) and does
*not* silently refill over the network. So a session-reuse API must call
`state.buffer_triples(net, 2 * expected_proofs)` up front, sized to the
expected workload, or a busy service will abort. Note also the asymmetry:
`get_pair` pops from the *back* (`shamir.rs:140-141`) while
`fork_with_pairs` drains from the *front*.

**Risk/effort:** low risk (additive API), small effort — but the panic above
means the buffer sizing must be got right, not left to the lazy-refill path.

---

### P1 — `setup()` serializes four blocking host→device transfers, and the async fix is blocked on the ark→icicle bridge

**Where:** [groth16_gpu.rs:101-144](../src/groth16_gpu.rs#L101-L144)
(`CoGroth16Icicle::setup`), default `evaluate_constraints`
([mpc.rs:122-173](../src/mpc.rs#L122-L173)),
`ark_to_icicle_scalars` ([bridges.rs:54-66](../src/bridges.rs#L54-L66))

**Problem:** `setup()` performs **four** blocking uploads in strict sequence
(five when `R::requires_eval_c()`), not three — and the two largest are
*inside* the step that looks like pure CPU work:

1. `evaluate_constraints` — despite the name, this does **not** return host
   `Vec`s. It returns `T::DeviceShares`. The rayon matrix evaluation
   ([mpc.rs:141](../src/mpc.rs#L141), [:150](../src/mpc.rs#L150),
   [:160](../src/mpc.rs#L160)) is followed immediately by
   `shares_to_device` at [mpc.rs:148](../src/mpc.rs#L148) and
   [:157](../src/mpc.rs#L157) — two **domain-sized** blocking uploads, the
   largest transfers in `setup()`.
2. `T::shares_to_device` for the private witness
   ([groth16_gpu.rs:128](../src/groth16_gpu.rs#L128)) → `from_host_slice`
   ([gpu_utils.rs:75-82](../src/gpu_utils.rs#L75-L82)), which allocates and
   calls `copy_from_host` with **no stream** (default/synchronous stream).
3. `ark_to_icicle_scalars(from_host_slice(public_inputs))`
   ([groth16_gpu.rs:132](../src/groth16_gpu.rs#L132)) → another blocking
   upload.

**Why the obvious fix does not work.** Every one of those paths funnels
through `ark_to_icicle_scalars`, which runs the Montgomery conversion on
`IcicleStream::default()` ([bridges.rs:63](../src/bridges.rs#L63)) — and
`Default for IcicleStream` is a **null handle** (`stream.rs:48-54` in the
icicle checkout). Consequences:

- The `from_mont` kernels serialize against *each other* on the default
  stream no matter which stream the H2D copy used. Switching the copies to
  `from_host_slice_async` on separate streams therefore buys nothing on its
  own.
- Worse: icicle's CUDA backend creates streams with plain `cudaStreamCreate`
  (`icicle/backend/cuda/src/cuda_device_api.cu:98-104`), i.e. *blocking*
  streams. Under legacy default-stream semantics a null-stream kernel is an
  implicit device-wide barrier against every other blocking stream —
  including the 8 key-upload streams and the 3 reduction streams. (This
  assumes icicle is not built with `--default-stream per-thread`; worth
  confirming before relying on it either way. The point about serializing
  on one stream holds regardless.)
- `icicle_stream_synchronize(nullptr)` maps to `cudaDeviceSynchronize`
  (`cuda_device_api.cu:90-95`), so the default stream really is device-wide.

**Also:** the rayon matrix evaluation is *already* parallel —
`evaluate_constraint` is `matrix.par_iter().with_min_len(256)`
([utils.rs:73-77](../src/utils.rs#L73-L77)). Both A and B already saturate
the thread pool, so wrapping them in a 2-way `rayon::join` adds nesting, not
parallelism; the only win is the ragged tail plus overlapping the interleaved
GPU uploads — which lands back on the default-stream problem above. (There
are also *three* independent evaluations when `requires_eval_c()`, not two.)

**Impact:** medium-high; scales with `domain_size` and witness length. Visible
in the existing `tracing::info!("Setup timings: ...")` log at
[groth16_gpu.rs:135-141](../src/groth16_gpu.rs#L135-L141) — but note that log
attributes the two domain-sized uploads to `evaluate_constraints`, so the
`witness_to_device` / `public_to_device` numbers understate the real transfer
cost. Splitting that timer is a prerequisite for sizing this finding.

**Recommended change (in order — step 1 is mandatory):**
1. Thread an `&IcicleStream` through `ark_to_icicle_scalars` (and the
   `MontgomeryConvertible::from_mont` call it wraps). This requires adding a
   stream parameter to the `shares_to_device` / `half_shares_to_device`
   **trait methods** ([mpc.rs:106-119](../src/mpc.rs#L106-L119)), which
   currently take none, plus the three driver impls. Nothing else in this
   finding can work until this is done.
2. Only then: issue the uploads on separate streams via
   `from_host_slice_async` ([gpu_utils.rs:84-92](../src/gpu_utils.rs#L84-L92))
   and synchronize once, right before `prove_inner` needs them.
3. Split the `Setup timings` log so `evaluate_constraints`' own uploads are
   measured separately from its CPU work.

**Risk/effort:** low risk (no protocol/semantic change, pure scheduling) but
**medium-large effort** — this is a trait-signature change across the driver
abstraction, not the localized stream plumbing that `ProvingKey::from_ark`
needed. Do [P0](#p0--proving-key-and-ntt-domain-are-rebuilt-per-proof-unless-the-caller-caches)
first; it is much cheaper per unit of win.

---

### P2 — Six independent MSMs are queued on a single CUDA stream

**Where:** [groth16_gpu.rs:255-322](../src/groth16_gpu.rs#L255-L322)
(`create_proof_with_assignment`)

**Problem:** `pub_acc_r_g1`, `priv_acc_r_g1`, `pub_acc_s_g1`,
`priv_acc_s_g1`, `l_acc`, and `h_acc` are all launched on the same
`stream_g1` ([groth16_gpu.rs:252](../src/groth16_gpu.rs#L252)). A CUDA
stream executes queued kernels FIFO regardless of `cfg.is_async`, so these
six MSMs run **serially** on-device even though they operate on entirely
independent point/scalar sets. The two "pub" MSMs are tiny (size =
`num_instance_variables`, often a handful to low hundreds of points) and
sit queued behind `l_acc`/`h_acc`/the "priv" MSMs, which are aux- or
domain-sized — i.e. the small, cheap work waits behind the large, expensive
work instead of overlapping with it.

**Impact:** medium; the `tracing::info!("MSM + stream sync took ...")` log
at [groth16_gpu.rs:326-329](../src/groth16_gpu.rs#L326-L329) already
measures this window — profile before changing to see how much of it is
"stream_g1 idle while a tiny MSM waits its turn" versus genuine GPU
saturation (if the GPU is already compute-bound on the large MSMs, spreading
across streams won't help much; if there's headroom, it should shorten the
critical path).

**Recommended change:** give `ProofStreams` one stream per independent MSM
and distribute the eight `msm_async` calls across them, keeping the existing
synchronize barrier (now over more streams) before `get_first`.

The upstream CPU reference driver
(`co-circom/co-groth16/src/groth16.rs`, `create_proof_with_assignment` spans
lines 201-326) is direct precedent for the concurrency, though note its exact
shape before copying it: `rayon_join5!` at line 221 runs 5 outer groups
(A/G1, B/G1, B/G2, `l_acc`, `h_acc`), and the nested `rayon::join` in
`calculate_coeff` (line 188) splitting public/private MSM is reached by only
**three** of those five — the `l_query` and `h_query` closures call
`msm_public_points_hs` directly with no split. So upstream's effective
concurrency is 3×2 + 2 = 8 units, which happens to match "one stream per MSM"
here. Pick that target rather than the 5-group framing.

**Two things to preserve while doing this:**
- `ProofStreams::new()` is called **per proof**
  ([groth16_gpu.rs:251](../src/groth16_gpu.rs#L251)), creating and destroying
  its streams every time ([gpu_utils.rs:197-211](../src/gpu_utils.rs#L197-L211)).
  The reduction path already solved this with a `thread_local!`
  ([reduction.rs:44-46](../src/groth16_gpu/reduction.rs#L44-L46)) — move
  `ProofStreams` to the same pattern, especially before multiplying the
  stream count.
- icicle's own `Drop for IcicleStream` is a **no-op** that only prints a
  warning; the destroy call is commented out (`stream.rs:56-65`). This
  repo's explicit `Drop` impls are therefore load-bearing — any new streams
  must keep them or they leak.

**Risk/effort:** low risk (no semantic change), small effort, but validate
with profiling first since the payoff depends on whether the GPU has spare
concurrency headroom during large MSMs.

---

### P3 — MSM precompute factor is a single global constant, coupled across two call sites

**Where:** [gpu_utils.rs:19-20](../src/gpu_utils.rs#L19-L20)
(`PRECOMPUTE_FACTOR_G1`/`G2 = 8`), applied at
[gpu_utils.rs:286-298](../src/gpu_utils.rs#L286-L298) (upload) and
[groth16_gpu.rs:257-322](../src/groth16_gpu.rs#L257-L322) (MSM)

**Problem:** precompute factor 8 multiplies device memory 8× for whichever
point set it's applied to. It's applied identically to the tiny "pub" query
slices (`a_query_pub`, `b_g1_query_pub`, `b_g2_query_pub` — a handful to low
hundreds of points) and to the large "priv"/`h_query`/`l_query` slices (aux-
or domain-sized). Precomputing 8× for a few dozen points buys essentially
nothing; on GPU-memory-constrained deployments with very large circuits, that
memory would be better spent elsewhere.

**This is not a per-call-site one-liner — the constant is load-bearing in two
places that must agree.** `upload_points_async` sizes the device buffer as
`points.len() * precompute_factor` ([gpu_utils.rs:63-65](../src/gpu_utils.rs#L63-L65)),
and `msm_async` separately sets `cfg.precompute_factor` from whatever the
*caller* passes ([groth16_gpu.rs:257-302](../src/groth16_gpu.rs#L257-L302)).
Lowering the factor at the upload site without lowering it at the MSM call
site makes icicle read 8× a 1×-sized buffer. In the current working tree
there is a third coupling:
`h_len = h_query.len() / (PRECOMPUTE_FACTOR_G1 as usize)` bakes the same
constant into a length computation (see
[Working-tree divergence](#working-tree-divergence)).

**Impact:** low for typical circuits (the "pub" arrays are small enough
that 8× is cheap in absolute terms). Worth doing for memory-constrained
deployments, and worth doing *properly* rather than as a quick edit.

**Recommended change:** store the precompute factor **per query set** on
`ProvingKey`, alongside each `DeviceVec`, and have `create_proof_with_assignment`
read it from there instead of from the global constant. Then set 1 for the
"pub" sets and 8 (or a tunable) for `a_query_priv`, `b_g1_query_priv`,
`b_g2_query_priv`, `h_query`, `l_query`, and derive `h_len` from the stored
factor rather than the constant. This removes the coupling rather than
relying on two call sites being edited in lockstep.

**Risk/effort:** small effort, but **not** trivial-risk as long as the two
call sites are coupled — the failure mode is an out-of-bounds device read,
not a compile error.

---

### P3 — `open_device_shares` / `open_device_half_shares` are unused but expensive if ever exercised

**Where:** trait definition [mpc.rs:222-232](../src/mpc.rs#L222-L232),
Shamir impl [mpc/shamir.rs:245-297](../src/mpc/shamir.rs#L245-L297)

**Problem:** grepping the crate, these are only ever defined
(`plain.rs`, `rep3.rs`, `shamir.rs`) and never called from
`groth16_gpu.rs` — consistent with the half-share design opening only
final points, never domain-sized vectors. That's good. But if any future
code path does call them, they perform: a full device→host copy
(`to_host_vec_icicle_scalar`), a rayon-parallel per-element ark
conversion, and then a `broadcast_next` over the **entire** vector
(`mpc-core/src/protocols/shamir/network.rs:96-126` in the pinned `co-snarks`
dependency, not this repo). At domain scale (potentially millions of field
elements) that would be a severe, easy-to-hit bottleneck disguised as an
innocuous trait method.

**On the cost, precisely:** `broadcast_next` serializes the vector with
`ark-serialize` exactly **once** (`network.rs:104-106`) and then fans out
`Bytes` refcount clones — there is an explicit in-tree comment at
`network.rs:108` noting that `clone` on `Bytes` is a refcount bump, not a
copy. So the real cost is one serialize plus one deserialize per peer
(`deserialize_uncompressed_unchecked`, `network.rs:121`), *not* one
serialization per recipient. Still O(domain_size) and still worth avoiding,
but roughly `(t+1)×` cheaper than a per-recipient reading would suggest.

**Impact:** zero today (dead code on the hot path); latent risk if reused
without realizing the cost.

**Recommended change:** either confirm and document that these are
intentionally reserved for a non-GPU/slow-path use case (e.g. debugging,
small-circuit fallback) with a doc comment warning about the O(domain_size)
broadcast cost, or remove them if there's no planned caller.

**Risk/effort:** documentation-only, trivial effort.

---

### P4 — `local_mul_vec`'s internal stream sync: two of three call sites are redundant, one is a real hazard

**Where:** [mpc/shamir.rs:175-192](../src/mpc/shamir.rs#L175-L192)
(and the identical pattern in `mpc/rep3.rs`, `mpc/plain.rs`), consumed from
**three** sites: [reduction.rs:112](../src/groth16_gpu/reduction.rs#L112),
[:131](../src/groth16_gpu/reduction.rs#L131), and
[:219](../src/groth16_gpu/reduction.rs#L219)

`local_mul_vec` calls `stream.synchronize()` right after launching
`mul_scalars`. The three call sites are **not** equivalent, and the
distinction determines whether the sync can be removed:

**Site [reduction.rs:112](../src/groth16_gpu/reduction.rs#L112) (`stream_c`) —
genuine cross-stream hazard, do not touch.** This runs *before*
`ifft_in_place` mutates `eval_a`/`eval_b` in place on `stream_a`/`stream_b`
(lines 115-122). Without the sync, those in-place overwrites could race
against `stream_c`'s read of the same buffers.

**Sites [reduction.rs:131](../src/groth16_gpu/reduction.rs#L131) and
[:219](../src/groth16_gpu/reduction.rs#L219) (`stream_a`) — provably
redundant, free to remove.** Both run *after* all the FFTs complete, on
`stream_a`, and the caller synchronizes `stream_a` on the very next line
([:133](../src/groth16_gpu/reduction.rs#L133) /
[:221](../src/groth16_gpu/reduction.rs#L221)). The internal sync is
duplicated by the caller's; nothing depends on it. Removing it needs no
CUDA events and no memory tradeoff. (The win is negligible — a sync on an
already-idle stream is cheap — but it removes a misleading signal that this
function must always block.)

**Why site 112 isn't a simple fix:** the icicle Rust wrapper
(`wrappers/rust/icicle-runtime/src/stream.rs`) exposes only `create`,
`synchronize`, `is_null`, and `destroy`. There is **no event API anywhere in
icicle** — not merely unbound in the Rust wrapper, but absent from the C API
(`icicle/include/icicle/runtime.h`: 27 `extern "C"` functions, zero events)
and from the backend abstraction itself (`icicle/include/icicle/device_api.h`:
15 pure-virtual methods, zero events). CUDA events exist in the tree only as
private internals of `icicle/backend/cuda/src/msm/cuda_msm.cuh`, unexported.
The only two synchronization primitives available are
`IcicleStream::synchronize()` and `runtime::device_synchronize()`, both full
host-side blocks. So there is genuinely no way to express "stream A waits for
stream C" device-side through icicle.

**Recommended change:**
- Remove the internal sync from the `stream_a` path — but note the sync lives
  in the shared `local_mul_vec` impl, so this means either moving the sync to
  the caller at site 112 or adding a "caller will synchronize" variant.
- Do **not** speculatively change site 112. Profile first (the reduction path
  already has `#[instrument(...)]` on `witness_map_from_r1cs_eval`). If it
  shows a meaningful stall, the options are (a) an unsafe FFI shim around raw
  CUDA events, or (b) copying `eval_a`/`eval_b` before the multiply so the
  IFFTs don't wait.

**Option (a) is cheaper than it sounds.** `IcicleStream.handle` is a
**public field** (`stream.rs:11`) with a `Deref` to it (`stream.rs:68-74`),
and the handle *is* a `cudaStream_t`: `cuda_device_api.cu:98-104`
`reinterpret_cast`s a `cudaStreamCreate` result straight into
`icicleStreamHandle`, which is `typedef void*`. So the shim can pass
`stream.handle` directly to `cudaEventRecord` / `cudaStreamWaitEvent` with
no wrapper-cracking and no changes to icicle. Two caveats: it is
CUDA-backend-specific and relies on an implementation detail of a
deliberately multi-backend abstraction (so it is unsound on a non-CUDA
backend), and a null handle escalates to a device-wide block.

**Risk/effort:** trivial for the two redundant syncs; investigate-first for
site 112.

## Working-tree divergence

The line numbers above are verified against `e2a71d0`. As of this writing the
working tree has uncommitted changes to `src/gpu_utils.rs` and
`src/groth16_gpu.rs` that affect two findings:

- **The `msm_async` divisibility fallback has been removed.** At `e2a71d0`,
  `msm_async` silently dropped to `precompute_factor = 1` when the point
  count didn't divide evenly (`gpu_utils.rs:410-420`). The working tree
  replaces this with a bare `cfg.precompute_factor = precompute_factor.max(1)`.
  An earlier draft of this document cited that fallback as evidence that the
  constant "already needs per-call judgment"; that evidence no longer exists,
  and the out-of-bounds hazard described in
  [P3](#p3--msm-precompute-factor-is-a-single-global-constant-coupled-across-two-call-sites)
  is now unguarded.
- **`h_len` now derives from the constant.** `create_proof_with_assignment`
  computes `h_len = h_query.len() / (PRECOMPUTE_FACTOR_G1 as usize)`, adding
  a third site coupled to the global precompute factor. Any per-set factor
  work must update this too.
- **`from_ark` now computes the domain and roots up front**, before the
  `rayon_join_5!` conversion, rather than after the 8-stream upload sync.
  This does not change [P0](#p0--proving-key-and-ntt-domain-are-rebuilt-per-proof-unless-the-caller-caches):
  `initialize_domain` is still called from `from_ark` and still does
  `release_domain` + `initialize_domain` on global state. It does mean the
  blocking `from_host_slice(&roots)` upload and the serial
  `domain_size`-iteration roots loop now sit ahead of all the parallel
  conversion work; if this path shows up in profiling, that loop is
  parallelizable via `root_of_unity.pow(chunk_start)` per rayon chunk.

## Tricks found in the `co-snarks` dependency

The local checkout at `/home/taceo/florin/co-snarks` is at exactly the pinned
rev (`b95c341`, clean tree), so there's no newer upstream fix to backport —
everything below is about patterns already established elsewhere in the org's
MPC code that this repo doesn't yet use. Confirmed via grep: there is no
icicle/GPU code anywhere in `co-snarks` (zero case-insensitive matches), so
nothing here is a GPU-specific trick, only MPC-protocol/networking ones.

**`ShamirState::fork(n)` is the established idiom for cheap session reuse
— it directly upgrades the P1 preprocessing finding above.**
`fork` (`mpc-core/src/protocols/shamir.rs:173-186`, the `MpcState` trait impl;
trait declared at `mpc-core/src/lib.rs:21-31`) splits `n` already-buffered
correlated-randomness pairs off a parent state into an independent child with
**no network communication** — it doesn't re-run the pairwise-seed handshake
that `ShamirPreprocessing::new` does. Established uses:

- `co-circom/co-plonk/src/round2.rs:161-162` — two forks
  (`state.fork(zkey.domain_size * 6 + 2)?` and `* 7 + 2`) feeding a 2-way
  `mpc_net::join`.
- `co-circom/co-plonk/src/round3.rs:22-29` — eight forks feeding
  `mpc_net::join8` (line 31); `:50-54` — five forks of differing sizes
  feeding `join5`. Both are inside the `macro_rules! mul4vec` definition
  (starting line 20), not straight-line function code. `:279-282` — four
  forks feeding `join4`.
- `co-noir/co-acvm/src/mpc/shamir.rs:180` and
  `co-noir/co-brillig/src/mpc/shamir.rs:67-68` — these are `fork(0)`, i.e.
  **zero pairs**. They illustrate cheap state cloning for branch handling,
  not "hand out N buffered pairs", and both carry in-tree TODOs
  (`// TODO maybe take corr rand pairs here?`). Don't cite them as precedent
  for the P1 pattern.

Sizing note, repeated from P1 because it is the failure mode that matters:
under-provisioning the parent makes `fork` **panic**, not error
(`rngs.rs:80-83`).

**`state.buffer_triples(net, exact_amount)` — size the buffer ahead of a
known workload instead of relying on lazy on-demand refill.**
`co-noir/co-acvm/src/mpc/shamir.rs:773-774` and
`mpc-core/src/gadgets/merkle_tree/shamir.rs:22` (and again at `:73`) both
compute exactly how many pairs an upcoming operation needs and call
`buffer_triples` once, up front, rather than letting `ShamirState::get_pair`
hit its lazy-refill path (`mpc-core/src/protocols/shamir.rs:128-137`), which
logs a warning and doubles the *next request size* — `generation_amount *= 2`
from a `DEFAULT_PAIR_GEN_AMOUNT` of 1024 — reactively, mid-operation. Note
the warning is **suppressed for `num_parties == 3`**, so a 3-party service
hitting the refill path gets no log signal at all. Groth16 already knows its
exact need (2 pairs/proof), so a session-reuse API should call
`state.buffer_triples(net, 2 * expected_proofs_in_flight)` once.

**Even the upstream CPU reference implementation has the identical
per-call preprocessing overhead — this isn't a GPU-port regression.**
`co-circom/co-groth16/src/groth16.rs:411-425` (`ShamirCoGroth16::prove`, the
non-GPU version) calls `ShamirPreprocessing::new` fresh on every invocation
too, requesting the same 2 pairs, byte-for-byte the same pattern flagged
above. (For contrast, `ShamirCoPlonk::prove` requests
`domain_size * 222 + 15` — `co-circom/co-plonk/src/lib.rs:255`.) So the fix
likely belongs in `co-groth16` first, or in parallel.

**The CPU driver already does the exact stream-parallelism P2 asks for —
on CPU threads instead of CUDA streams.**
`co-circom/co-groth16/src/groth16.rs`'s `create_proof_with_assignment` (lines
201-326) uses `rayon_join5!` (note: no second underscore, unlike this repo's
`rayon_join_5!`) at line 221 to compute five independent coefficients
concurrently. See [P2](#p2--six-independent-msms-are-queued-on-a-single-cuda-stream)
for the exact grouping, which is subtler than "five groups, each split in
two".

**Fine-grained `tracing` spans per sub-phase — cheap to adopt, useful for
validating every finding above.**
The CPU driver names a span for each sub-phase. The exact strings matter if
you intend to grep or filter on them — three of them are longer than they
look:

| Span name | Location |
| --- | --- |
| `"compute A in create proof with assignment"` | groth16.rs:224 |
| `"compute B/G1 in create proof with assignment"` | groth16.rs:241 |
| `"compute B/G2 in create proof with assignment"` | groth16.rs:259 |
| `"msm l_query"` | groth16.rs:275 |
| `"msm h_query"` | groth16.rs:282 |
| `"r*s without networking"` | groth16.rs:290 |
| `"network round after calc coeff"` | groth16.rs:301 |
| `"finish - open two points and some adds"` | groth16.rs:306 |
| `"create proof with assignment"` | groth16.rs:201 |
| `"root of unity"` | groth16.rs:90 |
| `"Groth16 - Proof"` | groth16.rs:124 |

The GPU driver only has a handful of coarse `tracing::info!` timers (`setup`,
`MSM + stream sync`, `Coefficient assembly`, `Point openings`). Adopting the
same granularity costs nothing and would make it much easier to confirm,
before touching any code, how much of the GPU driver's timing comes from the
stream serialization in P2 versus other work.

**Multi-channel networking (`&[N; 8]` + `mpc_net::join8`) exists upstream —
checked, and confirmed *not* applicable here, which is itself informative.**
`co-circom/co-plonk/src/lib.rs:83,223-224,243-244` take an array of 8
independently-established `Network` connections per party pair (the caller
sets these up, not the prover) and run up to 8 real, concurrent
Beaver-triple-style multiplications through them via `mpc_net::join8`
(`mpc-net/src/lib.rs:278`, implemented with `std::thread::scope`, spawning 7
threads and running the 8th on the calling thread), each on its own `fork()`ed
state, because PLONK's round 3 genuinely needs several concurrent networked
multiplications that would otherwise interleave incorrectly on one shared
channel. (`mpc_net::Network` is `pub trait Network: Send + Sync` with `&self`
methods — `mpc-net/src/lib.rs:34-47` — so this is generally available
infrastructure. Note 8 concurrent is the peak, not the norm: round2 uses 2 of
the 8 and round3's line 279 group uses 4.) Groth16 needs none of this: thanks
to the half-share trick (see [Baseline](#baseline-already-optimal-no-changes-recommended)),
there are zero networked multiplications anywhere in the witness map — only
the two final point-opening broadcasts. This explains *why* Groth16's design
is so much lighter than PLONK's here, and it's the reason nothing in this
document proposes adopting multi-channel networking — it would be solving a
problem this prover doesn't have.

## How to capture a baseline

Every finding from P2 down is gated on profiling. The harness already exists:

- The test module installs a `tracing_subscriber` with span timings
  ([lib.rs:57-73](../src/lib.rs#L57-L73)).
- Representative circuits, all `#[ignore]`d and named in CI: `transaction_batched`
  (`create_proof_transaction_batched_bn254_shamir`,
  [lib.rs:382](../src/lib.rs#L382)) and the `penumbra_*` bls12_377 cases.
- Run with `RUST_LOG=co_groth16_gpu=info cargo test --profile ci-dev \
  --include-ignored -- --test-threads=1 --nocapture <test_name>`.

**Gotcha:** the default filter is `EnvFilter::try_new("oimt=info")`
([lib.rs:67](../src/lib.rs#L67)). That target does not exist — it looks like a
typo — so with `RUST_LOG` unset **none** of the `tracing::info!` timers print.
Worth fixing on its own; until then, `RUST_LOG` must be set explicitly.

Note also that `--test-threads=1` is currently required (see
[P0](#p0--proving-key-and-ntt-domain-are-rebuilt-per-proof-unless-the-caller-caches)),
so these are strictly sequential measurements.

## Suggested rollout order

0. **Capture a baseline** using the timers above, and split the `Setup timings`
   log so `evaluate_constraints`' two domain-sized uploads are measured
   separately from its CPU work. Without this, P1-setup and P2 can't be sized.
1. **P0** (cache the proving key; move `initialize_domain` out of `from_ark`)
   — largest win per unit of effort for any multi-proof workload, and the NTT
   half is a concurrency-correctness fix, not just a perf one.
2. **P1 session reuse** (`fork()` + up-front `buffer_triples`) — mirrors an
   existing `co-snarks` idiom and applies equally to the non-GPU
   `co-groth16` driver, so decide whether to fix it there first/also. Get the
   buffer sizing right: under-provisioning panics.
3. **P2** (MSM stream fan-out, plus moving `ProofStreams` to a
   `thread_local!`) — profile the existing "MSM + stream sync" timing before
   and after to confirm real GPU headroom exists.
4. **P4's two redundant syncs** — trivial, no profiling needed.
5. **P3 items** — the per-set precompute factor (do it properly, via
   `ProvingKey`, not as two coupled edits) and the dead-code documentation.
6. **P1 setup pipelining** — deliberately last. It needs the
   `ark_to_icicle_scalars` stream change (a trait-signature change across all
   three drivers) before any of it can work, and the default-stream barrier
   means partial fixes buy nothing.
7. **P4 site 112** — investigate only; do not change without profiling data
   showing it's actually hot. The fix options are more invasive than anything
   else here.
