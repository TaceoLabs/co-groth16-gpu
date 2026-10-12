//! A Groth16 proof protocol that uses a collaborative MPC protocol to generate the proof.
use crate::cuda::{Event, on_cuda_device, retain_freed_device_memory};
use crate::gpu_utils::{PRECOMPUTE_FACTOR_G1, PRECOMPUTE_FACTOR_G2, msm_into, msm_window};
use crate::msm::{self as device_msm, Group, MsmScratch, MsmSort};
use ark_bn254::Bn254;
use co_circom_types::SharedWitness;
use co_groth16::ConstraintMatrices;
use eyre::{Context, Result};
use icicle_core::curve::{Affine, Curve, Projective};
use icicle_runtime::Device;
use icicle_runtime::memory::{DeviceSlice, DeviceVec, HostOrDeviceSlice, HostSlice};
use mpc_core::MpcState;
use mpc_core::protocols::rep3::conversion::A2BType;
use mpc_core::protocols::rep3::{Rep3PrimeFieldShare, Rep3State};
use mpc_core::protocols::shamir::{ShamirPreprocessing, ShamirPrimeFieldShare, ShamirState};
use mpc_net::Network;
use std::sync::Arc;
use std::{marker::PhantomData, mem::transmute, ops::IndexMut};

use icicle_core::msm::MSM;

use crate::bridges::{ArkIcicleBridge, Bn254Bridge, ark_scalars_to_device_into};
use crate::gpu_utils::{Proof, ProofStreams, ProvingKey, VerifyingKey};
use crate::mpc::CircomGroth16Prover;
use crate::mpc::plain::PlainGroth16Driver;
use crate::mpc::rep3::Rep3Groth16Driver;
use crate::mpc::shamir::ShamirGroth16Driver;
use crate::spmv::{self, DeviceMatrix};
use crate::utils::{evaluate_constraint, evaluate_constraint_half_share};

use reduction::ReductionScratch;
pub use reduction::{CircomReduction, LibSnarkReduction, R1CSToQAP};
mod reduction;

/// The plain [`Groth16`] type.
///
/// This type is actually the [`CoGroth16`] type initialized with
/// the [`PlainGroth16Driver`], a single party (you) MPC protocol (i.e., your everyday Groth16), and using the Circom R1CSToQAPReduction by default.
/// You can use this instance to create a proof, but we recommend against it for a real use-case.
/// Have a look at the [Groth16 implementation of arkworks](https://docs.rs/ark-groth16/latest/ark_groth16/)
/// for a plain Groth16 prover.
///
/// More interesting is the [`Groth16::verify`] method. You can verify any circom Groth16 proof, be it
/// from snarkjs or one created by this project. Under the hood we use the arkwork Groth16 project for verifying.
pub struct Groth16<P> {
    phantom_data: PhantomData<P>,
}

/// A type alias for a [CoGroth16] protocol using replicated secret sharing, using the Circom R1CSToQAPReduction by default.
pub struct Rep3CoGroth16<P> {
    phantom_data: PhantomData<P>,
}

/// A type alias for a [CoGroth16] protocol using Shamir secret sharing, using the Circom R1CSToQAPReduction by default.
pub struct ShamirCoGroth16<P> {
    phantom_data: PhantomData<P>,
}

/// The internal GPU prover backing the public prover types.
///
/// Owns the device-resident proving key and every reusable device buffer and stream, all
/// allocated in [`Self::new`], so repeated [`Self::prove`] calls only pay for the
/// witness-dependent uploads and compute.
struct CoGroth16Icicle<B: ArkIcicleBridge, T: CircomGroth16Prover<B::IcicleScalarField>> {
    prepared_key: Arc<ProvingKey<B::IcicleScalarField, B::IcicleG1, B::IcicleG2>>,
    scratch: ReductionScratch<B::IcicleScalarField>,
    streams: ProofStreams,
    /// Buffers for the witness-dependent inputs; contents are re-uploaded on every run.
    eval_a: T::DeviceShares,
    eval_b: T::DeviceShares,
    /// Only allocated when the reduction requires the evaluation of the `C` matrix.
    eval_c: Option<DeviceVec<B::IcicleScalarField>>,
    public_inputs: DeviceVec<B::IcicleScalarField>,
    /// The scalars of the batched G1 MSM over [`ProvingKey::g1_batch_bases`]: one copy of
    /// `public_inputs[1..] ++ witness` per MSM of the batch, since Icicle cannot share the
    /// scalars within a batch. The first copy also serves the `b_g2` MSM.
    msm_scalars: DeviceVec<B::IcicleScalarField>,
    /// The MSM results: the G1 batch (`a`, `b_g1`, `l`), `b_g2` and `h`.
    g1_results: DeviceVec<Projective<B::IcicleG1>>,
    g2_result: DeviceVec<Projective<B::IcicleG2>>,
    h_result: DeviceVec<Projective<B::IcicleG1>>,
    /// The `A` and `B` matrices on the device, if the constraints are evaluated there.
    device_matrices: Option<DeviceMatrices<B::IcicleScalarField>>,
    /// The device MSMs, if the kernels cover the key; Icicle's otherwise.
    device_msms: Option<DeviceMsms<B>>,
}

struct DeviceMsms<B: ArkIcicleBridge> {
    /// Shared by the `a`, `b_g1`, `l` and `b_g2` MSMs.
    witness: MsmSort,
    h: MsmSort,
    g1: MsmScratch<Projective<B::IcicleG1>>,
    g2: MsmScratch<Projective<B::IcicleG2>>,
    h_scratch: MsmScratch<Projective<B::IcicleG1>>,
    /// Orders the `b_g2` MSM after the sort on the G1 stream.
    sorted: Event,
}

impl<B: ArkIcicleBridge> DeviceMsms<B> {
    fn new(key: &ProvingKey<B::IcicleScalarField, B::IcicleG1, B::IcicleG2>) -> Option<Self> {
        // G1 and G2 share the sort.
        if !on_cuda_device() || PRECOMPUTE_FACTOR_G1 != PRECOMPUTE_FACTOR_G2 {
            return None;
        }
        let pf = PRECOMPUTE_FACTOR_G1 as usize;
        let sort = |n| MsmSort::new(n, msm_window(n) as u32, pf as u32);
        let witness = sort(key.num_instance_variables - 1 + key.num_witness_variables)?;
        let h = sort(key.h_query.len() / pf)?;
        Some(Self {
            g1: MsmScratch::new(&witness, Group::G1, G1_BATCH_SIZE),
            g2: MsmScratch::new(&witness, Group::G2, 1),
            h_scratch: MsmScratch::new(&h, Group::G1, 1),
            witness,
            h,
            sorted: Event::new(),
        })
    }
}

/// The number of MSMs in the G1 batch, see [`ProvingKey::g1_batch_bases`].
const G1_BATCH_SIZE: usize = 3;

/// Wraps `f` to run on `device`; the active Icicle device is thread-local.
fn on_device<R, F: FnOnce() -> R + Send>(device: &Device, f: F) -> impl FnOnce() -> R + Send {
    let device = device.clone();
    move || {
        icicle_runtime::set_device(&device).expect("Failed to set device");
        f()
    }
}

/// `slice.split_at_mut(mid)` for device memory.
fn split_device_slice_mut<T>(
    slice: &mut DeviceSlice<T>,
    mid: usize,
) -> (&mut DeviceSlice<T>, &mut DeviceSlice<T>) {
    let len = slice.len();
    assert!(mid <= len);
    // SAFETY: the halves are disjoint subranges of `slice`, which they borrow mutably.
    unsafe {
        let ptr = slice.as_mut_ptr();
        (
            DeviceSlice::from_mut_slice(std::slice::from_raw_parts_mut(ptr, mid)),
            DeviceSlice::from_mut_slice(std::slice::from_raw_parts_mut(ptr.add(mid), len - mid)),
        )
    }
}

/// The `A` and `B` constraint matrices on the device, uploaded by the first proof since the
/// prepared key does not hold them.
struct DeviceMatrices<F> {
    a: DeviceMatrix<F>,
    b: DeviceMatrix<F>,
    /// Identifies the host matrices the device ones were uploaded from, so that proving with
    /// other matrices uploads those instead of silently using stale ones.
    source: [(usize, usize); 2],
}

fn matrices_source<F: ark_ff::Field>(matrices: &ConstraintMatrices<F>) -> [(usize, usize); 2] {
    [
        (matrices.a.as_ptr() as usize, matrices.a.len()),
        (matrices.b.as_ptr() as usize, matrices.b.len()),
    ]
}

pub type Bn254PreparedKey = ProvingKey<
    <Bn254Bridge as ArkIcicleBridge>::IcicleScalarField,
    <Bn254Bridge as ArkIcicleBridge>::IcicleG1,
    <Bn254Bridge as ArkIcicleBridge>::IcicleG2,
>;

// bls12_377/LibSnarkReduction support removed: not needed for our case atm,
// and its h_query/domain_size length mismatch with LibSnarkReduction was
// causing problems.
// type Bls12_377PreparedKey = ProvingKey<
//     <Bls12_377Bridge as ArkIcicleBridge>::IcicleScalarField,
//     <Bls12_377Bridge as ArkIcicleBridge>::IcicleG1,
//     <Bls12_377Bridge as ArkIcicleBridge>::IcicleG2,
// >;

pub fn prepare_bn254_key<R: R1CSToQAP>(
    pkey: &ark_groth16::ProvingKey<Bn254>,
    num_constraints: usize,
    num_instance_variables: usize,
) -> Bn254PreparedKey {
    ProvingKey::from_ark(
        pkey,
        num_constraints,
        num_instance_variables,
        R::requires_eval_c(),
    )
}

// pub fn prepare_bls12_377_key<R: R1CSToQAP>(
//     pkey: &ark_groth16::ProvingKey<ark_bls12_377::Bls12_377>,
//     num_constraints: usize,
//     num_instance_variables: usize,
// ) -> Bls12_377PreparedKey {
//     ProvingKey::from_ark(
//         pkey,
//         num_constraints,
//         num_instance_variables,
//         R::requires_eval_c(),
//     )
// }

impl<B: ArkIcicleBridge, T: CircomGroth16Prover<B::IcicleScalarField>> CoGroth16Icicle<B, T> {
    /// Allocates all device buffers and streams for repeated proving with the given key.
    fn new(
        prepared_key: Arc<ProvingKey<B::IcicleScalarField, B::IcicleG1, B::IcicleG2>>,
        requires_eval_c: bool,
    ) -> Self {
        retain_freed_device_memory();
        let alloc = |len| DeviceVec::device_malloc(len).expect("Failed to allocate device vector");
        let domain_size = prepared_key.domain_size;
        Self {
            scratch: ReductionScratch::new(domain_size, requires_eval_c),
            streams: ProofStreams::new(),
            eval_a: T::alloc_device_shares(domain_size),
            eval_b: T::alloc_device_shares(domain_size),
            eval_c: requires_eval_c.then(|| alloc(domain_size)),
            public_inputs: alloc(prepared_key.num_instance_variables),
            msm_scalars: alloc(
                G1_BATCH_SIZE
                    * (prepared_key.num_instance_variables - 1
                        + prepared_key.num_witness_variables),
            ),
            g1_results: DeviceVec::device_malloc(G1_BATCH_SIZE)
                .expect("Failed to allocate MSM results"),
            g2_result: DeviceVec::device_malloc(1).expect("Failed to allocate MSM result"),
            h_result: DeviceVec::device_malloc(1).expect("Failed to allocate MSM result"),
            device_matrices: None,
            device_msms: DeviceMsms::new(&prepared_key),
            prepared_key,
        }
    }

    /// Execute the Groth16 prover using the internal MPC driver. `U` is the CPU-side driver
    /// matching `T`.
    ///
    /// After uploading the witness, the MSMs that only depend on it run (Icicle's on a second
    /// thread: its MSM calls return only once their bucket sort is done) while this one evaluates
    /// the constraints and runs the reduction. Icicle's streams are blocking streams, so while
    /// the MSMs run nothing here may use the legacy default stream (synchronous copies,
    /// memsets, allocations or frees): it would wait for them.
    fn prove<
        N: Network,
        R: R1CSToQAP,
        U: co_groth16::CircomGroth16Prover<B::ArkPairing> + 'static,
    >(
        &mut self,
        net: &N,
        state: &mut T::State,
        matrices: &ConstraintMatrices<B::ArkScalarField>,
        public_inputs: &[B::ArkScalarField],
        private_witness: &[U::ArithmeticShare],
    ) -> eyre::Result<Proof<B::IcicleScalarField, B::IcicleG1, B::IcicleG2>> {
        let setup_timer = std::time::Instant::now();
        let t_id = state.id();
        let adds_public = T::adds_public_values(t_id);
        // SAFETY: matching GPU/CPU driver pairs use the same PartyID type
        let id = unsafe {
            transmute::<&<T::State as MpcState>::PartyID, &<U::State as MpcState>::PartyID>(&t_id)
        };
        let domain_size = self.prepared_key.domain_size;
        let num_constraints = self.prepared_key.num_constraints;

        // Nothing runs on the device yet, so these may be synchronous.
        ark_scalars_to_device_into(public_inputs, &mut self.public_inputs);
        self.upload_msm_scalars::<U>(adds_public, public_inputs, private_witness);
        let evaluate_on_device = self.update_device_matrices(matrices);
        let device = icicle_runtime::get_active_device().expect("Failed to get active device");

        let Self {
            prepared_key,
            scratch,
            streams,
            eval_a,
            eval_b,
            eval_c,
            public_inputs: public_inputs_buf,
            msm_scalars,
            g1_results,
            g2_result,
            h_result,
            device_matrices,
            device_msms,
        } = self;
        let key = prepared_key.as_ref();
        let streams = &*streams;
        let msm_scalars = &*msm_scalars;
        let msm_size = msm_scalars.len() / G1_BATCH_SIZE;
        let pub_len = public_inputs.len() - 1;

        std::thread::scope(|s| -> eyre::Result<()> {
            // The `a`, `b_g1` and `l` MSMs as one batch, and `b_g2` on another stream.
            let msms = if let Some(m) = device_msms.as_mut() {
                m.witness.sort(&msm_scalars[..msm_size], &streams.g1);
                m.sorted.order(&streams.g1, &streams.g2);
                device_msm::msm(
                    Group::G1,
                    &m.witness,
                    &key.g1_batch_bases,
                    msm_size * PRECOMPUTE_FACTOR_G1 as usize,
                    &mut m.g1,
                    g1_results,
                    &streams.g1,
                );
                device_msm::msm(
                    Group::G2,
                    &m.witness,
                    &key.b_g2_query,
                    0,
                    &mut m.g2,
                    g2_result,
                    &streams.g2,
                );
                None
            } else {
                Some(s.spawn(on_device(&device, || {
                    msm_into(
                        &key.g1_batch_bases,
                        msm_scalars,
                        g1_results,
                        &streams.g1,
                        PRECOMPUTE_FACTOR_G1,
                    );
                    msm_into(
                        &key.b_g2_query,
                        &msm_scalars[..msm_size],
                        g2_result,
                        &streams.g2,
                        PRECOMPUTE_FACTOR_G2,
                    );
                })))
            };

            let stream = &streams.eval;
            if evaluate_on_device {
                let DeviceMatrices { a, b, .. } = device_matrices.as_ref().unwrap();
                for (matrix, evals) in [(a, &mut *eval_a), (b, &mut *eval_b)] {
                    let evals = T::single_device_vec(evals).unwrap();
                    evals
                        .index_mut(num_constraints..)
                        .memset_async(0, domain_size - num_constraints, stream)
                        .expect("Failed to zero device buffer tail");
                    matrix.mul_vec(
                        public_inputs_buf,
                        &msm_scalars[pub_len..msm_size],
                        &mut evals[..num_constraints],
                        stream,
                    );
                }
            } else {
                // Synchronous uploads, so these wait for the MSMs (only with Rep3).
                let eval = evaluate_constraint::<B::ArkPairing, U>(
                    *id,
                    domain_size,
                    &matrices.a,
                    public_inputs,
                    private_witness,
                );
                T::shares_to_device_into::<B, U>(&eval, eval_a);
                let eval = evaluate_constraint::<B::ArkPairing, U>(
                    *id,
                    domain_size,
                    &matrices.b,
                    public_inputs,
                    private_witness,
                );
                T::shares_to_device_into::<B, U>(&eval, eval_b);
            }
            if let Some(eval_c_buf) = eval_c.as_mut() {
                let eval = evaluate_constraint_half_share::<B::ArkPairing, U>(
                    *id,
                    domain_size,
                    &matrices.c,
                    public_inputs,
                    private_witness,
                );
                T::half_shares_to_device_into::<B, U>(&eval, eval_c_buf);
            }
            T::write_trivial_shares_into(t_id, public_inputs_buf, eval_a, num_constraints, stream);
            stream.synchronize().unwrap();
            tracing::info!(
                "Constraint evaluation + device upload took {} ms",
                setup_timer.elapsed().as_millis()
            );

            let timer = std::time::Instant::now();
            R::witness_map_from_r1cs_eval::<B, T>(
                state,
                eval_a,
                eval_b,
                eval_c.as_mut(),
                &key.precomputed_roots,
                domain_size,
                scratch,
            )?;
            tracing::info!(
                "Witness map computation took {} ms",
                timer.elapsed().as_millis()
            );
            if let Some(m) = device_msms.as_mut() {
                m.h.sort(&scratch.h, &streams.h);
                device_msm::msm(
                    Group::G1,
                    &m.h,
                    &key.h_query,
                    0,
                    &mut m.h_scratch,
                    h_result,
                    &streams.h,
                );
            } else {
                msm_into(
                    &key.h_query,
                    &scratch.h,
                    h_result,
                    &streams.h,
                    PRECOMPUTE_FACTOR_G1,
                );
            }

            if let Some(msms) = msms {
                msms.join()
                    .unwrap_or_else(|panic| std::panic::resume_unwind(panic));
            }
            Ok(())
        })?;

        let (r, s) = (T::rand::<_, B>(net, state)?, T::rand::<_, B>(net, state)?);
        self.finish_proof(net, state, r, s)
    }

    /// Writes `public_inputs[1..] ++ witness` into every copy in `self.msm_scalars`. A party that
    /// does not add public values (see [`CircomGroth16Prover::adds_public_values`]) writes zeros
    /// for the public inputs, so that they still count exactly once in the reconstruction.
    fn upload_msm_scalars<U: co_groth16::CircomGroth16Prover<B::ArkPairing> + 'static>(
        &mut self,
        adds_public: bool,
        public_inputs: &[B::ArkScalarField],
        private_witness: &[U::ArithmeticShare],
    ) {
        let msm_size = self.msm_scalars.len() / G1_BATCH_SIZE;
        let pub_len = public_inputs.len() - 1;
        let (first, copies) = split_device_slice_mut(&mut self.msm_scalars, msm_size);
        {
            let (public, witness) = split_device_slice_mut(first, pub_len);
            if adds_public {
                ark_scalars_to_device_into(&public_inputs[1..], public);
            } else {
                public
                    .memset(0, pub_len)
                    .expect("Failed to zero device buffer");
            }
            T::shares_to_half_share_device_into::<B, U>(private_witness, witness);
        }
        for i in 0..G1_BATCH_SIZE - 1 {
            copies
                .index_mut(i * msm_size..(i + 1) * msm_size)
                .copy(first)
                .expect("Failed to copy MSM scalars");
        }
    }

    /// Makes sure `self.device_matrices` holds `matrices` if the constraints can be evaluated
    /// on the device, i.e. the CUDA kernels were built and a share is a single field
    /// element (not with Rep3). Returns whether they can.
    fn update_device_matrices(&mut self, matrices: &ConstraintMatrices<B::ArkScalarField>) -> bool {
        if !spmv::available() || T::single_device_vec(&mut self.eval_a).is_none() {
            return false;
        }
        let source = matrices_source(matrices);
        if self.device_matrices.as_ref().map(|m| m.source) != Some(source) {
            self.device_matrices = Some(DeviceMatrices {
                a: DeviceMatrix::new(&matrices.a),
                b: DeviceMatrix::new(&matrices.b),
                source,
            });
        }
        true
    }

    /// `initial + first_query + vk_param + acc`, where the public `first_query` and `vk_param`
    /// are only added by the parties that may, and `acc` is an MSM result over scalars that
    /// already account for that (see [`Self::upload_msm_scalars`]).
    fn calculate_coeff<C>(
        id: <T::State as MpcState>::PartyID,
        initial: Affine<C>,
        first_query: Affine<C>,
        vk_param: Affine<C>,
        acc: Affine<C>,
    ) -> Affine<C>
    where
        C: Curve<ScalarField = B::IcicleScalarField> + MSM<C>,
    {
        let mut res = initial;
        T::add_assign_points_public_hs::<C>(id, &mut res, &first_query);
        T::add_assign_points_public_hs::<C>(id, &mut res, &vk_param);
        (res.to_projective() + acc.to_projective()).into()
    }

    /// Waits for the MSMs and assembles the proof. The first opening only needs the G1 batch,
    /// so it happens while the `h` and `b_g2` MSMs may still be running.
    fn finish_proof<N: Network>(
        &mut self,
        net: &N,
        state: &mut T::State,
        r: T::ArithmeticShare,
        s: T::ArithmeticShare,
    ) -> eyre::Result<Proof<B::IcicleScalarField, B::IcicleG1, B::IcicleG2>> {
        let total_timer = std::time::Instant::now();
        let ProvingKey {
            vk,
            beta_g1,
            delta_g1,
            a_query_first,
            b_g1_query_first,
            b_g2_query_first,
            ..
        } = self.prepared_key.as_ref();
        let VerifyingKey {
            alpha_g1,
            beta_g2,
            delta_g2,
            ..
        } = vk;
        let delta_g1 = delta_g1.to_projective();
        let delta_g2 = delta_g2.to_projective();
        let id = state.id();
        let streams = &self.streams;

        // The results are read on the MSMs' own streams: a synchronous copy would wait for all.
        let mut g1_batch = [Projective::<B::IcicleG1>::zero(); G1_BATCH_SIZE];
        self.g1_results
            .copy_to_host_async(HostSlice::from_mut_slice(&mut g1_batch), &streams.g1)
            .expect("Failed to read back MSM results");
        streams.g1.synchronize().unwrap();
        let [acc_r_g1, acc_s_g1, l_acc] = g1_batch;
        tracing::info!(
            "Waiting for the G1 MSMs took {} ms",
            total_timer.elapsed().as_millis()
        );

        let r_hs = T::to_half_share(&r);
        let g_a = Self::calculate_coeff::<B::IcicleG1>(
            id,
            (delta_g1 * r_hs).into(),
            *a_query_first,
            *alpha_g1,
            acc_r_g1.into(),
        );
        // In original implementation this is skipped if r==0, however r is shared in our case
        let s_hs = T::to_half_share(&s);
        let g1_b = Self::calculate_coeff::<B::IcicleG1>(
            id,
            (delta_g1 * s_hs).into(),
            *b_g1_query_first,
            *beta_g1,
            acc_s_g1.into(),
        );
        // Opening g1_b = B*G1 is safe: B is masked by the fresh uniform s, and its exponent is
        // published in the proof as b = B*G2 anyway. With B*G1 public, r*B*G1 is a local
        // scalar multiplication, so both values can be opened in a single round.
        let (g_a_opened, g1_b_opened) = T::open_two_half_points_g1::<_, B>(g_a, g1_b, net, state)
            .expect("Failed to open g_a and g1_b");

        let mut h_acc = [Projective::<B::IcicleG1>::zero()];
        self.h_result
            .copy_to_host_async(HostSlice::from_mut_slice(&mut h_acc), &streams.h)
            .expect("Failed to read back MSM result");
        let mut acc_s_g2 = [Projective::<B::IcicleG2>::zero()];
        self.g2_result
            .copy_to_host_async(HostSlice::from_mut_slice(&mut acc_s_g2), &streams.g2)
            .expect("Failed to read back MSM result");
        streams.h.synchronize().unwrap();
        streams.g2.synchronize().unwrap();

        let g2_b = Self::calculate_coeff::<B::IcicleG2>(
            id,
            (delta_g2 * s_hs).into(),
            *b_g2_query_first,
            *beta_g2,
            acc_s_g2[0].into(),
        );
        let rs = T::local_mul::<B>(&r, &s, state);
        let mut g_c = g_a_opened.to_projective() * s_hs;
        g_c = g_c + g1_b_opened.to_projective() * r_hs;
        g_c = g_c - delta_g1 * rs;
        g_c = g_c + l_acc;
        g_c = g_c + h_acc[0];
        let (g_c_opened, g2_b_opened) =
            T::open_two_half_points_g1g2::<_, B>(g_c.into(), g2_b, net, state)?;
        tracing::info!(
            "Proof with assignment took {} ms",
            total_timer.elapsed().as_millis()
        );

        Ok(Proof {
            a: g_a_opened,
            b: g2_b_opened,
            c: g_c_opened,
        })
    }
}

/// Transmutes the prove inputs from the generic pairing `P` into a concrete pairing.
/// Yields `(matrices, witness_shares, public_inputs)`.
///
/// # Safety / Invariant
/// Only sound when `P` *is* the destination pairing, i.e. the values are actually built
/// for it and merely referenced through the generic types.
macro_rules! cast_prove_inputs {
    ($SrcShare:ty => $DstShare:ty, $DstField:ty, $matrices:expr, $private_witness:expr) => {{
        unsafe {
            (
                transmute::<&ConstraintMatrices<P::ScalarField>, &ConstraintMatrices<$DstField>>(
                    $matrices,
                ),
                transmute::<&[$SrcShare], &[$DstShare]>($private_witness.witness.as_slice()),
                transmute::<&[P::ScalarField], &[$DstField]>(
                    $private_witness.public_inputs.as_slice(),
                ),
            )
        }
    }};
}

/// Transmutes a proof over the concrete pairing `Src` back to the generic `Dst`.
///
/// # Safety
/// Only sound when `Src` and `Dst` are the same concrete pairing.
unsafe fn cast_proof<Src: ark_ec::pairing::Pairing, Dst: ark_ec::pairing::Pairing>(
    proof: ark_groth16::Proof<Src>,
) -> ark_groth16::Proof<Dst> {
    unsafe { transmute::<&ark_groth16::Proof<Src>, &ark_groth16::Proof<Dst>>(&proof) }.clone()
}

impl<P: ark_ec::pairing::Pairing> Groth16<P> {
    /// *Locally* create a `Groth16` proof. This is just the [`CoGroth16`] prover
    /// initialized with the [`PlainGroth16Driver`].
    ///
    /// DOES NOT PERFORM ANY MPC. For a plain prover checkout the [Groth16 implementation of arkworks](https://docs.rs/ark-groth16/latest/ark_groth16/).
    ///
    /// This is a one-shot convenience wrapper around [`Groth16Prover`]; to amortize GPU
    /// setup cost over multiple proofs, construct a [`Groth16Prover`] once and call
    /// [`Groth16Prover::prove`] repeatedly.
    pub fn plain_prove<R: R1CSToQAP>(
        pkey: &ark_groth16::ProvingKey<P>,
        prepared_bn_254_key: Option<Arc<Bn254PreparedKey>>,
        matrices: &ConstraintMatrices<P::ScalarField>,
        private_witness: SharedWitness<P::ScalarField, P::ScalarField>,
    ) -> Result<ark_groth16::Proof<P>> {
        let mut prover = match prepared_bn_254_key {
            Some(prepared_key)
                if std::any::TypeId::of::<P>() == std::any::TypeId::of::<ark_bn254::Bn254>() =>
            {
                Groth16Prover::<P, R>::from_prepared_bn254_key(prepared_key)
            }
            _ => Groth16Prover::<P, R>::new(pkey, matrices),
        };
        prover.prove(matrices, private_witness)
    }
}

/// A stateful plain (single-party) Groth16 GPU prover for the reduction `R`.
///
/// The constructor prepares the proving key on the device and allocates all scratch
/// buffers and streams, so repeated [`Self::prove`] calls only pay for the
/// witness-dependent uploads and compute.
///
/// Only BN254 is supported; bls12_377/LibSnarkReduction support was removed since its
/// h_query/domain_size length mismatch with LibSnarkReduction was causing real problems.
pub struct Groth16Prover<P, R = CircomReduction> {
    inner: CoGroth16Icicle<Bn254Bridge, PlainGroth16Driver>,
    phantom_data: PhantomData<(P, R)>,
}

impl<P: ark_ec::pairing::Pairing, R: R1CSToQAP> Groth16Prover<P, R> {
    /// Prepares the proving key on the device and creates a prover.
    pub fn new(
        pkey: &ark_groth16::ProvingKey<P>,
        matrices: &ConstraintMatrices<P::ScalarField>,
    ) -> Self {
        let inner = if std::any::TypeId::of::<P>() == std::any::TypeId::of::<ark_bn254::Bn254>() {
            // SAFETY: P == Bn254, checked above
            let pkey = unsafe {
                transmute::<&ark_groth16::ProvingKey<P>, &ark_groth16::ProvingKey<ark_bn254::Bn254>>(
                    pkey,
                )
            };
            let prepared_key = prepare_bn254_key::<R>(
                pkey,
                matrices.num_constraints,
                matrices.num_instance_variables,
            );
            CoGroth16Icicle::new(Arc::new(prepared_key), R::requires_eval_c())
        } else {
            panic!("Unsupported pairing")
        };
        Self {
            inner,
            phantom_data: PhantomData,
        }
    }

    /// Creates a BN254 prover from an already device-prepared proving key, which must
    /// have been prepared for the same reduction `R`.
    pub fn from_prepared_bn254_key(prepared_key: Arc<Bn254PreparedKey>) -> Self {
        if std::any::TypeId::of::<P>() != std::any::TypeId::of::<ark_bn254::Bn254>() {
            panic!("Unsupported pairing");
        }
        Self {
            inner: CoGroth16Icicle::new(prepared_key, R::requires_eval_c()),
            phantom_data: PhantomData,
        }
    }

    /// Creates a proof, reusing the cached GPU resources from previous runs.
    pub fn prove(
        &mut self,
        matrices: &ConstraintMatrices<P::ScalarField>,
        private_witness: SharedWitness<P::ScalarField, P::ScalarField>,
    ) -> Result<ark_groth16::Proof<P>> {
        // SAFETY (all casts below): `new`/`from_prepared_bn254_key` only succeed for P == Bn254.
        let (matrices, witness, public_inputs) = cast_prove_inputs!(
            P::ScalarField => ark_bn254::Fr,
            ark_bn254::Fr,
            matrices,
            private_witness
        );
        let icicle_proof = self
            .inner
            .prove::<_, R, co_groth16::mpc::PlainGroth16Driver>(
                &(),
                &mut (),
                matrices,
                public_inputs,
                witness,
            )?;
        Ok(unsafe { cast_proof(icicle_proof.to_ark::<Bn254Bridge>()) })
    }
}

impl<P: ark_ec::pairing::Pairing> Rep3CoGroth16<P> {
    /// This is a one-shot convenience wrapper around [`Rep3CoGroth16Prover`]; to amortize
    /// GPU setup cost over multiple proofs, construct a [`Rep3CoGroth16Prover`] once and
    /// call [`Rep3CoGroth16Prover::prove`] repeatedly.
    pub fn prove<N: Network, R: R1CSToQAP>(
        net: &N,
        pkey: &ark_groth16::ProvingKey<P>,
        prepared_bn254_key: Option<Arc<Bn254PreparedKey>>,
        matrices: &ConstraintMatrices<P::ScalarField>,
        private_witness: SharedWitness<P::ScalarField, Rep3PrimeFieldShare<P::ScalarField>>,
    ) -> Result<ark_groth16::Proof<P>> {
        let mut prover = match prepared_bn254_key {
            Some(prepared_key) => Rep3CoGroth16Prover::<P, R>::from_prepared_key(prepared_key),
            None => Rep3CoGroth16Prover::<P, R>::new(pkey, matrices),
        };
        prover.prove::<N>(net, matrices, private_witness)
    }

    /// Create a [`ark_groth16::Proof`] by locally translating the REP3 `witness` into a 3-party
    /// Shamir sharing (no communication) and running [`ShamirCoGroth16::prove`], giving the same
    /// trust assumption as [`Self::prove`] with a cheaper online phase.
    ///
    /// # Errors
    /// Returns an error if `net.id()` is not a valid REP3 party id (0, 1, or 2).
    pub fn prove_with_shamir_bridge<N: Network, R: R1CSToQAP>(
        net: &N,
        pkey: &ark_groth16::ProvingKey<P>,
        prepared_bn254_key: Option<Arc<Bn254PreparedKey>>,
        matrices: &ConstraintMatrices<P::ScalarField>,
        witness: SharedWitness<P::ScalarField, Rep3PrimeFieldShare<P::ScalarField>>,
    ) -> Result<ark_groth16::Proof<P>> {
        let translated_witness = ShamirState::translate_primefield_repshare_vec(
            witness.witness,
            net.id().try_into().context("not a valid party id")?,
        );
        ShamirCoGroth16::<P>::prove::<N, R>(
            net,
            3, // number of parties is 3 for REP3
            1, // threshold is 1 for REP3
            pkey,
            prepared_bn254_key,
            matrices,
            SharedWitness {
                public_inputs: witness.public_inputs,
                witness: translated_witness,
            },
        )
    }
}

impl<P: ark_ec::pairing::Pairing> ShamirCoGroth16<P> {
    /// Create a proof by running the collaborative Groth16 prover under Shamir secret sharing,
    /// secure against `threshold` semi-honest corruptions among `num_parties` parties.
    /// `num_parties` must be at least `2 * threshold + 1`, since `g_c` is opened as a
    /// degree-`2*threshold` sharing.
    ///
    /// Correlated randomness is preprocessed over `net` before the online phase.
    ///
    /// This is a one-shot convenience wrapper around [`ShamirCoGroth16Prover`]; to amortize
    /// GPU setup cost over multiple proofs, construct a [`ShamirCoGroth16Prover`] once and
    /// call [`ShamirCoGroth16Prover::prove`] repeatedly.
    pub fn prove<N: Network, R: R1CSToQAP>(
        net: &N,
        num_parties: usize,
        threshold: usize,
        pkey: &ark_groth16::ProvingKey<P>,
        prepared_bn254_key: Option<Arc<Bn254PreparedKey>>,
        matrices: &ConstraintMatrices<P::ScalarField>,
        private_witness: SharedWitness<P::ScalarField, ShamirPrimeFieldShare<P::ScalarField>>,
    ) -> Result<ark_groth16::Proof<P>> {
        let mut prover = match prepared_bn254_key {
            Some(prepared_key) => ShamirCoGroth16Prover::<P, R>::from_prepared_key(
                num_parties,
                threshold,
                prepared_key,
            ),
            None => ShamirCoGroth16Prover::<P, R>::new(num_parties, threshold, pkey, matrices),
        };
        prover.prove::<N>(net, matrices, private_witness)
    }
}

/// A stateful REP3 Groth16 GPU prover for the reduction `R`.
///
/// The constructor allocates all GPU resources (device-resident proving key, scratch
/// buffers, streams), so repeated [`Self::prove`] calls only pay for the
/// witness-dependent uploads and compute.
///
/// Currently only supports BN254.
pub struct Rep3CoGroth16Prover<P, R = CircomReduction> {
    inner: CoGroth16Icicle<Bn254Bridge, Rep3Groth16Driver>,
    phantom_data: PhantomData<(P, R)>,
}

impl<P: ark_ec::pairing::Pairing, R: R1CSToQAP> Rep3CoGroth16Prover<P, R> {
    /// Creates a prover from an already device-prepared proving key, which must have been
    /// prepared for the same reduction `R`.
    pub fn from_prepared_key(prepared_key: Arc<Bn254PreparedKey>) -> Self {
        if std::any::TypeId::of::<P>() != std::any::TypeId::of::<ark_bn254::Bn254>() {
            panic!("Unsupported pairing");
        }
        Self {
            inner: CoGroth16Icicle::new(prepared_key, R::requires_eval_c()),
            phantom_data: PhantomData,
        }
    }

    /// Prepares the proving key on the device and creates a prover.
    pub fn new(
        pkey: &ark_groth16::ProvingKey<P>,
        matrices: &ConstraintMatrices<P::ScalarField>,
    ) -> Self {
        if std::any::TypeId::of::<P>() != std::any::TypeId::of::<ark_bn254::Bn254>() {
            panic!("Unsupported pairing");
        }
        // SAFETY: P == Bn254, checked above
        let pkey = unsafe {
            transmute::<&ark_groth16::ProvingKey<P>, &ark_groth16::ProvingKey<ark_bn254::Bn254>>(
                pkey,
            )
        };
        let prepared_key = prepare_bn254_key::<R>(
            pkey,
            matrices.num_constraints,
            matrices.num_instance_variables,
        );
        Self::from_prepared_key(Arc::new(prepared_key))
    }

    /// Creates a proof, reusing the cached GPU resources from previous runs.
    /// See [`Rep3CoGroth16::prove`] for the protocol description.
    pub fn prove<N: Network>(
        &mut self,
        net: &N,
        matrices: &ConstraintMatrices<P::ScalarField>,
        private_witness: SharedWitness<P::ScalarField, Rep3PrimeFieldShare<P::ScalarField>>,
    ) -> Result<ark_groth16::Proof<P>> {
        // SAFETY: the constructors guarantee P == Bn254
        let (matrices, witness, public_inputs) = cast_prove_inputs!(
            Rep3PrimeFieldShare<P::ScalarField> => Rep3PrimeFieldShare<ark_bn254::Fr>,
            ark_bn254::Fr,
            matrices,
            private_witness
        );

        let mut state = Rep3State::new(net, A2BType::default())?;

        let icicle_proof = self
            .inner
            .prove::<N, R, co_groth16::mpc::Rep3Groth16Driver>(
                net,
                &mut state,
                matrices,
                public_inputs,
                witness,
            )?;
        // SAFETY: the constructors guarantee P == Bn254
        Ok(unsafe { cast_proof(icicle_proof.to_ark::<Bn254Bridge>()) })
    }
}

/// A stateful Shamir Groth16 GPU prover for the reduction `R`.
///
/// The constructor allocates all GPU resources (device-resident proving key, scratch
/// buffers, streams), so repeated [`Self::prove`] calls only pay for the
/// witness-dependent uploads and compute.
///
/// Currently only supports BN254.
pub struct ShamirCoGroth16Prover<P, R = CircomReduction> {
    num_parties: usize,
    threshold: usize,
    inner: CoGroth16Icicle<Bn254Bridge, ShamirGroth16Driver<ark_bn254::Fr>>,
    phantom_data: PhantomData<(P, R)>,
}

impl<P: ark_ec::pairing::Pairing, R: R1CSToQAP> ShamirCoGroth16Prover<P, R> {
    /// Creates a prover from an already device-prepared proving key, which must have been
    /// prepared for the same reduction `R`.
    pub fn from_prepared_key(
        num_parties: usize,
        threshold: usize,
        prepared_key: Arc<Bn254PreparedKey>,
    ) -> Self {
        if std::any::TypeId::of::<P>() != std::any::TypeId::of::<ark_bn254::Bn254>() {
            panic!("Unsupported pairing");
        }
        Self {
            num_parties,
            threshold,
            inner: CoGroth16Icicle::new(prepared_key, R::requires_eval_c()),
            phantom_data: PhantomData,
        }
    }

    /// Prepares the proving key on the device and creates a prover.
    pub fn new(
        num_parties: usize,
        threshold: usize,
        pkey: &ark_groth16::ProvingKey<P>,
        matrices: &ConstraintMatrices<P::ScalarField>,
    ) -> Self {
        if std::any::TypeId::of::<P>() != std::any::TypeId::of::<ark_bn254::Bn254>() {
            panic!("Unsupported pairing");
        }
        // SAFETY: P == Bn254, checked above
        let pkey = unsafe {
            transmute::<&ark_groth16::ProvingKey<P>, &ark_groth16::ProvingKey<ark_bn254::Bn254>>(
                pkey,
            )
        };
        let prepared_key = prepare_bn254_key::<R>(
            pkey,
            matrices.num_constraints,
            matrices.num_instance_variables,
        );
        Self::from_prepared_key(num_parties, threshold, Arc::new(prepared_key))
    }

    /// Creates a proof, reusing the cached GPU resources from previous runs.
    /// See [`ShamirCoGroth16::prove`] for the protocol description.
    pub fn prove<N: Network>(
        &mut self,
        net: &N,
        matrices: &ConstraintMatrices<P::ScalarField>,
        private_witness: SharedWitness<P::ScalarField, ShamirPrimeFieldShare<P::ScalarField>>,
    ) -> Result<ark_groth16::Proof<P>> {
        // SAFETY: the constructors guarantee P == Bn254
        let (matrices, witness, public_inputs) = cast_prove_inputs!(
            ShamirPrimeFieldShare<P::ScalarField> => ShamirPrimeFieldShare<ark_bn254::Fr>,
            ark_bn254::Fr,
            matrices,
            private_witness
        );

        // we need 2 corr rand pairs for the two rand calls
        let num_pairs = 2;
        let preprocessing =
            ShamirPreprocessing::new(self.num_parties, self.threshold, num_pairs, net)?;
        let mut state = ShamirState::from(preprocessing);

        let icicle_proof = self
            .inner
            .prove::<N, R, co_groth16::mpc::ShamirGroth16Driver>(
                net,
                &mut state,
                matrices,
                public_inputs,
                witness,
            )?;
        // SAFETY: the constructors guarantee P == Bn254
        Ok(unsafe { cast_proof(icicle_proof.to_ark::<Bn254Bridge>()) })
    }
}
