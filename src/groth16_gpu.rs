//! A Groth16 proof protocol that uses a collaborative MPC protocol to generate the proof.
use crate::gpu_utils::{PRECOMPUTE_FACTOR_G1, PRECOMPUTE_FACTOR_G2, msm_into};
use ark_bn254::Bn254;
use ark_ec::CurveGroup;
use co_circom_types::SharedWitness;
use co_groth16::ConstraintMatrices;
use eyre::{Context, Result};
use icicle_core::curve::{Affine, Projective};
use icicle_runtime::memory::{DeviceVec, HostOrDeviceSlice, HostSlice};
use mpc_core::MpcState;
use mpc_core::protocols::rep3::conversion::A2BType;
use mpc_core::protocols::rep3::{Rep3PrimeFieldShare, Rep3State};
use mpc_core::protocols::shamir::{ShamirPreprocessing, ShamirPrimeFieldShare, ShamirState};
use mpc_net::Network;
use std::sync::Arc;
use std::{marker::PhantomData, mem::transmute, ops::IndexMut};

use crate::bridges::{
    ArkIcicleBridge, Bn254Bridge, ark_scalars_to_device_into, icicle_to_ark_scalar,
};
use crate::gpu_utils::{ProofStreams, ProvingKey};
use crate::mpc::CircomGroth16Prover;
use crate::mpc::plain::PlainGroth16Driver;
use crate::mpc::rep3::Rep3Groth16Driver;
use crate::mpc::shamir::ShamirGroth16Driver;
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
    /// `public_inputs[1..] ++ private_witness_half_shares`, kept contiguous so the
    /// `a_query`, `b_g1_query`, and `b_g2_query` MSMs can each run once over the whole
    /// instance+witness vector instead of once for the public part and once for the private
    /// part; the `l_query` MSM also reads its `[pub_len..]` tail directly. The private
    /// segment is uploaded straight into this buffer (see `prove`), so it only needs one
    /// device write per proof, not a separate upload-then-copy.
    combined_scalars: DeviceVec<B::IcicleScalarField>,
    /// Result slots for the G1 MSMs, in the order listed by [`MSM_RESULTS_G1`].
    msm_results_g1: DeviceVec<Projective<B::IcicleG1>>,
    /// Result slots for the G2 MSMs, in the order listed by [`MSM_RESULTS_G2`].
    msm_results_g2: DeviceVec<Projective<B::IcicleG2>>,
    ark_key: ArkKeyConstants<B>,
}

/// The fixed proving-key points that proof assembly needs in arkworks form.
///
/// Converted once when the prover is built rather than on every proof: `icicle_to_ark_g2`
/// goes through `ark_ec::short_weierstrass::Affine::new`, whose subgroup check costs about
/// 77 us per BN254 G2 point.
struct ArkKeyConstants<B: ArkIcicleBridge> {
    alpha_g1: B::ArkG1,
    beta_g1: B::ArkG1,
    beta_g2: B::ArkG2,
    delta_g1: B::ArkG1,
    delta_g2: B::ArkG2,
    a_query_first: B::ArkG1,
    b_g1_query_first: B::ArkG1,
    b_g2_query_first: B::ArkG2,
}

impl<B: ArkIcicleBridge> ArkKeyConstants<B> {
    fn new(key: &ProvingKey<B::IcicleScalarField, B::IcicleG1, B::IcicleG2>) -> Self {
        let g1 = |p: &Affine<B::IcicleG1>| B::ArkG1::from(B::icicle_to_ark_g1(*p));
        let g2 = |p: &Affine<B::IcicleG2>| B::ArkG2::from(B::icicle_to_ark_g2(*p));
        Self {
            alpha_g1: g1(&key.vk.alpha_g1),
            beta_g1: g1(&key.beta_g1),
            beta_g2: g2(&key.vk.beta_g2),
            delta_g1: g1(&key.delta_g1),
            delta_g2: g2(&key.vk.delta_g2),
            a_query_first: g1(&key.a_query_first),
            b_g1_query_first: g1(&key.b_g1_query_first),
            b_g2_query_first: g2(&key.b_g2_query_first),
        }
    }
}

/// The G1 MSMs of a single proof, in the order they occupy `msm_results_g1`.
const MSM_RESULTS_G1: usize = 4;
/// The G2 MSMs of a single proof, in the order they occupy `msm_results_g2`.
const MSM_RESULTS_G2: usize = 1;

/// Slot indices within `msm_results_g1`/`msm_results_g2`. Shared by
/// `CoGroth16Icicle::launch_h_msm` and `launch_witness_independent_msms` (which write into
/// these slots) and `finish_proof_with_assignment` (which reads them back), so the mapping
/// only needs to be edited in one place.
const MSM_SLOT_A: usize = 0;
const MSM_SLOT_B_G1: usize = 1;
const MSM_SLOT_L: usize = 2;
const MSM_SLOT_H: usize = 3;
const MSM_SLOT_B_G2: usize = 0;

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
        let alloc = |len| DeviceVec::device_malloc(len).expect("Failed to allocate device vector");
        let domain_size = prepared_key.domain_size;
        let ark_key = ArkKeyConstants::new(&prepared_key);

        // `eval_a`/`eval_b`/`eval_c` only ever get `num_constraints` entries from the host on
        // each `prove` call; the domain padding beyond that is always zero and, since
        // `num_constraints` is fixed for the lifetime of this prover (one instance per
        // circuit), only needs zeroing once here rather than on every proof.
        let mut eval_a = T::alloc_device_shares(domain_size);
        let mut eval_b = T::alloc_device_shares(domain_size);
        T::zero_device_shares_from(&mut eval_a, prepared_key.num_constraints);
        T::zero_device_shares_from(&mut eval_b, prepared_key.num_constraints);
        let mut eval_c = requires_eval_c.then(|| alloc(domain_size));
        if let Some(eval_c_buf) = eval_c.as_mut() {
            eval_c_buf
                .index_mut(prepared_key.num_constraints..)
                .memset(0, domain_size - prepared_key.num_constraints)
                .expect("Failed to zero device buffer tail");
        }

        Self {
            scratch: ReductionScratch::new(domain_size, requires_eval_c),
            streams: ProofStreams::new(),
            eval_a,
            eval_b,
            eval_c,
            public_inputs: alloc(prepared_key.num_instance_variables),
            combined_scalars: alloc(
                prepared_key.num_instance_variables - 1 + prepared_key.num_witness_variables,
            ),
            msm_results_g1: DeviceVec::device_malloc(MSM_RESULTS_G1)
                .expect("Failed to allocate G1 MSM result buffer"),
            msm_results_g2: DeviceVec::device_malloc(MSM_RESULTS_G2)
                .expect("Failed to allocate G2 MSM result buffer"),
            ark_key,
            prepared_key,
        }
    }

    /// Execute the Groth16 prover using the internal MPC driver: evaluates the constraints
    /// on the host, uploads the witness-dependent inputs into the pre-allocated device
    /// buffers, and creates the proof. `U` is the CPU-side driver matching `T`.
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
    ) -> eyre::Result<ark_groth16::Proof<B::ArkPairing>> {
        let setup_timer = std::time::Instant::now();
        let id = state.id();
        // SAFETY: matching GPU/CPU driver pairs use the same PartyID type
        let id = unsafe {
            transmute::<&<T::State as MpcState>::PartyID, &<U::State as MpcState>::PartyID>(&id)
        };
        let eval_a = evaluate_constraint::<B::ArkPairing, U>(
            *id,
            &matrices.a,
            public_inputs,
            private_witness,
        );
        T::shares_to_device_into::<B, U>(&eval_a, &mut self.eval_a, 0);

        let eval_b = evaluate_constraint::<B::ArkPairing, U>(
            *id,
            &matrices.b,
            public_inputs,
            private_witness,
        );
        T::shares_to_device_into::<B, U>(&eval_b, &mut self.eval_b, 0);

        if let Some(eval_c_buf) = self.eval_c.as_mut() {
            let eval_c = evaluate_constraint_half_share::<B::ArkPairing, U>(
                *id,
                &matrices.c,
                public_inputs,
                private_witness,
            );
            T::half_shares_to_device_into::<B, U>(&eval_c, eval_c_buf, 0);
        }

        // Upload the private witness straight into `combined_scalars`'s private segment,
        // where the `a`/`b_g1`/`b_g2`/`l` MSMs read it from directly (see
        // `launch_witness_independent_msms`); no separate witness buffer or copy needed.
        let pub_len = self.prepared_key.num_instance_variables - 1;
        T::shares_to_half_share_device_into::<B, U>(
            private_witness,
            &mut self.combined_scalars,
            pub_len,
        );
        ark_scalars_to_device_into(public_inputs, &mut self.public_inputs);

        tracing::info!(
            "Constraint evaluation + device upload took {} ms",
            setup_timer.elapsed().as_millis()
        );

        self.prove_inner::<N, R>(net, state)
    }

    /// Computes the QAP witness and creates the proof from the uploaded inputs.
    ///
    /// Launches the MSMs that only depend on the already-uploaded public inputs and witness
    /// shares *before* the witness-map reduction, so the GPU can work on both at once: the
    /// MSMs run on `self.streams.g1`/`g2`, the reduction on its own (thread-local) streams,
    /// and the host issues both without blocking in between.
    fn prove_inner<N: Network, R: R1CSToQAP>(
        &mut self,
        net: &N,
        state: &mut T::State,
    ) -> eyre::Result<ark_groth16::Proof<B::ArkPairing>> {
        let msm_launch_timer = std::time::Instant::now();
        self.launch_witness_independent_msms(state.id());
        tracing::info!(
            "Launching witness-independent MSMs took {} ms",
            msm_launch_timer.elapsed().as_millis()
        );

        let timer_start = std::time::Instant::now();
        R::witness_map_from_r1cs_eval::<B, T>(
            state,
            &mut self.eval_a,
            &mut self.eval_b,
            self.eval_c.as_mut(),
            &self.public_inputs,
            &self.prepared_key.precomputed_roots,
            self.prepared_key.num_constraints,
            self.prepared_key.domain_size,
            &mut self.scratch,
        )?;
        tracing::info!(
            "Witness map computation took {} ms",
            timer_start.elapsed().as_millis()
        );

        // `h_acc`'s MSM only depends on `scratch.h` (just produced above), not on `r`/`s`,
        // so it's launched here rather than in `finish_proof_with_assignment`: it can then
        // run on the GPU concurrently with the (typically local, but not free) `rand` calls
        // below instead of waiting for them first.
        self.launch_h_msm();

        let (r, s) = (T::rand::<_, B>(net, state)?, T::rand::<_, B>(net, state)?);

        self.finish_proof_with_assignment(net, state, r, s)
    }

    /// Launches the one MSM (`h_acc`) that depends on the QAP witness `h` rather than only
    /// on the already-uploaded public inputs and witness shares. See
    /// [`Self::launch_witness_independent_msms`] for the other four.
    fn launch_h_msm(&mut self) {
        let ProvingKey { h_query, .. } = self.prepared_key.as_ref();
        let h = &self.scratch.h;

        msm_into(
            h_query,
            h,
            &mut self.msm_results_g1[MSM_SLOT_H..MSM_SLOT_H + 1],
            &self.streams.g1,
            PRECOMPUTE_FACTOR_G1,
        );
    }

    /// Builds the combined MSM scalar buffer and launches the four MSMs that only depend on
    /// it (i.e. not on the QAP witness `h`, which the witness map produces): `a`, `b_g1`,
    /// `b_g2`, and `l`. Paired with [`Self::finish_proof_with_assignment`], which launches
    /// the remaining (`h`-dependent) MSM, waits for all five, and assembles the proof.
    fn launch_witness_independent_msms(&mut self, id: <T::State as MpcState>::PartyID) {
        let ProvingKey {
            a_query,
            b_g1_query,
            b_g2_query,
            l_query,
            ..
        } = self.prepared_key.as_ref();

        let stream_g1 = &self.streams.g1;
        let stream_g2 = &self.streams.g2;

        // Fill the public segment of the combined scalar buffer (gated per-protocol, see
        // `write_combined_public_segment`); the private segment was already uploaded
        // directly into `combined_scalars[pub_len..]` in `prove`. Together they're used by
        // the `a`/`b_g1`/`b_g2` MSMs below instead of a separate public-only and
        // private-only MSM each.
        let pub_len = self.public_inputs.len() - 1;
        {
            let public_tail = &self.public_inputs[1..];
            T::write_combined_public_segment(
                id,
                public_tail,
                &mut self.combined_scalars[..pub_len],
            );
        }

        msm_into(
            a_query,
            &self.combined_scalars,
            &mut self.msm_results_g1[MSM_SLOT_A..MSM_SLOT_A + 1],
            stream_g1,
            PRECOMPUTE_FACTOR_G1,
        );
        msm_into(
            b_g1_query,
            &self.combined_scalars,
            &mut self.msm_results_g1[MSM_SLOT_B_G1..MSM_SLOT_B_G1 + 1],
            stream_g1,
            PRECOMPUTE_FACTOR_G1,
        );
        msm_into(
            b_g2_query,
            &self.combined_scalars,
            &mut self.msm_results_g2[MSM_SLOT_B_G2..MSM_SLOT_B_G2 + 1],
            stream_g2,
            PRECOMPUTE_FACTOR_G2,
        );
        msm_into(
            l_query,
            &self.combined_scalars[pub_len..],
            &mut self.msm_results_g1[MSM_SLOT_L..MSM_SLOT_L + 1],
            stream_g1,
            PRECOMPUTE_FACTOR_G1,
        );
    }

    /// `initial + first_query + vk_param + combined_acc`, where `first_query`/`vk_param` are
    /// public and therefore only added by the parties that may do so (see
    /// [`CircomGroth16Prover::add_assign_point_public`]); `combined_acc` is the result of an
    /// MSM already carrying both the public and private-witness contributions (see
    /// [`CircomGroth16Prover::write_combined_public_segment`]) and is therefore always added.
    ///
    /// Stays projective throughout; the caller converts to affine once, at the end.
    fn calculate_coeff<C: CurveGroup>(
        id: <T::State as MpcState>::PartyID,
        initial: C,
        first_query: C,
        vk_param: C,
        combined_acc: C,
    ) -> C {
        let mut res = initial;
        T::add_assign_point_public::<C>(id, &mut res, &first_query);
        T::add_assign_point_public::<C>(id, &mut res, &vk_param);
        res + combined_acc
    }

    /// Waits for all five MSMs launched by [`Self::launch_witness_independent_msms`] and
    /// [`Self::launch_h_msm`], and assembles the proof.
    fn finish_proof_with_assignment<N: Network>(
        &mut self,
        net: &N,
        state: &mut T::State,
        r: T::ArithmeticShare,
        s: T::ArithmeticShare,
    ) -> eyre::Result<ark_groth16::Proof<B::ArkPairing>> {
        let total_timer = std::time::Instant::now();
        let id = state.id();

        let stream_g1 = &self.streams.g1;
        let stream_g2 = &self.streams.g2;

        let msm_timer = std::time::Instant::now();
        stream_g1.synchronize().unwrap();
        stream_g2.synchronize().unwrap();
        tracing::info!(
            "MSM stream sync took {} ms",
            msm_timer.elapsed().as_millis()
        );

        let coeff_timer = std::time::Instant::now();
        // One device-to-host copy per stream instead of one per MSM result.
        let mut results_g1 = [Projective::<B::IcicleG1>::zero(); MSM_RESULTS_G1];
        self.msm_results_g1
            .copy_to_host(HostSlice::from_mut_slice(&mut results_g1))
            .expect("Failed to read back G1 MSM results");
        let mut results_g2 = [Projective::<B::IcicleG2>::zero(); MSM_RESULTS_G2];
        self.msm_results_g2
            .copy_to_host(HostSlice::from_mut_slice(&mut results_g2))
            .expect("Failed to read back G2 MSM results");

        let (acc_r_g1, acc_s_g1, l_acc, h_acc) = (
            results_g1[MSM_SLOT_A],
            results_g1[MSM_SLOT_B_G1],
            results_g1[MSM_SLOT_L],
            results_g1[MSM_SLOT_H],
        );
        let acc_s_g2 = results_g2[MSM_SLOT_B_G2];

        // Everything below is host-side curve arithmetic, so move into arkworks (built with
        // the `asm` feature here) rather than icicle's generic host C++ implementations, and
        // stay in projective coordinates so only the three published points pay for the
        // conversion to affine.
        let msm_g1 = |p: Projective<B::IcicleG1>| -> B::ArkG1 {
            B::ArkG1::from(B::icicle_to_ark_g1(p.into()))
        };
        let msm_g2 = |p: Projective<B::IcicleG2>| -> B::ArkG2 {
            B::ArkG2::from(B::icicle_to_ark_g2(p.into()))
        };
        let ArkKeyConstants {
            alpha_g1,
            beta_g1,
            beta_g2,
            delta_g1,
            delta_g2,
            a_query_first,
            b_g1_query_first,
            b_g2_query_first,
        } = self.ark_key;

        let r_hs: B::ArkScalarField = icicle_to_ark_scalar(T::to_half_share(&r));
        let g_a = Self::calculate_coeff::<B::ArkG1>(
            id,
            delta_g1 * r_hs,
            a_query_first,
            alpha_g1,
            msm_g1(acc_r_g1),
        );

        // In original implementation this is skipped if r==0, however r is shared in our case
        let s_hs: B::ArkScalarField = icicle_to_ark_scalar(T::to_half_share(&s));
        let g1_b = Self::calculate_coeff::<B::ArkG1>(
            id,
            delta_g1 * s_hs,
            b_g1_query_first,
            beta_g1,
            msm_g1(acc_s_g1),
        );

        let g2_b = Self::calculate_coeff::<B::ArkG2>(
            id,
            delta_g2 * s_hs,
            b_g2_query_first,
            beta_g2,
            msm_g2(acc_s_g2),
        );

        // Compute r * s
        let rs: B::ArkScalarField = icicle_to_ark_scalar(T::local_mul::<B>(&r, &s, state));
        let r_s_delta_g1 = delta_g1 * rs;
        tracing::info!(
            "Coefficient assembly took {} ms",
            coeff_timer.elapsed().as_millis()
        );

        let open_timer = std::time::Instant::now();
        // Opening g1_b = B*G1 is safe: B is masked by the fresh uniform s, and its exponent is
        // published in the proof as b = B*G2 anyway. With B*G1 public, r*B*G1 is a local
        // scalar multiplication, so both values can be opened in a single round.
        let (g_a_opened, g1_b_opened) = T::open_two_half_points_g1::<_, B>(g_a, g1_b, net, state)
            .expect("Failed to open g_a and g1_b");

        let mut g_c = g_a_opened * s_hs;
        g_c += g1_b_opened * r_hs;
        g_c -= r_s_delta_g1;
        g_c += msm_g1(l_acc);
        g_c += msm_g1(h_acc);

        let (g_c_opened, g2_b_opened) =
            T::open_two_half_points_g1g2::<_, B>(g_c, g2_b, net, state)?;
        tracing::info!(
            "Point openings took {} ms",
            open_timer.elapsed().as_millis()
        );
        tracing::info!(
            "Proof with assignment took {} ms",
            total_timer.elapsed().as_millis()
        );

        Ok(ark_groth16::Proof {
            a: g_a_opened.into_affine(),
            b: g2_b_opened.into_affine(),
            c: g_c_opened.into_affine(),
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
        let proof = self
            .inner
            .prove::<_, R, co_groth16::mpc::PlainGroth16Driver>(
                &(),
                &mut (),
                matrices,
                public_inputs,
                witness,
            )?;
        Ok(unsafe { cast_proof(proof) })
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

        let proof = self
            .inner
            .prove::<N, R, co_groth16::mpc::Rep3Groth16Driver>(
                net,
                &mut state,
                matrices,
                public_inputs,
                witness,
            )?;
        // SAFETY: the constructors guarantee P == Bn254
        Ok(unsafe { cast_proof(proof) })
    }
}

/// Correlated-randomness pairs a single proof consumes (one per `rand` call).
const SHAMIR_PAIRS_PER_PROOF: usize = 2;

/// A stateful Shamir Groth16 GPU prover for the reduction `R`.
///
/// The constructor allocates all GPU resources (device-resident proving key, scratch
/// buffers, streams), so repeated [`Self::prove`] calls only pay for the
/// witness-dependent uploads and compute -- this part is always safe to rely on, regardless
/// of what the other parties are running.
///
/// By default, [`Self::prove`] also matches upstream exactly: every call runs a fresh
/// Shamir preprocessing (a seed-exchange round trip), the same as
/// `co_groth16::ShamirCoGroth16::prove`, so this prover's network behavior never depends on
/// its own call history and stays compatible with *any* co-party, including one running the
/// plain upstream reference implementation.
///
/// Call [`Self::preprocess`] (or [`Self::enable_persistent_shamir_state`]) to opt into
/// keeping the Shamir MPC state across proofs instead, so only the first proof pays for the
/// seed-exchange round trip. **This changes how many network messages a `prove` call
/// exchanges, which is only safe if every other party in the computation makes the exact
/// same change at the exact same time** -- e.g. if every party is running this same
/// persistent-prover pattern. Enabling it on this party alone while co-parties keep calling
/// a one-shot API (including upstream's `co_groth16::ShamirCoGroth16::prove`) desyncs the
/// network the moment this party's second `prove` call skips a preprocessing round its
/// co-parties still perform: the resulting misaligned messages get deserialized as the wrong
/// type, surfacing as spurious `ark_serialize` "invalid data" errors deep in an unrelated
/// opening call, not as an error at the point of misuse.
///
/// Currently only supports BN254.
pub struct ShamirCoGroth16Prover<P, R = CircomReduction> {
    num_parties: usize,
    threshold: usize,
    /// `Some` only once [`Self::preprocess`]/[`Self::enable_persistent_shamir_state`] has been
    /// called; until then every [`Self::prove`] call preprocesses fresh and discards the
    /// result, matching upstream's one-shot behavior exactly.
    state: Option<ShamirState<ark_bn254::Fr>>,
    /// Set once persistence is requested; see the struct docs for the symmetry requirement
    /// this implies.
    persist_shamir_state: bool,
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
            state: None,
            persist_shamir_state: false,
            inner: CoGroth16Icicle::new(prepared_key, R::requires_eval_c()),
            phantom_data: PhantomData,
        }
    }

    /// Opts into keeping the Shamir MPC state across [`Self::prove`] calls, without
    /// preprocessing anything yet (that happens lazily, on the next `prove`).
    ///
    /// See the struct docs: every other party in the computation must make the same change,
    /// at the same time, or the network desyncs.
    pub fn enable_persistent_shamir_state(&mut self) {
        self.persist_shamir_state = true;
    }

    /// Opts into persistent Shamir state (see [`Self::enable_persistent_shamir_state`]) and
    /// preprocesses enough correlated randomness for `num_proofs` proofs ahead of time, so
    /// the (one-time) seed-exchange round trip happens now rather than on the next `prove`.
    ///
    /// Calling this is optional even once persistence is enabled -- `prove` preprocesses on
    /// demand -- but doing it ahead of time keeps the setup off the critical path. Topping up
    /// the randomness on top of an already-established state needs no communication at all
    /// in the 3-party case, so a prover that is reused never pays for that round trip again.
    pub fn preprocess<N: Network>(&mut self, net: &N, num_proofs: usize) -> Result<()> {
        self.persist_shamir_state = true;
        let pairs = num_proofs.saturating_mul(SHAMIR_PAIRS_PER_PROOF);
        match self.state.as_mut() {
            Some(state) => state.buffer_triples(net, pairs)?,
            None => {
                let preprocessing =
                    ShamirPreprocessing::new(self.num_parties, self.threshold, pairs, net)?;
                self.state = Some(ShamirState::from(preprocessing));
            }
        }
        Ok(())
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

        // Safe default: a fresh preprocessing (and hence seed-exchange round trip) every
        // call, discarded afterwards, exactly matching upstream's one-shot behavior -- see
        // the struct docs for why this must stay the default.
        if !self.persist_shamir_state {
            let preprocessing = ShamirPreprocessing::new(
                self.num_parties,
                self.threshold,
                SHAMIR_PAIRS_PER_PROOF,
                net,
            )?;
            let mut state = ShamirState::from(preprocessing);
            let proof = self
                .inner
                .prove::<N, R, co_groth16::mpc::ShamirGroth16Driver>(
                    net,
                    &mut state,
                    matrices,
                    public_inputs,
                    witness,
                )?;
            // SAFETY: the constructors guarantee P == Bn254
            return Ok(unsafe { cast_proof(proof) });
        }

        // Opted in via `enable_persistent_shamir_state`/`preprocess`: reuse the state built
        // by an earlier proof when there is one, so only the first proof pays for the seed
        // exchange.
        self.preprocess(net, 1)?;
        let state = self
            .state
            .as_mut()
            .expect("preprocess installs the Shamir state");

        let proof = self
            .inner
            .prove::<N, R, co_groth16::mpc::ShamirGroth16Driver>(
                net,
                state,
                matrices,
                public_inputs,
                witness,
            )?;
        // SAFETY: the constructors guarantee P == Bn254
        Ok(unsafe { cast_proof(proof) })
    }
}

#[cfg(test)]
mod tests {
    use ark_ec::VariableBaseMSM;
    use ark_ff::UniformRand;
    use rand::thread_rng;

    /// Sanity check for the load-bearing identity behind merging the public and
    /// private-witness MSMs into one (`create_proof_with_assignment`'s `combined_scalars`):
    /// MSM is linear, so splitting the bases/scalars anywhere and computing two separate
    /// MSMs then summing them must equal one MSM over the concatenation. This doesn't
    /// exercise icicle's actual MSM/precompute machinery (no GPU here), but it does verify
    /// the algebraic claim the optimization depends on, independent of any of this crate's
    /// device code.
    #[test]
    fn msm_is_linear_in_split_point() {
        let mut rng = thread_rng();
        let n_pub = 37;
        let n_priv = 101;

        let scalars = (0..n_pub + n_priv)
            .map(|_| ark_bn254::Fr::rand(&mut rng))
            .collect::<Vec<_>>();
        let bases = (0..n_pub + n_priv)
            .map(|_| ark_bn254::G1Projective::rand(&mut rng).into())
            .collect::<Vec<ark_bn254::G1Affine>>();

        let combined = ark_bn254::G1Projective::msm(&bases, &scalars).unwrap();

        let split = ark_bn254::G1Projective::msm(&bases[..n_pub], &scalars[..n_pub]).unwrap()
            + ark_bn254::G1Projective::msm(&bases[n_pub..], &scalars[n_pub..]).unwrap();

        assert_eq!(combined, split);
    }
}
