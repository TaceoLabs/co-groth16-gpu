//! A Groth16 proof protocol that uses a collaborative MPC protocol to generate the proof.
use crate::gpu_utils::{
    PRECOMPUTE_FACTOR_G1, PRECOMPUTE_FACTOR_G2, from_host_slice, get_first, msm_async,
};
use ark_bn254::Bn254;
use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};
use co_circom_types::SharedWitness;
use co_groth16::ConstraintMatrices;
use eyre::{Context, Result};
use icicle_core::curve::{Affine, Curve, Projective};
use icicle_runtime::memory::DeviceVec;
use mpc_core::MpcState;
use mpc_core::protocols::rep3::conversion::A2BType;
use mpc_core::protocols::rep3::{Rep3PrimeFieldShare, Rep3State};
use mpc_core::protocols::shamir::{ShamirPreprocessing, ShamirPrimeFieldShare, ShamirState};
use mpc_net::Network;
use std::sync::Arc;
use std::{marker::PhantomData, mem::transmute};

use icicle_core::msm::MSM;

use crate::bridges::{
    ArkIcicleBridge, Bn254Bridge, ark_scalars_to_device_into, ark_to_icicle_scalars,
};
use crate::gpu_utils::{Proof, ProofStreams, ProvingKey, VerifyingKey};
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

/// A Groth16 proof protocol that uses a collaborative MPC protocol to generate the proof.
pub struct CoGroth16Icicle<B: ArkIcicleBridge, T: CircomGroth16Prover<B::IcicleScalarField>> {
    phantom_data: PhantomData<(B, T)>,
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
    #[expect(clippy::type_complexity)]
    fn setup<U: co_groth16::CircomGroth16Prover<B::ArkPairing> + 'static, R: R1CSToQAP>(
        id: <T::State as MpcState>::PartyID,
        matrices: &ConstraintMatrices<B::ArkScalarField>,
        private_witness: &[U::ArithmeticShare],
        public_inputs: &[B::ArkScalarField],
        domain_size: usize,
    ) -> eyre::Result<(
        T::DeviceShares,
        T::DeviceShares,
        Option<DeviceVec<B::IcicleScalarField>>,
        DeviceVec<B::IcicleScalarField>,
        T::DeviceShares,
    )> {
        let setup_timer = std::time::Instant::now();

        let eval_timer = std::time::Instant::now();
        let (eval_a, eval_b, eval_c) = T::evaluate_constraints::<B, U>(
            id,
            matrices,
            public_inputs,
            private_witness,
            R::requires_eval_c(),
            domain_size,
        );
        let eval_elapsed = eval_timer.elapsed();

        let witness_timer = std::time::Instant::now();
        let private_witness = T::shares_to_device::<B, U>(private_witness);
        let witness_elapsed = witness_timer.elapsed();

        let public_timer = std::time::Instant::now();
        let public_inputs = ark_to_icicle_scalars(from_host_slice(public_inputs)).unwrap();
        let public_elapsed = public_timer.elapsed();

        tracing::info!(
            "Setup timings: evaluate_constraints={} ms, witness_to_device={} ms, public_to_device={} ms, total={} ms",
            eval_elapsed.as_millis(),
            witness_elapsed.as_millis(),
            public_elapsed.as_millis(),
            setup_timer.elapsed().as_millis()
        );

        Ok((eval_a, eval_b, eval_c, public_inputs, private_witness))
    }

    /// Execute the Groth16 prover using the internal MPC driver.
    /// This version takes the Circom-generated constraint matrices as input and does not re-calculate them.
    /// `scratch` and `streams` are reused across calls when the caller keeps them alive.
    #[expect(clippy::too_many_arguments)]
    fn prove_inner<N: Network, R: R1CSToQAP>(
        net: &N,
        state: &mut T::State,
        eval_a: &mut T::DeviceShares,
        eval_b: &mut T::DeviceShares,
        eval_c: Option<&mut DeviceVec<B::IcicleScalarField>>,
        pkey: &ProvingKey<B::IcicleScalarField, B::IcicleG1, B::IcicleG2>,
        private_witness_half_shares: &DeviceVec<B::IcicleScalarField>,
        public_inputs: &DeviceVec<B::IcicleScalarField>,
        scratch: &mut ReductionScratch<B::IcicleScalarField>,
        streams: &ProofStreams,
    ) -> eyre::Result<Proof<B::IcicleScalarField, B::IcicleG1, B::IcicleG2>> {
        let timer_start = std::time::Instant::now();
        R::witness_map_from_r1cs_eval::<B, T>(
            state,
            eval_a,
            eval_b,
            eval_c,
            public_inputs,
            &pkey.precomputed_roots,
            pkey.num_constraints,
            pkey.domain_size,
            scratch,
        )?;
        tracing::info!(
            "Witness map computation took {} ms",
            timer_start.elapsed().as_millis()
        );

        let (r, s) = (T::rand::<_, B>(net, state)?, T::rand::<_, B>(net, state)?);

        let h = scratch.h.as_ref().expect("witness map left h in scratch");

        Self::create_proof_with_assignment(
            net,
            state,
            pkey,
            r,
            s,
            h,
            public_inputs,
            private_witness_half_shares,
            streams,
        )
    }

    fn calculate_coeff<C>(
        id: <T::State as MpcState>::PartyID,
        initial: Affine<C>,
        first_query: Affine<C>,
        vk_param: Affine<C>,
        pub_acc: Affine<C>,
        priv_acc: Affine<C>,
    ) -> Affine<C>
    where
        C: Curve<ScalarField = B::IcicleScalarField> + MSM<C>,
    {
        let mut res = initial;
        T::add_assign_points_public_hs::<C>(id, &mut res, &first_query);
        T::add_assign_points_public_hs::<C>(id, &mut res, &vk_param);
        T::add_assign_points_public_hs::<C>(id, &mut res, &pub_acc);
        (res.to_projective() + priv_acc.to_projective()).into()
    }

    #[expect(clippy::too_many_arguments)]
    fn create_proof_with_assignment<N: Network>(
        net: &N,
        state: &mut T::State,
        pkey: &ProvingKey<B::IcicleScalarField, B::IcicleG1, B::IcicleG2>,
        r: T::ArithmeticShare,
        s: T::ArithmeticShare,
        h: &DeviceVec<B::IcicleScalarField>,
        input_assignment: &DeviceVec<B::IcicleScalarField>,
        aux_assignment: &DeviceVec<B::IcicleScalarField>,
        streams: &ProofStreams,
    ) -> eyre::Result<Proof<B::IcicleScalarField, B::IcicleG1, B::IcicleG2>> {
        let total_timer = std::time::Instant::now();
        let ProvingKey {
            vk,
            beta_g1,
            delta_g1,
            a_query_first,
            b_g1_query_first,
            b_g2_query_first,
            a_query_pub,
            a_query_priv,
            b_g1_query_pub,
            b_g1_query_priv,
            b_g2_query_pub,
            b_g2_query_priv,
            l_query,
            h_query,
            ..
        } = pkey;

        let VerifyingKey {
            alpha_g1,
            beta_g2,
            delta_g2,
            ..
        } = vk;

        let delta_g1 = delta_g1.to_projective();
        let delta_g2 = delta_g2.to_projective();

        let id = state.id();

        let stream_g1 = &streams.g1;
        let stream_g2 = &streams.g2;

        let msm_timer = std::time::Instant::now();
        // Compute A
        let (pub_acc_r_g1, priv_acc_r_g1) = (
            msm_async(
                a_query_pub,
                &input_assignment[1..],
                stream_g1,
                PRECOMPUTE_FACTOR_G1,
            ),
            msm_async(
                a_query_priv,
                aux_assignment,
                stream_g1,
                PRECOMPUTE_FACTOR_G1,
            ),
        );

        // Compute B in G1
        let (pub_acc_s_g1, priv_acc_s_g1) = (
            msm_async(
                b_g1_query_pub,
                &input_assignment[1..],
                stream_g1,
                PRECOMPUTE_FACTOR_G1,
            ),
            msm_async(
                b_g1_query_priv,
                aux_assignment,
                stream_g1,
                PRECOMPUTE_FACTOR_G1,
            ),
        );

        // Compute B in G2
        let (pub_acc_s_g2, priv_acc_s_g2) = (
            msm_async(
                b_g2_query_pub,
                &input_assignment[1..],
                stream_g2,
                PRECOMPUTE_FACTOR_G2,
            ),
            msm_async(
                b_g2_query_priv,
                aux_assignment,
                stream_g2,
                PRECOMPUTE_FACTOR_G2,
            ),
        );

        // Compute msm(l_query, aux_assignment)
        let l_acc = msm_async(l_query, aux_assignment, stream_g1, PRECOMPUTE_FACTOR_G1);

        // Compute msm(h_query, h)
        let h_acc = msm_async(h_query, h, stream_g1, PRECOMPUTE_FACTOR_G1);

        stream_g1.synchronize().unwrap();
        stream_g2.synchronize().unwrap();
        tracing::info!(
            "MSM + stream sync took {} ms",
            msm_timer.elapsed().as_millis()
        );

        let coeff_timer = std::time::Instant::now();
        let pub_acc_r_g1 = get_first(&pub_acc_r_g1);
        let priv_acc_r_g1 = get_first(&priv_acc_r_g1);
        let pub_acc_s_g1 = get_first(&pub_acc_s_g1);
        let priv_acc_s_g1 = get_first(&priv_acc_s_g1);
        let l_acc = get_first(&l_acc);
        let h_acc = get_first(&h_acc);
        let pub_acc_s_g2 = get_first(&pub_acc_s_g2);
        let priv_acc_s_g2 = get_first(&priv_acc_s_g2);

        let r_hs = T::to_half_share(&r);
        let r_g1 = delta_g1 * r_hs;
        let r_g1 = Self::calculate_coeff::<B::IcicleG1>(
            id,
            r_g1.into(),
            *a_query_first,
            *alpha_g1,
            pub_acc_r_g1.into(),
            priv_acc_r_g1.into(),
        );

        // In original implementation this is skipped if r==0, however r is shared in our case
        let s_hs = T::to_half_share(&s);
        let s_g1 = delta_g1 * s_hs;
        let s_g1 = Self::calculate_coeff::<B::IcicleG1>(
            id,
            s_g1.into(),
            *b_g1_query_first,
            *beta_g1,
            pub_acc_s_g1.into(),
            priv_acc_s_g1.into(),
        );

        let s_g2 = delta_g2 * s_hs;
        let s_g2 = Self::calculate_coeff::<B::IcicleG2>(
            id,
            s_g2.into(),
            *b_g2_query_first,
            *beta_g2,
            pub_acc_s_g2.into(),
            priv_acc_s_g2.into(),
        );

        // Compute r * s
        let rs = T::local_mul::<B>(&r, &s, state);
        let r_s_delta_g1 = delta_g1 * rs;
        tracing::info!(
            "Coefficient assembly took {} ms",
            coeff_timer.elapsed().as_millis()
        );

        let open_timer = std::time::Instant::now();
        let g_a = r_g1;
        let g1_b = s_g1;

        // Opening g1_b = B*G1 is safe: B is masked by the fresh uniform s, and its exponent is
        // published in the proof as b = B*G2 anyway. With B*G1 public, r*B*G1 is a local
        // scalar multiplication, so both values can be opened in a single round.
        let (g_a_opened, g1_b_opened) = T::open_two_half_points_g1::<_, B>(g_a, g1_b, net, state)
            .expect("Failed to open g_a and g1_b");
        let r_g1_b: Projective<<B as ArkIcicleBridge>::IcicleG1> =
            g1_b_opened.to_projective() * r_hs;

        let s_g_a: Projective<<B as ArkIcicleBridge>::IcicleG1> = g_a_opened.to_projective() * s_hs;

        let mut g_c = s_g_a;
        g_c = g_c + r_g1_b;
        g_c = g_c - r_s_delta_g1;
        g_c = g_c + l_acc;
        g_c = g_c + h_acc;

        let g2_b = s_g2;
        let (g_c_opened, g2_b_opened) =
            T::open_two_half_points_g1g2::<_, B>(g_c.into(), g2_b, net, state)?;
        tracing::info!(
            "Point openings took {} ms",
            open_timer.elapsed().as_millis()
        );
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

/// Transmutes Groth16 artifacts from a generic pairing `P` into concrete pairing
///
/// # Safety / Invariant
/// This is only sound if the values you pass in are *actually* built for `$DstPair` / `$DstField`,
/// but are currently being referenced through the generic `P` / `P::ScalarField` types.
/// (I.e. `P == $DstPair` and `P::ScalarField == $DstField` in reality.)
#[macro_export]
macro_rules! transmute_groth16_artifacts {
    (
        src_pairing = $SrcPair:ty,
        dst_pairing = $DstPair:ty,
        dst_field   = $DstField:ty,
        src_arithmetic_share = $SrcArithmeticShare:ty,
        dst_arithmetic_share = $DstArithmeticShare:ty,
        $pkey:expr,
        $matrices:expr,
        $private_witness:expr,
        $public_inputs:expr
    ) => {{
        use core::mem::{size_of, transmute};

        // Optional sanity checks (won't prove correctness, but can catch obvious mismatches)
        debug_assert_eq!(
            size_of::<ark_groth16::ProvingKey<$SrcPair>>(),
            size_of::<ark_groth16::ProvingKey<$DstPair>>(),
        );
        debug_assert_eq!(
            size_of::<Vec<<$SrcPair as ark_ec::pairing::Pairing>::ScalarField>>(),
            size_of::<Vec<$DstField>>(),
        );

        let pkey = $pkey;
        let matrices = $matrices;
        let private_witness = $private_witness;
        let public_inputs = $public_inputs;

        unsafe {
            (
                transmute::<&ark_groth16::ProvingKey<$SrcPair>, &ark_groth16::ProvingKey<$DstPair>>(
                    pkey,
                ),
                transmute::<&Vec<$SrcArithmeticShare>, &Vec<$DstArithmeticShare>>(private_witness),
                transmute::<
                    &ConstraintMatrices<<$SrcPair as ark_ec::pairing::Pairing>::ScalarField>,
                    &ConstraintMatrices<$DstField>,
                >(matrices),
                transmute::<
                    &Vec<<$SrcPair as ark_ec::pairing::Pairing>::ScalarField>,
                    &Vec<$DstField>,
                >(public_inputs),
            )
        }
    }};
}

impl<P: ark_ec::pairing::Pairing> Groth16<P> {
    /// *Locally* create a `Groth16` proof. This is just the [`CoGroth16`] prover
    /// initialized with the [`PlainGroth16Driver`].
    ///
    /// DOES NOT PERFORM ANY MPC. For a plain prover checkout the [Groth16 implementation of arkworks](https://docs.rs/ark-groth16/latest/ark_groth16/).
    pub fn plain_prove<R: R1CSToQAP>(
        pkey: &ark_groth16::ProvingKey<P>,
        prepared_bn_254_key: Option<Arc<Bn254PreparedKey>>,
        matrices: &ConstraintMatrices<P::ScalarField>,
        private_witness: SharedWitness<P::ScalarField, P::ScalarField>,
    ) -> Result<ark_groth16::Proof<P>> {
        let public_inputs = &private_witness.public_inputs;
        let private_witness = &private_witness.witness;

        // TODO CESAR: Duplicate
        let domain = GeneralEvaluationDomain::<P::ScalarField>::new(
            matrices.num_constraints + matrices.num_instance_variables,
        )
        .ok_or(eyre::eyre!("Polynomial Degree too large"))?;
        let domain_size = domain.size();

        if std::any::TypeId::of::<P>() == std::any::TypeId::of::<ark_bn254::Bn254>() {
            let (key, private_witness, matrices, public_inputs) = transmute_groth16_artifacts!(
                src_pairing = P,
                dst_pairing = ark_bn254::Bn254,
                dst_field = ark_bn254::Fr,
                src_arithmetic_share = P::ScalarField,
                dst_arithmetic_share = ark_bn254::Fr,
                pkey,
                matrices,
                private_witness,
                public_inputs
            );

            let prepared_key = prepared_bn_254_key.unwrap_or_else(|| {
                Arc::new(prepare_bn254_key::<R>(
                    key,
                    matrices.num_constraints,
                    matrices.num_instance_variables,
                ))
            });

            let (mut eval_a, mut eval_b, mut eval_c, public_inputs, private_witness) =
                CoGroth16Icicle::<Bn254Bridge, PlainGroth16Driver>::setup::<
                    co_groth16::mpc::PlainGroth16Driver,
                    R,
                >(
                    0, // id irrelevant in the plain case
                    matrices,
                    private_witness,
                    public_inputs,
                    domain_size,
                )?;

            let private_witness_hs = PlainGroth16Driver::to_half_share_vec(private_witness);
            let mut scratch = ReductionScratch::new();
            let streams = ProofStreams::new();
            let icicle_proof =
                CoGroth16Icicle::<Bn254Bridge, PlainGroth16Driver>::prove_inner::<_, R>(
                    &(),
                    &mut (),
                    &mut eval_a,
                    &mut eval_b,
                    eval_c.as_mut(),
                    &prepared_key,
                    &private_witness_hs,
                    &public_inputs,
                    &mut scratch,
                    &streams,
                )?;

            let proof = icicle_proof.to_ark::<Bn254Bridge>();

            let proof = unsafe {
                transmute::<&ark_groth16::Proof<ark_bn254::Bn254>, &ark_groth16::Proof<P>>(&proof)
            };

            Ok(proof.clone())
        // bls12_377/LibSnarkReduction support removed: not needed for our
        // use case, and its h_query/domain_size length mismatch with
        // LibSnarkReduction was causing real problems.
        // } else if std::any::TypeId::of::<P>() == std::any::TypeId::of::<ark_bls12_377::Bls12_377>()
        // {
        //     let (key, private_witness, matrices, public_inputs) = transmute_groth16_artifacts!(
        //         src_pairing = P,
        //         dst_pairing = ark_bls12_377::Bls12_377,
        //         dst_field = ark_bls12_377::Fr,
        //         src_arithmetic_share = P::ScalarField,
        //         dst_arithmetic_share = ark_bls12_377::Fr,
        //         pkey,
        //         matrices,
        //         private_witness,
        //         public_inputs
        //     );

        //     let (mut eval_a, mut eval_b, mut eval_c, public_inputs, private_witness) =
        //         CoGroth16Icicle::<Bls12_377Bridge, PlainGroth16Driver>::setup::<
        //             co_groth16::mpc::PlainGroth16Driver,
        //             R,
        //         >(
        //             0, // id irrelevant in the
        //             matrices,
        //             private_witness,
        //             public_inputs,
        //             domain_size,
        //         )?;

        //     let prepared_key = prepare_bls12_377_key::<R>(
        //         key,
        //         matrices.num_constraints,
        //         matrices.num_instance_variables,
        //     );

        //     let private_witness_hs = PlainGroth16Driver::to_half_share_vec(private_witness);
        //     let mut scratch = ReductionScratch::new();
        //     let streams = ProofStreams::new();
        //     let icicle_proof =
        //         CoGroth16Icicle::<Bls12_377Bridge, PlainGroth16Driver>::prove_inner::<_, R>(
        //             &(),
        //             &mut (),
        //             &mut eval_a,
        //             &mut eval_b,
        //             eval_c.as_mut(),
        //             &prepared_key,
        //             &private_witness_hs,
        //             &public_inputs,
        //             &mut scratch,
        //             &streams,
        //         )?;

        //     let proof = icicle_proof.to_ark::<Bls12_377Bridge>();

        //     let proof = unsafe {
        //         transmute::<&ark_groth16::Proof<ark_bls12_377::Bls12_377>, &ark_groth16::Proof<P>>(
        //             &proof,
        //         )
        //     };

        //     Ok(proof.clone())
        } else {
            panic!("Unsupported pairing")
        }
    }
}

impl<P: ark_ec::pairing::Pairing> Rep3CoGroth16<P> {
    pub fn prove<N: Network, R: R1CSToQAP>(
        net: &N,
        pkey: &ark_groth16::ProvingKey<P>,
        prepared_bn254_key: Option<Arc<Bn254PreparedKey>>,
        matrices: &ConstraintMatrices<P::ScalarField>,
        private_witness: SharedWitness<P::ScalarField, Rep3PrimeFieldShare<P::ScalarField>>,
    ) -> Result<ark_groth16::Proof<P>> {
        let public_inputs = &private_witness.public_inputs;
        let private_witness = &private_witness.witness;

        // TODO CESAR: Duplicate
        let domain = GeneralEvaluationDomain::<P::ScalarField>::new(
            matrices.num_constraints + matrices.num_instance_variables,
        )
        .ok_or(eyre::eyre!("Polynomial Degree too large"))?;
        let domain_size = domain.size();

        let mut state = Rep3State::new(net, A2BType::default())?;

        if std::any::TypeId::of::<P>() == std::any::TypeId::of::<ark_bn254::Bn254>() {
            let (key, private_witness, matrices, public_inputs) = transmute_groth16_artifacts!(
                src_pairing = P,
                dst_pairing = ark_bn254::Bn254,
                dst_field = ark_bn254::Fr,
                src_arithmetic_share = Rep3PrimeFieldShare<P::ScalarField>,
                dst_arithmetic_share = Rep3PrimeFieldShare<ark_bn254::Fr>,
                pkey,
                matrices,
                private_witness,
                public_inputs
            );

            let (mut eval_a, mut eval_b, mut eval_c, public_inputs, private_witness) =
                CoGroth16Icicle::<Bn254Bridge, Rep3Groth16Driver>::setup::<
                    co_groth16::mpc::Rep3Groth16Driver,
                    R,
                >(
                    state.id(),
                    matrices,
                    private_witness,
                    public_inputs,
                    domain_size,
                )?;

            let prepared_key = prepared_bn254_key.unwrap_or_else(|| {
                Arc::new(prepare_bn254_key::<R>(
                    key,
                    matrices.num_constraints,
                    matrices.num_instance_variables,
                ))
            });
            let private_witness_hs = Rep3Groth16Driver::to_half_share_vec(private_witness);
            let mut scratch = ReductionScratch::new();
            let streams = ProofStreams::new();
            let icicle_proof =
                CoGroth16Icicle::<Bn254Bridge, Rep3Groth16Driver>::prove_inner::<N, R>(
                    net,
                    &mut state,
                    &mut eval_a,
                    &mut eval_b,
                    eval_c.as_mut(),
                    &prepared_key,
                    &private_witness_hs,
                    &public_inputs,
                    &mut scratch,
                    &streams,
                )?;

            let proof = icicle_proof.to_ark::<Bn254Bridge>();

            let proof = unsafe {
                transmute::<&ark_groth16::Proof<ark_bn254::Bn254>, &ark_groth16::Proof<P>>(&proof)
            };

            Ok(proof.clone())
        } else {
            panic!("Unsupported pairing")
        }
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
            Some(prepared_key) => {
                ShamirCoGroth16Prover::from_prepared_key(num_parties, threshold, prepared_key)
            }
            None => ShamirCoGroth16Prover::new::<R>(num_parties, threshold, pkey, matrices),
        };
        prover.prove::<N, R>(net, matrices, private_witness)
    }
}

type IcicleFr = <Bn254Bridge as ArkIcicleBridge>::IcicleScalarField;

/// Device buffers reused across proof runs. Their contents are witness-dependent and are
/// re-uploaded on every run; only the allocations (and the streams) are kept alive.
struct ShamirGpuCache {
    eval_a: Option<DeviceVec<IcicleFr>>,
    eval_b: Option<DeviceVec<IcicleFr>>,
    eval_c: Option<DeviceVec<IcicleFr>>,
    witness: Option<DeviceVec<IcicleFr>>,
    public_inputs: Option<DeviceVec<IcicleFr>>,
    reduction: reduction::ReductionScratch<IcicleFr>,
    streams: ProofStreams,
}

impl ShamirGpuCache {
    fn new() -> Self {
        Self {
            eval_a: None,
            eval_b: None,
            eval_c: None,
            witness: None,
            public_inputs: None,
            reduction: ReductionScratch::new(),
            streams: ProofStreams::new(),
        }
    }
}

/// A stateful Shamir Groth16 GPU prover.
///
/// Keeps the device-resident proving key (MSM bases incl. precomputation, roots) and all
/// scratch device buffers/streams alive across [`Self::prove`] calls, so repeated proofs
/// for the same circuit only pay for the witness-dependent uploads and compute.
///
/// Currently only supports BN254.
pub struct ShamirCoGroth16Prover<P> {
    num_parties: usize,
    threshold: usize,
    prepared_key: Arc<Bn254PreparedKey>,
    cache: ShamirGpuCache,
    phantom_data: PhantomData<P>,
}

impl<P: ark_ec::pairing::Pairing> ShamirCoGroth16Prover<P> {
    /// Creates a prover from an already device-prepared proving key.
    pub fn from_prepared_key(
        num_parties: usize,
        threshold: usize,
        prepared_key: Arc<Bn254PreparedKey>,
    ) -> Self {
        Self {
            num_parties,
            threshold,
            prepared_key,
            cache: ShamirGpuCache::new(),
            phantom_data: PhantomData,
        }
    }

    /// Prepares the proving key on the device and creates a prover.
    /// `R` must match the reduction later passed to [`Self::prove`].
    pub fn new<R: R1CSToQAP>(
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
    pub fn prove<N: Network, R: R1CSToQAP>(
        &mut self,
        net: &N,
        matrices: &ConstraintMatrices<P::ScalarField>,
        private_witness: SharedWitness<P::ScalarField, ShamirPrimeFieldShare<P::ScalarField>>,
    ) -> Result<ark_groth16::Proof<P>> {
        if std::any::TypeId::of::<P>() != std::any::TypeId::of::<ark_bn254::Bn254>() {
            panic!("Unsupported pairing");
        }
        // SAFETY: P == Bn254, checked above; ShamirPrimeFieldShare is repr(transparent)
        // over its field element.
        let (matrices, witness, public_inputs) = unsafe {
            (
                transmute::<&ConstraintMatrices<P::ScalarField>, &ConstraintMatrices<ark_bn254::Fr>>(
                    matrices,
                ),
                transmute::<
                    &[ShamirPrimeFieldShare<P::ScalarField>],
                    &[ShamirPrimeFieldShare<ark_bn254::Fr>],
                >(private_witness.witness.as_slice()),
                transmute::<&[P::ScalarField], &[ark_bn254::Fr]>(
                    private_witness.public_inputs.as_slice(),
                ),
            )
        };

        // we need 2 corr rand pairs for the two rand calls
        let num_pairs = 2;
        let preprocessing =
            ShamirPreprocessing::new(self.num_parties, self.threshold, num_pairs, net)?;
        let mut state = ShamirState::from(preprocessing);
        let id = state.id();
        let domain_size = self.prepared_key.domain_size;

        let setup_timer = std::time::Instant::now();
        let cache = &mut self.cache;

        type CpuDriver = co_groth16::mpc::ShamirGroth16Driver;
        // SAFETY: ShamirPrimeFieldShare is repr(transparent) over its field element
        let eval_a = evaluate_constraint::<ark_bn254::Bn254, CpuDriver>(
            id,
            domain_size,
            &matrices.a,
            public_inputs,
            witness,
        );
        let eval_a = unsafe {
            transmute::<&[ShamirPrimeFieldShare<ark_bn254::Fr>], &[ark_bn254::Fr]>(
                eval_a.as_slice(),
            )
        };
        ark_scalars_to_device_into(eval_a, &mut cache.eval_a);

        let eval_b = evaluate_constraint::<ark_bn254::Bn254, CpuDriver>(
            id,
            domain_size,
            &matrices.b,
            public_inputs,
            witness,
        );
        // SAFETY: as above
        let eval_b = unsafe {
            transmute::<&[ShamirPrimeFieldShare<ark_bn254::Fr>], &[ark_bn254::Fr]>(
                eval_b.as_slice(),
            )
        };
        ark_scalars_to_device_into(eval_b, &mut cache.eval_b);

        if R::requires_eval_c() {
            let eval_c = evaluate_constraint_half_share::<ark_bn254::Bn254, CpuDriver>(
                id,
                domain_size,
                &matrices.c,
                public_inputs,
                witness,
            );
            ark_scalars_to_device_into(&eval_c, &mut cache.eval_c);
        } else {
            cache.eval_c = None;
        }

        // SAFETY: as above
        let witness = unsafe {
            transmute::<&[ShamirPrimeFieldShare<ark_bn254::Fr>], &[ark_bn254::Fr]>(witness)
        };
        ark_scalars_to_device_into(witness, &mut cache.witness);
        ark_scalars_to_device_into(public_inputs, &mut cache.public_inputs);

        tracing::info!(
            "Setup (constraint eval + device upload) took {} ms",
            setup_timer.elapsed().as_millis()
        );

        let icicle_proof =
            CoGroth16Icicle::<Bn254Bridge, ShamirGroth16Driver<ark_bn254::Fr>>::prove_inner::<N, R>(
                net,
                &mut state,
                cache.eval_a.as_mut().expect("uploaded above"),
                cache.eval_b.as_mut().expect("uploaded above"),
                cache.eval_c.as_mut(),
                &self.prepared_key,
                // A degree-t Shamir share is already a valid half share
                cache.witness.as_ref().expect("uploaded above"),
                cache.public_inputs.as_ref().expect("uploaded above"),
                &mut cache.reduction,
                &cache.streams,
            )?;

        let proof = icicle_proof.to_ark::<Bn254Bridge>();

        // SAFETY: P == Bn254, checked above
        let proof = unsafe {
            transmute::<&ark_groth16::Proof<ark_bn254::Bn254>, &ark_groth16::Proof<P>>(&proof)
        };

        Ok(proof.clone())
    }
}
