//! A Groth16 proof protocol that uses a collaborative MPC protocol to generate the proof.
use crate::gpu_utils::{
    PRECOMPUTE_FACTOR_G1, PRECOMPUTE_FACTOR_G2, from_host_slice, get_first, msm_async,
};
use ark_bn254::Bn254;
use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};
use ark_relations::r1cs::ConstraintMatrices;
use co_circom_types::SharedWitness;
use eyre::Result;
use icicle_core::curve::{Affine, Curve, Projective};
use icicle_runtime::memory::{DeviceVec, HostOrDeviceSlice};
use mpc_core::MpcState;
use mpc_core::protocols::rep3::conversion::A2BType;
use mpc_core::protocols::rep3::{Rep3PrimeFieldShare, Rep3State};
use mpc_net::Network;
use std::sync::Arc;
use std::{marker::PhantomData, mem::transmute};

use icicle_core::msm::MSM;

use crate::bridges::{ArkIcicleBridge, Bls12_377Bridge, Bn254Bridge, ark_to_icicle_scalars};
use crate::gpu_utils::{Proof, ProvingKey, VerifyingKey};
use crate::mpc::CircomGroth16Prover;
use crate::mpc::plain::PlainGroth16Driver;
use crate::mpc::rep3::Rep3Groth16Driver;

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

/// A Groth16 proof protocol that uses a collaborative MPC protocol to generate the proof.
pub struct CoGroth16Icicle<B: ArkIcicleBridge, T: CircomGroth16Prover<B::IcicleScalarField>> {
    phantom_data: PhantomData<(B, T)>,
}

pub type Bn254PreparedKey = ProvingKey<
    <Bn254Bridge as ArkIcicleBridge>::IcicleScalarField,
    <Bn254Bridge as ArkIcicleBridge>::IcicleG1,
    <Bn254Bridge as ArkIcicleBridge>::IcicleG2,
>;

type Bls12_377PreparedKey = ProvingKey<
    <Bls12_377Bridge as ArkIcicleBridge>::IcicleScalarField,
    <Bls12_377Bridge as ArkIcicleBridge>::IcicleG1,
    <Bls12_377Bridge as ArkIcicleBridge>::IcicleG2,
>;

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

pub fn prepare_bls12_377_key<R: R1CSToQAP>(
    pkey: &ark_groth16::ProvingKey<ark_bls12_377::Bls12_377>,
    num_constraints: usize,
    num_instance_variables: usize,
) -> Bls12_377PreparedKey {
    ProvingKey::from_ark(
        pkey,
        num_constraints,
        num_instance_variables,
        R::requires_eval_c(),
    )
}

impl<B: ArkIcicleBridge, T: CircomGroth16Prover<B::IcicleScalarField>> CoGroth16Icicle<B, T> {
    fn setup<U: co_groth16::CircomGroth16Prover<B::ArkPairing> + 'static, R: R1CSToQAP>(
        id: <T::State as MpcState>::PartyID,
        matrices: &ConstraintMatrices<B::ArkScalarField>,
        private_witness: &Vec<U::ArithmeticShare>,
        public_inputs: &Vec<B::ArkScalarField>,
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

        println!(
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
    fn prove_inner<N: Network, R: R1CSToQAP>(
        net0: &N,
        net1: &N,
        state0: &mut T::State,
        state1: &mut T::State,
        eval_a: &mut T::DeviceShares,
        eval_b: &mut T::DeviceShares,
        eval_c: Option<&mut DeviceVec<B::IcicleScalarField>>,
        pkey: &ProvingKey<B::IcicleScalarField, B::IcicleG1, B::IcicleG2>,
        private_witness: T::DeviceShares,
        public_inputs: &DeviceVec<B::IcicleScalarField>,
    ) -> eyre::Result<Proof<B::IcicleScalarField, B::IcicleG1, B::IcicleG2>> {
        let timer_start = std::time::Instant::now();
        let h = R::witness_map_from_r1cs_eval::<B, T>(
            state0,
            eval_a,
            eval_b,
            eval_c,
            public_inputs,
            &pkey.precomputed_roots,
            pkey.num_constraints,
            pkey.domain_size,
        )?;
        println!(
            "Witness map computation took {} ms",
            timer_start.elapsed().as_millis()
        );

        let (r, s) = (
            T::rand::<_, B>(net0, state0)?,
            T::rand::<_, B>(net0, state0)?,
        );

        let private_witness_half_shares = T::to_half_share_vec(&private_witness);

        Self::create_proof_with_assignment(
            net0,
            net1,
            state0,
            state1,
            pkey,
            r,
            s,
            h,
            &public_inputs,
            &private_witness_half_shares,
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
        net0: &N,
        net1: &N,
        state0: &mut T::State,
        state1: &mut T::State,
        pkey: &ProvingKey<B::IcicleScalarField, B::IcicleG1, B::IcicleG2>,
        r: T::ArithmeticShare,
        s: T::ArithmeticShare,
        h: DeviceVec<B::IcicleScalarField>,
        input_assignment: &DeviceVec<B::IcicleScalarField>,
        aux_assignment: &DeviceVec<B::IcicleScalarField>,
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
            proof_streams,
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

        let id = state0.id();

        let stream_g1 = &proof_streams.g1;
        let stream_g2 = &proof_streams.g2;

        println!(
            "msm lens: pub={} aux={} | a_pub={} a_priv={} | b1_pub={} b1_priv={} | b2_pub={} b2_priv={} | l_query={} h_query={} h={}",
            input_assignment.len().saturating_sub(1),
            aux_assignment.len(),
            a_query_pub.len(),
            a_query_priv.len(),
            b_g1_query_pub.len(),
            b_g1_query_priv.len(),
            b_g2_query_pub.len(),
            b_g2_query_priv.len(),
            l_query.len(),
            h_query.len(),
            h.len(),
        );

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
        let h_acc = msm_async(h_query, &h, stream_g1, PRECOMPUTE_FACTOR_G1);

        stream_g1.synchronize().unwrap();
        stream_g2.synchronize().unwrap();
        println!(
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
        let rs = T::local_mul::<B>(&r, &s, state0);
        let r_s_delta_g1 = delta_g1 * rs;
        println!(
            "Coefficient assembly took {} ms",
            coeff_timer.elapsed().as_millis()
        );

        let open_timer = std::time::Instant::now();
        let g_a = r_g1;
        let g1_b = s_g1;

        let (g_a_opened, r_g1_b) = rayon::join(
            || T::open_half_point_g1::<_, B>(g_a, net0, state0).expect("Failed to open g_a"),
            || {
                T::scalar_mul_g1::<_, B>(&g1_b, r, net1, state1)
                    .expect("Failed to scalar mul g1_b with r")
            },
        );

        let s_g_a: Projective<<B as ArkIcicleBridge>::IcicleG1> = g_a_opened.to_projective() * s_hs;

        let mut g_c = s_g_a;
        g_c = g_c + r_g1_b.to_projective();
        g_c = g_c - r_s_delta_g1;
        g_c = g_c + l_acc;
        g_c = g_c + h_acc;

        let g2_b = s_g2;
        let (g_c_opened, g2_b_opened) = rayon::join(
            || T::open_half_point_g1::<_, B>(g_c.into(), net0, state0),
            || T::open_half_point_g2::<_, B>(g2_b, net1, state1),
        );
        println!(
            "Point openings took {} ms",
            open_timer.elapsed().as_millis()
        );
        println!(
            "Proof with assignment took {} ms",
            total_timer.elapsed().as_millis()
        );

        Ok(Proof {
            a: g_a_opened,
            b: g2_b_opened?,
            c: g_c_opened?,
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

        unsafe {
            (
                transmute::<&ark_groth16::ProvingKey<$SrcPair>, &ark_groth16::ProvingKey<$DstPair>>(
                    $pkey,
                ),
                transmute::<&Vec<$SrcArithmeticShare>, &Vec<$DstArithmeticShare>>($private_witness),
                transmute::<
                    &ConstraintMatrices<<$SrcPair as ark_ec::pairing::Pairing>::ScalarField>,
                    &ConstraintMatrices<$DstField>,
                >($matrices),
                transmute::<
                    &Vec<<$SrcPair as ark_ec::pairing::Pairing>::ScalarField>,
                    &Vec<$DstField>,
                >($public_inputs),
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

            let icicle_proof =
                CoGroth16Icicle::<Bn254Bridge, PlainGroth16Driver>::prove_inner::<_, R>(
                    &(),
                    &(),
                    &mut (),
                    &mut (),
                    &mut eval_a,
                    &mut eval_b,
                    eval_c.as_mut(),
                    &prepared_key,
                    private_witness,
                    &public_inputs,
                )?;

            let proof = icicle_proof.to_ark::<Bn254Bridge>();

            let proof = unsafe {
                transmute::<&ark_groth16::Proof<ark_bn254::Bn254>, &ark_groth16::Proof<P>>(&proof)
            };

            return Ok(proof.clone());
        } else if std::any::TypeId::of::<P>() == std::any::TypeId::of::<ark_bls12_377::Bls12_377>()
        {
            let (key, private_witness, matrices, public_inputs) = transmute_groth16_artifacts!(
                src_pairing = P,
                dst_pairing = ark_bls12_377::Bls12_377,
                dst_field = ark_bls12_377::Fr,
                src_arithmetic_share = P::ScalarField,
                dst_arithmetic_share = ark_bls12_377::Fr,
                pkey,
                matrices,
                private_witness,
                public_inputs
            );

            let (mut eval_a, mut eval_b, mut eval_c, public_inputs, private_witness) =
                CoGroth16Icicle::<Bls12_377Bridge, PlainGroth16Driver>::setup::<
                    co_groth16::mpc::PlainGroth16Driver,
                    R,
                >(
                    0, // id irrelevant in the
                    matrices,
                    private_witness,
                    public_inputs,
                    domain_size,
                )?;

            let prepared_key = prepare_bls12_377_key::<R>(
                key,
                matrices.num_constraints,
                matrices.num_instance_variables,
            );

            let icicle_proof =
                CoGroth16Icicle::<Bls12_377Bridge, PlainGroth16Driver>::prove_inner::<_, R>(
                    &(),
                    &(),
                    &mut (),
                    &mut (),
                    &mut eval_a,
                    &mut eval_b,
                    eval_c.as_mut(),
                    &prepared_key,
                    private_witness,
                    &public_inputs,
                )?;

            let proof = icicle_proof.to_ark::<Bls12_377Bridge>();

            let proof = unsafe {
                transmute::<&ark_groth16::Proof<ark_bls12_377::Bls12_377>, &ark_groth16::Proof<P>>(
                    &proof,
                )
            };

            return Ok(proof.clone());
        } else {
            panic!("Unsupported pairing")
        }
    }
}

impl<P: ark_ec::pairing::Pairing> Rep3CoGroth16<P> {
    pub fn prove<N: Network, R: R1CSToQAP>(
        net0: &N,
        net1: &N,
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

        // we need 3 number of corr rand pairs. 2 for two rand calls, 1 for scalar_mul
        let mut state0 = Rep3State::new(net0, A2BType::default())?;
        let mut state1 = state0.fork(0)?;

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
                    state0.id(),
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
            let icicle_proof =
                CoGroth16Icicle::<Bn254Bridge, Rep3Groth16Driver>::prove_inner::<N, R>(
                    &net0,
                    &net1,
                    &mut state0,
                    &mut state1,
                    &mut eval_a,
                    &mut eval_b,
                    eval_c.as_mut(),
                    &prepared_key,
                    private_witness,
                    &public_inputs,
                )?;

            let proof = icicle_proof.to_ark::<Bn254Bridge>();

            let proof = unsafe {
                transmute::<&ark_groth16::Proof<ark_bn254::Bn254>, &ark_groth16::Proof<P>>(&proof)
            };

            return Ok(proof.clone());
        } else {
            panic!("Unsupported pairing")
        };
    }
}
