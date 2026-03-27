//! A Groth16 proof protocol that uses a collaborative MPC protocol to generate the proof.
use crate::gpu_utils::{from_host_slice, initialize_domain, msm_async};
use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};
use ark_relations::r1cs::ConstraintMatrices;
use co_circom_types::SharedWitness;
use eyre::Result;
use icicle_core::curve::{Affine, Curve, Projective};
use icicle_runtime::memory::{DeviceVec, HostOrDeviceSlice, HostSlice};
use icicle_runtime::stream::IcicleStream;
use mpc_core::MpcState;
use mpc_core::protocols::rep3::conversion::A2BType;
use mpc_core::protocols::rep3::{Rep3PrimeFieldShare, Rep3State};
use mpc_net::Network;
use std::{cell::RefCell, collections::HashMap, marker::PhantomData, mem::transmute, rc::Rc};

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

type Bn254PreparedKey = ProvingKey<
    <Bn254Bridge as ArkIcicleBridge>::IcicleScalarField,
    <Bn254Bridge as ArkIcicleBridge>::IcicleG1,
    <Bn254Bridge as ArkIcicleBridge>::IcicleG2,
>;

type Bls12_377PreparedKey = ProvingKey<
    <Bls12_377Bridge as ArkIcicleBridge>::IcicleScalarField,
    <Bls12_377Bridge as ArkIcicleBridge>::IcicleG1,
    <Bls12_377Bridge as ArkIcicleBridge>::IcicleG2,
>;

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct PreparedKeyCacheId {
    pkey_addr: usize,
    num_constraints: usize,
    num_instance_variables: usize,
    eval_c: bool,
    a_query_len: usize,
    b_g1_query_len: usize,
    b_g2_query_len: usize,
    h_query_len: usize,
    l_query_len: usize,
}

fn make_prepared_key_cache_id<P: ark_ec::pairing::Pairing>(
    pkey: &ark_groth16::ProvingKey<P>,
    num_constraints: usize,
    num_instance_variables: usize,
    eval_c: bool,
) -> PreparedKeyCacheId {
    PreparedKeyCacheId {
        pkey_addr: (pkey as *const _) as usize,
        num_constraints,
        num_instance_variables,
        eval_c,
        a_query_len: pkey.a_query.len(),
        b_g1_query_len: pkey.b_g1_query.len(),
        b_g2_query_len: pkey.b_g2_query.len(),
        h_query_len: pkey.h_query.len(),
        l_query_len: pkey.l_query.len(),
    }
}

thread_local! {
    static BN254_PREPARED_KEY_CACHE: RefCell<HashMap<PreparedKeyCacheId, Rc<Bn254PreparedKey>>> =
        RefCell::new(HashMap::new());
    static BLS12_377_PREPARED_KEY_CACHE: RefCell<HashMap<PreparedKeyCacheId, Rc<Bls12_377PreparedKey>>> =
        RefCell::new(HashMap::new());
    static PROOF_STREAMS: RefCell<ProofStreams> = RefCell::new(ProofStreams::new());
}

struct ProofStreams {
    g1: IcicleStream,
    g2: IcicleStream,
}

impl ProofStreams {
    fn new() -> Self {
        Self {
            g1: IcicleStream::create().unwrap(),
            g2: IcicleStream::create().unwrap(),
        }
    }
}

impl Drop for ProofStreams {
    fn drop(&mut self) {
        let _ = self.g1.destroy();
        let _ = self.g2.destroy();
    }
}

pub fn get_or_prepare_bn254_key<R: R1CSToQAP>(
    pkey: &ark_groth16::ProvingKey<ark_bn254::Bn254>,
    num_constraints: usize,
    num_instance_variables: usize,
) -> Rc<Bn254PreparedKey> {
    let cache_id = make_prepared_key_cache_id(pkey, num_constraints, num_instance_variables, R::requires_eval_c());
    BN254_PREPARED_KEY_CACHE.with(|cache| {
        if let Some(prepared) = cache.borrow().get(&cache_id).cloned() {
            return prepared;
        }

        let prepared = Rc::new(ProvingKey::from_ark(
            pkey,
            num_constraints,
            num_instance_variables,
            R::requires_eval_c(),
        ));
        cache.borrow_mut().insert(cache_id, Rc::clone(&prepared));
        prepared
    })
}

pub fn get_or_prepare_bls12_377_key<R: R1CSToQAP>(
    pkey: &ark_groth16::ProvingKey<ark_bls12_377::Bls12_377>,
    num_constraints: usize,
    num_instance_variables: usize,
) -> Rc<Bls12_377PreparedKey> {
    let cache_id = make_prepared_key_cache_id(pkey, num_constraints, num_instance_variables, R::requires_eval_c());
    BLS12_377_PREPARED_KEY_CACHE.with(|cache| {
        if let Some(prepared) = cache.borrow().get(&cache_id).cloned() {
            return prepared;
        }

        let prepared = Rc::new(ProvingKey::from_ark(
            pkey,
            num_constraints,
            num_instance_variables,
            R::requires_eval_c(),
        ));
        cache.borrow_mut().insert(cache_id, Rc::clone(&prepared));
        prepared
    })
}

impl<B: ArkIcicleBridge, T: CircomGroth16Prover<B::IcicleScalarField>> CoGroth16Icicle<B, T> {
    fn setup<U: co_groth16::CircomGroth16Prover<B::ArkPairing> + 'static, R: R1CSToQAP>(
        id: <T::State as MpcState>::PartyID,
        prepared_key: Rc<ProvingKey<B::IcicleScalarField, B::IcicleG1, B::IcicleG2>>,
        matrices: &ConstraintMatrices<B::ArkScalarField>,
        private_witness: &Vec<U::ArithmeticShare>,
        public_inputs: &Vec<B::ArkScalarField>,
        domain_size: usize,
    ) -> eyre::Result<(
        Rc<ProvingKey<B::IcicleScalarField, B::IcicleG1, B::IcicleG2>>,
        T::DeviceShares,
        T::DeviceShares,
        Option<DeviceVec<B::IcicleScalarField>>,
        DeviceVec<B::IcicleScalarField>,
        T::DeviceShares,
    )>
    where
        B::IcicleScalarField: 'static,
    {
        let (eval_a, eval_b, eval_c) = T::evaluate_constraints::<B, U>(
            id,
            matrices,
            public_inputs,
            private_witness,
            R::requires_eval_c(),
            domain_size,
        );

        initialize_domain::<B::IcicleScalarField>(domain_size);

        let private_witness = T::shares_to_device::<B, U>(private_witness);

        let public_inputs = ark_to_icicle_scalars(from_host_slice(public_inputs)).unwrap();

        Ok((
            prepared_key,
            eval_a,
            eval_b,
            eval_c,
            public_inputs,
            private_witness,
        ))
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

        let timer_start = std::time::Instant::now();
        let out = Self::create_proof_with_assignment(
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
        );
        println!(
            "Proof with assignment took {} ms",
            timer_start.elapsed().as_millis()
        );
        out
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
            msm_precompute_factor,
            msm_c,
            msm_large_bucket_factor,
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

        let precompute_factor = (*msm_precompute_factor) as usize;
        debug_assert_eq!(
            a_query_pub.len() / precompute_factor,
            input_assignment.len() - 1
        );
        debug_assert_eq!(a_query_priv.len() / precompute_factor, aux_assignment.len());
        debug_assert_eq!(
            b_g1_query_pub.len() / precompute_factor,
            input_assignment.len() - 1
        );
        debug_assert_eq!(
            b_g1_query_priv.len() / precompute_factor,
            aux_assignment.len()
        );
        debug_assert_eq!(
            b_g2_query_pub.len() / precompute_factor,
            input_assignment.len() - 1
        );
        debug_assert_eq!(
            b_g2_query_priv.len() / precompute_factor,
            aux_assignment.len()
        );
        debug_assert_eq!(l_query.len() / precompute_factor, aux_assignment.len());
        debug_assert_eq!(h_query.len() / precompute_factor, h.len());

        let (
            pub_acc_r_g1,
            priv_acc_r_g1,
            pub_acc_s_g1,
            priv_acc_s_g1,
            pub_acc_s_g2,
            priv_acc_s_g2,
            l_acc,
            h_acc,
        ) = PROOF_STREAMS.with(|streams| {
            let mut streams = streams.borrow_mut();
            let ProofStreams {
                g1: stream_g1,
                g2: stream_g2,
            } = &mut *streams;

            // Compute A
            let (pub_acc_r_g1, priv_acc_r_g1) = (
                msm_async(
                    a_query_pub,
                    &input_assignment[1..],
                    *msm_precompute_factor,
                    *msm_c,
                    *msm_large_bucket_factor,
                    stream_g1,
                ),
                msm_async(
                    a_query_priv,
                    aux_assignment,
                    *msm_precompute_factor,
                    *msm_c,
                    *msm_large_bucket_factor,
                    stream_g1,
                ),
            );

            // Compute B in G1
            let (pub_acc_s_g1, priv_acc_s_g1) = (
                msm_async(
                    b_g1_query_pub,
                    &input_assignment[1..],
                    *msm_precompute_factor,
                    *msm_c,
                    *msm_large_bucket_factor,
                    stream_g1,
                ),
                msm_async(
                    b_g1_query_priv,
                    aux_assignment,
                    *msm_precompute_factor,
                    *msm_c,
                    *msm_large_bucket_factor,
                    stream_g1,
                ),
            );

            // Compute B in G2
            let (pub_acc_s_g2, priv_acc_s_g2) = (
                msm_async(
                    b_g2_query_pub,
                    &input_assignment[1..],
                    *msm_precompute_factor,
                    *msm_c,
                    *msm_large_bucket_factor,
                    stream_g2,
                ),
                msm_async(
                    b_g2_query_priv,
                    aux_assignment,
                    *msm_precompute_factor,
                    *msm_c,
                    *msm_large_bucket_factor,
                    stream_g2,
                ),
            );

            // Compute msm(l_query, aux_assignment)
            let l_acc = msm_async(
                l_query,
                aux_assignment,
                *msm_precompute_factor,
                *msm_c,
                *msm_large_bucket_factor,
                stream_g1,
            );

            // Compute msm(h_query, h)
            let h_acc = msm_async(
                h_query,
                &h,
                *msm_precompute_factor,
                *msm_c,
                *msm_large_bucket_factor,
                stream_g1,
            );

            let mut pub_acc_r_g1_h: [Projective<B::IcicleG1>; 1] =
                std::array::from_fn(|_| Projective::<B::IcicleG1>::zero());
            let mut priv_acc_r_g1_h: [Projective<B::IcicleG1>; 1] =
                std::array::from_fn(|_| Projective::<B::IcicleG1>::zero());
            let mut pub_acc_s_g1_h: [Projective<B::IcicleG1>; 1] =
                std::array::from_fn(|_| Projective::<B::IcicleG1>::zero());
            let mut priv_acc_s_g1_h: [Projective<B::IcicleG1>; 1] =
                std::array::from_fn(|_| Projective::<B::IcicleG1>::zero());
            let mut pub_acc_s_g2_h: [Projective<B::IcicleG2>; 1] =
                std::array::from_fn(|_| Projective::<B::IcicleG2>::zero());
            let mut priv_acc_s_g2_h: [Projective<B::IcicleG2>; 1] =
                std::array::from_fn(|_| Projective::<B::IcicleG2>::zero());
            let mut l_acc_h: [Projective<B::IcicleG1>; 1] =
                std::array::from_fn(|_| Projective::<B::IcicleG1>::zero());
            let mut h_acc_h: [Projective<B::IcicleG1>; 1] =
                std::array::from_fn(|_| Projective::<B::IcicleG1>::zero());

            pub_acc_r_g1
                .copy_to_host_async(HostSlice::from_mut_slice(&mut pub_acc_r_g1_h), stream_g1)
                .unwrap();
            priv_acc_r_g1
                .copy_to_host_async(HostSlice::from_mut_slice(&mut priv_acc_r_g1_h), stream_g1)
                .unwrap();
            pub_acc_s_g1
                .copy_to_host_async(HostSlice::from_mut_slice(&mut pub_acc_s_g1_h), stream_g1)
                .unwrap();
            priv_acc_s_g1
                .copy_to_host_async(HostSlice::from_mut_slice(&mut priv_acc_s_g1_h), stream_g1)
                .unwrap();
            l_acc
                .copy_to_host_async(HostSlice::from_mut_slice(&mut l_acc_h), stream_g1)
                .unwrap();
            h_acc
                .copy_to_host_async(HostSlice::from_mut_slice(&mut h_acc_h), stream_g1)
                .unwrap();

            pub_acc_s_g2
                .copy_to_host_async(HostSlice::from_mut_slice(&mut pub_acc_s_g2_h), stream_g2)
                .unwrap();
            priv_acc_s_g2
                .copy_to_host_async(HostSlice::from_mut_slice(&mut priv_acc_s_g2_h), stream_g2)
                .unwrap();

            stream_g1.synchronize().unwrap();
            stream_g2.synchronize().unwrap();

            (
                pub_acc_r_g1_h[0],
                priv_acc_r_g1_h[0],
                pub_acc_s_g1_h[0],
                priv_acc_s_g1_h[0],
                pub_acc_s_g2_h[0],
                priv_acc_s_g2_h[0],
                l_acc_h[0],
                h_acc_h[0],
            )
        });

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

        let g_a = r_g1;
        let g1_b = s_g1;

        let (g_a_opened, r_g1_b) = rayon::join(
            || T::open_half_point_g1::<_, B>(g_a, net0, state0).expect("Failed to open g_a"),
            || {
                T::scalar_mul_g1::<_, B>(&g1_b, r, net1, state1)
                    .expect("Failed to scalar mul g1_b with r")
            },
        );

        let s_g_a = g_a_opened.to_projective() * s_hs;

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

            let prepared_key = get_or_prepare_bn254_key::<R>(key, matrices.num_constraints, matrices.num_instance_variables);
            let (key, mut eval_a, mut eval_b, mut eval_c, public_inputs, private_witness) =
                CoGroth16Icicle::<Bn254Bridge, PlainGroth16Driver>::setup::<
                    co_groth16::mpc::PlainGroth16Driver,
                    R,
                >(
                    0, // id irrelevant in the
                    prepared_key,
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
                    key.as_ref(),
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

            let prepared_key = get_or_prepare_bls12_377_key::<R>(key, matrices.num_constraints, matrices.num_instance_variables);
            let (key, mut eval_a, mut eval_b, mut eval_c, public_inputs, private_witness) =
                CoGroth16Icicle::<Bls12_377Bridge, PlainGroth16Driver>::setup::<
                    co_groth16::mpc::PlainGroth16Driver,
                    R,
                >(
                    0, // id irrelevant in the
                    prepared_key,
                    matrices,
                    private_witness,
                    public_inputs,
                    domain_size,
                )?;

            let icicle_proof =
                CoGroth16Icicle::<Bls12_377Bridge, PlainGroth16Driver>::prove_inner::<_, R>(
                    &(),
                    &(),
                    &mut (),
                    &mut (),
                    &mut eval_a,
                    &mut eval_b,
                    eval_c.as_mut(),
                    key.as_ref(),
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

            let prepared_key = get_or_prepare_bn254_key::<R>(key, matrices.num_constraints, matrices.num_instance_variables);
            let (key, mut eval_a, mut eval_b, mut eval_c, public_inputs, private_witness) =
                CoGroth16Icicle::<Bn254Bridge, Rep3Groth16Driver>::setup::<
                    co_groth16::mpc::Rep3Groth16Driver,
                    R,
                >(
                    state0.id(),
                    prepared_key,
                    matrices,
                    private_witness,
                    public_inputs,
                    domain_size,
                )?;

            let icicle_proof =
                CoGroth16Icicle::<Bn254Bridge, Rep3Groth16Driver>::prove_inner::<N, R>(
                    &net0,
                    &net1,
                    &mut state0,
                    &mut state1,
                    &mut eval_a,
                    &mut eval_b,
                    eval_c.as_mut(),
                    key.as_ref(),
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
