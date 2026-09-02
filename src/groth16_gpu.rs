//! A Groth16 proof protocol that uses a collaborative MPC protocol to generate the proof.
use crate::gpu_utils::{PRECOMPUTE_FACTOR_G1, PRECOMPUTE_FACTOR_G2, get_first, msm_async};
use ark_bn254::Bn254;
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

use crate::bridges::{ArkIcicleBridge, Bn254Bridge, ark_scalars_to_device_into};
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
    witness_half_shares: DeviceVec<B::IcicleScalarField>,
    public_inputs: DeviceVec<B::IcicleScalarField>,
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
        let alloc = |len| DeviceVec::device_malloc(len).expect("Failed to allocate device vector");
        let domain_size = prepared_key.domain_size;
        Self {
            scratch: ReductionScratch::new(domain_size, requires_eval_c),
            streams: ProofStreams::new(),
            eval_a: T::alloc_device_shares(domain_size),
            eval_b: T::alloc_device_shares(domain_size),
            eval_c: requires_eval_c.then(|| alloc(domain_size)),
            witness_half_shares: alloc(prepared_key.num_witness_variables),
            public_inputs: alloc(prepared_key.num_instance_variables),
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
    ) -> eyre::Result<Proof<B::IcicleScalarField, B::IcicleG1, B::IcicleG2>> {
        let setup_timer = std::time::Instant::now();
        let id = state.id();
        // SAFETY: matching GPU/CPU driver pairs use the same PartyID type
        let id = unsafe {
            transmute::<&<T::State as MpcState>::PartyID, &<U::State as MpcState>::PartyID>(&id)
        };
        let domain_size = self.prepared_key.domain_size;

        let eval_a = evaluate_constraint::<B::ArkPairing, U>(
            *id,
            domain_size,
            &matrices.a,
            public_inputs,
            private_witness,
        );
        T::shares_to_device_into::<B, U>(&eval_a, &mut self.eval_a);

        let eval_b = evaluate_constraint::<B::ArkPairing, U>(
            *id,
            domain_size,
            &matrices.b,
            public_inputs,
            private_witness,
        );
        T::shares_to_device_into::<B, U>(&eval_b, &mut self.eval_b);

        if let Some(eval_c_buf) = self.eval_c.as_mut() {
            let eval_c = evaluate_constraint_half_share::<B::ArkPairing, U>(
                *id,
                domain_size,
                &matrices.c,
                public_inputs,
                private_witness,
            );
            T::half_shares_to_device_into::<B, U>(&eval_c, eval_c_buf);
        }

        T::shares_to_half_share_device_into::<B, U>(private_witness, &mut self.witness_half_shares);
        ark_scalars_to_device_into(public_inputs, &mut self.public_inputs);

        tracing::info!(
            "Constraint evaluation + device upload took {} ms",
            setup_timer.elapsed().as_millis()
        );

        self.prove_inner::<N, R>(net, state)
    }

    /// Computes the QAP witness and creates the proof from the uploaded inputs.
    fn prove_inner<N: Network, R: R1CSToQAP>(
        &mut self,
        net: &N,
        state: &mut T::State,
    ) -> eyre::Result<Proof<B::IcicleScalarField, B::IcicleG1, B::IcicleG2>> {
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

        let (r, s) = (T::rand::<_, B>(net, state)?, T::rand::<_, B>(net, state)?);

        self.create_proof_with_assignment(net, state, r, s)
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

    /// Creates the proof from the QAP witness left in `self.scratch.h` by the reduction.
    fn create_proof_with_assignment<N: Network>(
        &self,
        net: &N,
        state: &mut T::State,
        r: T::ArithmeticShare,
        s: T::ArithmeticShare,
    ) -> eyre::Result<Proof<B::IcicleScalarField, B::IcicleG1, B::IcicleG2>> {
        let h = &self.scratch.h;
        let input_assignment = &self.public_inputs;
        let aux_assignment = &self.witness_half_shares;
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

        let stream_g1 = &self.streams.g1;
        let stream_g2 = &self.streams.g2;

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
