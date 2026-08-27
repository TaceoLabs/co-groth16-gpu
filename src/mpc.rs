pub(crate) mod plain;
pub(crate) mod rep3;
pub(crate) mod shamir;

use std::mem::transmute;

use co_groth16::ConstraintMatrices;
use icicle_core::{
    curve::{Affine, Curve},
    ntt::NTT,
    traits::{Arithmetic, FieldImpl, MontgomeryConvertible},
    vec_ops::{VecOps, VecOpsConfig, mul_scalars},
};
use icicle_runtime::{
    memory::{DeviceSlice, DeviceVec},
    stream::IcicleStream,
};
use mpc_core::MpcState;
use mpc_net::Network;

use crate::{
    bridges::ArkIcicleBridge,
    utils::{evaluate_constraint, evaluate_constraint_half_share},
};

pub use plain::PlainGroth16Driver;
pub use rep3::Rep3Groth16Driver;
pub use shamir::ShamirGroth16Driver;

/// This trait represents the operations used during Groth16 proof generation
pub trait CircomGroth16Prover<
    F: FieldImpl<Config: VecOps<F> + NTT<F, F>> + Arithmetic + MontgomeryConvertible,
>: Send + Sized
{
    /// The arithmetic share type
    type ArithmeticShare: Send;

    /// Represents a vector of field shares on the device
    type DeviceShares;

    /// Represents a vector of point shares on the device
    type DevicePointShares<C: Curve<ScalarField = F>>;

    /// Internal state of used MPC protocol
    type State: MpcState + Send;

    /// Elementwise transformation of a vector of public values into a vector of shared values: \[a_i\] = a_i.
    fn promote_to_trivial_shares(
        id: <Self::State as MpcState>::PartyID,
        public_values: &DeviceSlice<F>,
    ) -> Self::DeviceShares;

    /// Computes the \[coeffs_i\] *= c * g^i for the coefficients in 0 <= i < coeff.len()
    fn distribute_powers_and_mul_by_const(
        coeffs: &mut Self::DeviceShares,
        roots: &DeviceSlice<F>,
        stream: &IcicleStream,
    );

    /// Computes the \[coeffs_i\] *= c * g^i for the coefficients in 0 <= i < coeff.len()
    fn distribute_powers_and_mul_by_const_hs(
        coeffs: &mut DeviceVec<F>,
        roots: &DeviceSlice<F>,
        stream: &IcicleStream,
    ) {
        let mut cfg = VecOpsConfig::default();
        cfg.stream_handle = **stream;
        cfg.is_async = true;

        // SAFETY: elementwise mul so in place aliasing is sound
        let coeffs_in: &DeviceSlice<F> = unsafe { &*(&**coeffs as *const DeviceSlice<F>) };
        mul_scalars(coeffs_in, roots, coeffs.as_mut_slice(), &cfg).unwrap();
    }

    /// Converts a shared value to a half shared value. Local interaction only.
    fn to_half_share(a: &Self::ArithmeticShare) -> F;

    /// Converts shared values to half shared values. Local interaction only.
    fn to_half_share_vec(a: Self::DeviceShares) -> DeviceVec<F>;

    /// Add a public point B in place to the shared point A
    fn add_assign_points_public_hs<C: Curve<ScalarField = F>>(
        _: <Self::State as MpcState>::PartyID,
        a: &mut Affine<C>,
        b: &Affine<C>,
    );

    /// Performs the Fast Fourier Transform (FFT) in place.
    fn fft_in_place(input: &mut Self::DeviceShares, stream: &IcicleStream, coset_gen: Option<F>);

    /// Performs the Inverse Fast Fourier Transform (IFFT) in place.
    fn ifft_in_place(input: &mut Self::DeviceShares, stream: &IcicleStream, coset_gen: Option<F>);

    /// Copies a slice of device shares to another device shares vector,
    /// starting and ending at the specified indices.
    fn copy_to_device_shares(
        src: &Self::DeviceShares,
        dst: &mut Self::DeviceShares,
        start: usize,
        end: usize,
    );

    // ICICLE <-> ARK functions

    /// Converts a vector of arithmetic shares to device shares. Runs asynchronously on `stream`;
    /// the caller is responsible for synchronizing before relying on the result from a different
    /// stream (or from the host).
    fn shares_to_device<
        B: ArkIcicleBridge<IcicleScalarField = F>,
        T: co_groth16::CircomGroth16Prover<B::ArkPairing> + 'static,
    >(
        shares: &[T::ArithmeticShare],
        stream: &IcicleStream,
    ) -> Self::DeviceShares;

    /// Converts a vector of arithmetic shares to device shares. Runs asynchronously on `stream`;
    /// the caller is responsible for synchronizing before relying on the result from a different
    /// stream (or from the host).
    fn half_shares_to_device<
        B: ArkIcicleBridge<IcicleScalarField = F>,
        T: co_groth16::CircomGroth16Prover<B::ArkPairing> + 'static,
    >(
        shares: &[T::ArithmeticHalfShare],
        stream: &IcicleStream,
    ) -> DeviceVec<F>;

    /// Evaluates the constraints for a given party ID and transforms the results into device shares.
    ///
    /// `stream_a`/`stream_b`/`stream_c` are used for the respective `a`/`b`/`c` uploads. Since the
    /// upload dispatch is asynchronous, the CPU-bound evaluation of the next constraint set
    /// overlaps with the GPU-bound upload of the previous one; the caller must synchronize the
    /// three streams before relying on the results from a different stream (or from the host).
    #[expect(clippy::too_many_arguments)]
    fn evaluate_constraints<
        B: ArkIcicleBridge<IcicleScalarField = F>,
        T: co_groth16::CircomGroth16Prover<B::ArkPairing> + 'static,
    >(
        id: <Self::State as MpcState>::PartyID,
        matrices: &ConstraintMatrices<B::ArkScalarField>,
        public_inputs: &[B::ArkScalarField],
        private_witness: &[T::ArithmeticShare],
        eval_c: bool,
        domain_size: usize,
        stream_a: &IcicleStream,
        stream_b: &IcicleStream,
        stream_c: &IcicleStream,
    ) -> (
        Self::DeviceShares,
        Self::DeviceShares,
        Option<DeviceVec<B::IcicleScalarField>>,
    ) {
        let id = unsafe {
            transmute::<&<Self::State as MpcState>::PartyID, &<T::State as MpcState>::PartyID>(&id)
        };

        let eval_a = evaluate_constraint::<B::ArkPairing, T>(
            *id,
            domain_size,
            &matrices.a,
            public_inputs,
            private_witness,
        );
        let eval_a = Self::shares_to_device::<B, T>(&eval_a, stream_a);

        let eval_b = evaluate_constraint::<B::ArkPairing, T>(
            *id,
            domain_size,
            &matrices.b,
            public_inputs,
            private_witness,
        );
        let eval_b = Self::shares_to_device::<B, T>(&eval_b, stream_b);

        let eval_c = if eval_c {
            let eval_c = evaluate_constraint_half_share::<B::ArkPairing, T>(
                *id,
                domain_size,
                &matrices.c,
                public_inputs,
                private_witness,
            );
            Some(Self::half_shares_to_device::<B, T>(&eval_c, stream_c))
        } else {
            None
        };

        (eval_a, eval_b, eval_c)
    }

    /// Performs element-wise multiplication of two vectors of shared values.
    /// Does not perform any networking.
    ///
    /// # Security
    /// You must *NOT* perform additional non-linear operations on the result of this function.
    fn local_mul_vec<B: ArkIcicleBridge<IcicleScalarField = F>>(
        a: &Self::DeviceShares,
        b: &Self::DeviceShares,
        state: &mut Self::State,
        stream: &IcicleStream,
    ) -> DeviceVec<F>;

    /// Performs multiplication of two shared values.
    /// Does not perform any networking.
    ///
    /// # Security
    /// You must *NOT* perform additional non-linear operations on the result of this function.
    fn local_mul<B: ArkIcicleBridge<IcicleScalarField = F>>(
        a: &Self::ArithmeticShare,
        b: &Self::ArithmeticShare,
        state: &mut Self::State,
    ) -> F;

    /// Generate a random arithmetic share
    fn rand<N: Network, B: ArkIcicleBridge<IcicleScalarField = F>>(
        net: &N,
        state: &mut Self::State,
    ) -> eyre::Result<Self::ArithmeticShare>;

    /// Reconstructs two shared points in G1 in a single communication round:
    /// (A, B) = (Open(\[A\]), Open(\[B\])).
    fn open_two_half_points_g1<N: Network, B: ArkIcicleBridge<IcicleScalarField = F>>(
        a: Affine<B::IcicleG1>,
        b: Affine<B::IcicleG1>,
        net: &N,
        state: &mut Self::State,
    ) -> eyre::Result<(Affine<B::IcicleG1>, Affine<B::IcicleG1>)>;

    /// Reconstructs a shared point in G1 together with a shared point in G2 in a single
    /// communication round: (A, B) = (Open(\[A\]), Open(\[B\])).
    fn open_two_half_points_g1g2<N: Network, B: ArkIcicleBridge<IcicleScalarField = F>>(
        a: Affine<B::IcicleG1>,
        b: Affine<B::IcicleG2>,
        net: &N,
        state: &mut Self::State,
    ) -> eyre::Result<(Affine<B::IcicleG1>, Affine<B::IcicleG2>)>;

    fn open_device_shares<N: Network, B: ArkIcicleBridge<IcicleScalarField = F>>(
        shares: &Self::DeviceShares,
        net: &N,
        state: &mut Self::State,
    ) -> eyre::Result<Vec<B::ArkScalarField>>;

    fn open_device_half_shares<N: Network, B: ArkIcicleBridge<IcicleScalarField = F>>(
        shares: &DeviceVec<F>,
        net: &N,
        state: &mut Self::State,
    ) -> eyre::Result<Vec<B::ArkScalarField>>;
}
