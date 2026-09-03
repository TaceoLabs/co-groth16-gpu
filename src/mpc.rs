pub(crate) mod plain;
pub(crate) mod rep3;
pub(crate) mod shamir;

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

use crate::bridges::ArkIcicleBridge;

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

    /// Allocates an (uninitialized) vector of device shares of the given length.
    fn alloc_device_shares(len: usize) -> Self::DeviceShares;

    /// Uploads a vector of arithmetic shares into the pre-allocated device shares.
    fn shares_to_device_into<
        B: ArkIcicleBridge<IcicleScalarField = F>,
        T: co_groth16::CircomGroth16Prover<B::ArkPairing> + 'static,
    >(
        shares: &[T::ArithmeticShare],
        dst: &mut Self::DeviceShares,
    );

    /// Uploads a vector of arithmetic half shares into the pre-allocated device vector.
    fn half_shares_to_device_into<
        B: ArkIcicleBridge<IcicleScalarField = F>,
        T: co_groth16::CircomGroth16Prover<B::ArkPairing> + 'static,
    >(
        shares: &[T::ArithmeticHalfShare],
        dst: &mut DeviceVec<F>,
    );

    /// Uploads the half-share component of a vector of arithmetic shares into the
    /// pre-allocated device vector. Local interaction only.
    fn shares_to_half_share_device_into<
        B: ArkIcicleBridge<IcicleScalarField = F>,
        T: co_groth16::CircomGroth16Prover<B::ArkPairing> + 'static,
    >(
        shares: &[T::ArithmeticShare],
        dst: &mut DeviceVec<F>,
    );

    /// Performs element-wise multiplication of two vectors of shared values, writing the
    /// result into `result` (which must have the same length as the inputs).
    /// Does not perform any networking.
    ///
    /// # Security
    /// You must *NOT* perform additional non-linear operations on the result of this function.
    fn local_mul_vec<B: ArkIcicleBridge<IcicleScalarField = F>>(
        a: &Self::DeviceShares,
        b: &Self::DeviceShares,
        state: &mut Self::State,
        stream: &IcicleStream,
        result: &mut DeviceSlice<F>,
    );

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
