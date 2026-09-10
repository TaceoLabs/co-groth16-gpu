pub(crate) mod plain;
pub(crate) mod rep3;
pub(crate) mod shamir;

use icicle_core::{
    curve::Curve,
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

    /// Writes the trivial sharing of the public `public_values` into
    /// `dst[start..start + public_values.len()]`: \[a_i\] = a_i.
    ///
    /// Writes straight into the destination rather than returning a fresh buffer: on the
    /// CUDA backend both `icicle_malloc` and the matching `icicle_free` synchronize the
    /// whole device, so a per-proof allocation here would also stall any stream running
    /// concurrently with the witness map.
    fn write_trivial_shares_into(
        id: <Self::State as MpcState>::PartyID,
        public_values: &DeviceSlice<F>,
        dst: &mut Self::DeviceShares,
        start: usize,
    );

    /// Zeroes `dst[from..]`.
    ///
    /// Used to clear the domain padding of the constraint-evaluation buffers, which is
    /// always zero, without transferring it from the host: the host only computes
    /// `num_constraints`-length vectors (see [`crate::utils::evaluate_constraint`]), and the
    /// tail is zeroed here instead, then partly overwritten by
    /// [`Self::write_trivial_shares_into`].
    fn zero_device_shares_from(dst: &mut Self::DeviceShares, from: usize);

    /// Writes the public-input segment of the combined `a`/`b_g1`/`b_g2`-query MSM scalar
    /// buffer (see [`crate::groth16_gpu`]'s `combined_scalars`): the segment corresponding
    /// to `query[1..num_instance_variables]`, i.e. the actual public inputs.
    ///
    /// Merging the public and private-witness MSMs into one only works if the public
    /// contribution is still added exactly once across all parties once reconstructed, and
    /// how to achieve that is protocol-specific:
    /// - Rep3 reconstructs by a plain, unweighted sum of every party's share, so only one
    ///   party (`PartyID::ID0`, matching [`Self::add_assign_point_public`]'s gating) may
    ///   write the real public values; every other party must write zero, or the public
    ///   contribution would be counted once per party.
    /// - Shamir reconstructs via Lagrange weights that sum to 1, so writing the same public
    ///   values at every party is already correct (a constant added to every share of a
    ///   polynomial shifts its evaluation-at-0 by exactly that constant).
    /// - Plain has only one party, so there is nothing to gate.
    fn write_combined_public_segment(
        id: <Self::State as MpcState>::PartyID,
        public_values: &DeviceSlice<F>,
        dst: &mut DeviceSlice<F>,
    );

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

    /// Adds the public point `point` into the shared point accumulator `acc`.
    ///
    /// Takes arkworks points: everything after the MSMs is host-side curve arithmetic, and
    /// icicle's host implementations are generic C++ while `ark-ff` is built with `asm`
    /// here. Working in projective coordinates also keeps the conversion to affine (a field
    /// inversion each) to just the points that end up in the proof.
    fn add_assign_point_public<C: ark_ec::CurveGroup>(
        _: <Self::State as MpcState>::PartyID,
        acc: &mut C,
        point: &C,
    );

    /// Performs the Fast Fourier Transform (FFT) in place.
    fn fft_in_place(input: &mut Self::DeviceShares, stream: &IcicleStream, coset_gen: Option<F>);

    /// Performs the Inverse Fast Fourier Transform (IFFT) in place.
    fn ifft_in_place(input: &mut Self::DeviceShares, stream: &IcicleStream, coset_gen: Option<F>);

    // ICICLE <-> ARK functions

    /// Allocates an (uninitialized) vector of device shares of the given length.
    fn alloc_device_shares(len: usize) -> Self::DeviceShares;

    /// Uploads a vector of arithmetic shares into `dst` at the given offset.
    fn shares_to_device_into<
        B: ArkIcicleBridge<IcicleScalarField = F>,
        T: co_groth16::CircomGroth16Prover<B::ArkPairing> + 'static,
    >(
        shares: &[T::ArithmeticShare],
        dst: &mut Self::DeviceShares,
        start: usize,
    );

    /// Uploads a vector of arithmetic half shares into `dst` at the given offset.
    fn half_shares_to_device_into<
        B: ArkIcicleBridge<IcicleScalarField = F>,
        T: co_groth16::CircomGroth16Prover<B::ArkPairing> + 'static,
    >(
        shares: &[T::ArithmeticHalfShare],
        dst: &mut DeviceVec<F>,
        start: usize,
    );

    /// Uploads the half-share component of a vector of arithmetic shares into `dst` at the
    /// given offset. Local interaction only.
    fn shares_to_half_share_device_into<
        B: ArkIcicleBridge<IcicleScalarField = F>,
        T: co_groth16::CircomGroth16Prover<B::ArkPairing> + 'static,
    >(
        shares: &[T::ArithmeticShare],
        dst: &mut DeviceVec<F>,
        start: usize,
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
        a: B::ArkG1,
        b: B::ArkG1,
        net: &N,
        state: &mut Self::State,
    ) -> eyre::Result<(B::ArkG1, B::ArkG1)>;

    /// Reconstructs a shared point in G1 together with a shared point in G2 in a single
    /// communication round: (A, B) = (Open(\[A\]), Open(\[B\])).
    fn open_two_half_points_g1g2<N: Network, B: ArkIcicleBridge<IcicleScalarField = F>>(
        a: B::ArkG1,
        b: B::ArkG2,
        net: &N,
        state: &mut Self::State,
    ) -> eyre::Result<(B::ArkG1, B::ArkG2)>;

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
