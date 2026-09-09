use std::{marker::PhantomData, mem::transmute, ops::IndexMut};

use ark_ec::CurveGroup;
use ark_ff::{BigInteger, PrimeField};
use icicle_core::{
    curve::{Affine, Curve},
    ntt::NTT,
    traits::{Arithmetic, FieldImpl, MontgomeryConvertible},
    vec_ops::{VecOps, VecOpsConfig, mul_scalars},
};
use icicle_runtime::{
    memory::{DeviceSlice, DeviceVec, HostOrDeviceSlice},
    stream::IcicleStream,
};
use mpc_core::{
    MpcState,
    protocols::shamir::{
        ShamirPrimeFieldShare, ShamirState, arithmetic, network::ShamirNetworkExt,
    },
};
use mpc_net::Network;
use rayon::prelude::*;

use crate::{
    bridges::{
        ArkIcicleBridge, ark_scalars_to_device_into_at, ark_to_icicle_scalar, icicle_to_ark_scalar,
    },
    gpu_utils::{fft_inplace, ifft_inplace, to_host_vec_icicle_scalar},
};

use super::CircomGroth16Prover;

/// A Groth16 driver for Shamir secret sharing.
///
/// This driver is generic over the arkworks scalar field `Fr` used by the concrete
/// `ArkIcicleBridge`, since (unlike Rep3) [`ShamirState`] carries field-dependent
/// Lagrange coefficients and thus cannot be field-agnostic.
pub struct ShamirGroth16Driver<Fr>(PhantomData<Fr>);

/// Casts a `ShamirState<Fr>` to a `ShamirState<ArkF>`.
///
/// This is only sound when `Fr` and `ArkF` are the same type, which callers must
/// guarantee by only ever using a bridge `B` whose `B::ArkScalarField` matches the
/// driver's `Fr`. Verified at runtime via a safe `Any` downcast rather than an
/// unchecked transmute.
///
/// TODO: this whole cast (and the `Fr` type parameter on `ShamirGroth16Driver`) can
/// be removed if `CircomGroth16Prover` is redefined to be generic over a single
/// `B: ArkIcicleBridge` per impl (instead of each method separately taking its own
/// `B: ArkIcicleBridge<IcicleScalarField = F>`). Then `ShamirGroth16Driver<B>` could
/// declare `type State = ShamirState<B::ArkScalarField>` directly, and the
/// driver/bridge pairing would be enforced by the type system instead of by callers
/// consistently picking a matching `Fr`. That requires updating the trait in
/// `mpc.rs`, all three driver impls, and the call sites in `groth16_gpu.rs`.
fn cast_state<Fr: PrimeField + 'static, ArkF: PrimeField + 'static>(
    state: &mut ShamirState<Fr>,
) -> &mut ShamirState<ArkF> {
    (state as &mut dyn std::any::Any)
        .downcast_mut::<ShamirState<ArkF>>()
        .expect("Invalid bridge: ArkScalarField does not match driver's scalar field")
}

/// Window size in bits for [`reconstruct_point`]'s interleaved ladder.
const RECONSTRUCT_WINDOW: usize = 4;
/// Number of precomputed multiples per share: `1*P, 2*P, ..., (2^w - 1)*P`.
const RECONSTRUCT_TABLE_LEN: usize = (1 << RECONSTRUCT_WINDOW) - 1;

/// Reconstructs a point from its Shamir shares: `sum_i lagrange_i * shares_i`.
///
/// Same value as `mpc_core`'s `reconstruct_point`, computed with an interleaved windowed
/// ladder so that the ~254 doublings are shared across all `2t+1` shares instead of being
/// repeated once per share. Opening a point is the one step where Shamir is markedly more
/// expensive than REP3 (which merely adds its shares), and every proof opens four of them.
fn reconstruct_point<C: CurveGroup>(shares: &[C], lagrange: &[C::ScalarField]) -> C {
    assert_eq!(
        shares.len(),
        lagrange.len(),
        "one lagrange coefficient per share"
    );

    // `tables[i * TABLE_LEN + (d - 1)] == d * shares[i]`.
    //
    // Kept projective on purpose: normalizing would let the ladder use mixed additions,
    // but `CurveGroup::normalize_batch` goes through a parallel iterator whose overhead
    // dwarfs the work at this size -- measured ~2.3 ms for 45 BN254 G1 points, against
    // ~63 us for the entire ladder.
    let mut tables = Vec::with_capacity(shares.len() * RECONSTRUCT_TABLE_LEN);
    for share in shares {
        let base = tables.len();
        tables.push(*share);
        for d in 1..RECONSTRUCT_TABLE_LEN {
            let next = tables[base + d - 1] + share;
            tables.push(next);
        }
    }

    let scalars = lagrange
        .iter()
        .map(|coeff| coeff.into_bigint())
        .collect::<Vec<_>>();

    let num_windows = (C::ScalarField::MODULUS_BIT_SIZE as usize).div_ceil(RECONSTRUCT_WINDOW);
    let mut acc = C::zero();
    for window in (0..num_windows).rev() {
        for _ in 0..RECONSTRUCT_WINDOW {
            acc.double_in_place();
        }
        let low_bit = window * RECONSTRUCT_WINDOW;
        for (i, scalar) in scalars.iter().enumerate() {
            let mut digit = 0usize;
            for bit in 0..RECONSTRUCT_WINDOW {
                if scalar.get_bit(low_bit + bit) {
                    digit |= 1 << bit;
                }
            }
            if digit != 0 {
                acc += tables[i * RECONSTRUCT_TABLE_LEN + digit - 1];
            }
        }
    }
    acc
}

/// Opens two degree-`2t` point shares (possibly on different curves) in a single round.
///
/// Mirrors `mpc_core`'s `pointshare::open_two_half_points`, but reconstructs with
/// [`reconstruct_point`] above.
fn open_two_half_points<C1, C2, N: Network>(
    a: C1,
    b: C2,
    net: &N,
    state: &ShamirState<C1::ScalarField>,
) -> eyre::Result<(C1, C2)>
where
    C1: CurveGroup,
    C2: CurveGroup<ScalarField = C1::ScalarField>,
{
    let rcv = net.broadcast_next(state.num_parties, 2 * state.threshold + 1, (a, b))?;
    let (rcv_a, rcv_b): (Vec<_>, Vec<_>) = rcv.into_iter().unzip();
    Ok((
        reconstruct_point(&rcv_a, &state.open_lagrange_2t),
        reconstruct_point(&rcv_b, &state.open_lagrange_2t),
    ))
}

impl<F, Fr> CircomGroth16Prover<F> for ShamirGroth16Driver<Fr>
where
    F: FieldImpl<Config: VecOps<F> + NTT<F, F>> + Arithmetic + MontgomeryConvertible,
    Fr: PrimeField,
{
    type ArithmeticShare = F;

    type DeviceShares = DeviceVec<F>;
    type DevicePointShares<C: Curve<ScalarField = F>> = DeviceVec<Affine<C>>;

    type State = ShamirState<Fr>;

    fn to_half_share(a: &Self::ArithmeticShare) -> F {
        *a
    }

    fn write_trivial_shares_into(
        _: <Self::State as MpcState>::PartyID,
        public_values: &DeviceSlice<F>,
        dst: &mut Self::DeviceShares,
        start: usize,
    ) {
        dst.index_mut(start..start + public_values.len())
            .copy(public_values)
            .expect("Failed to write trivial shares into device buffer");
    }

    fn zero_device_shares_from(dst: &mut Self::DeviceShares, from: usize) {
        let len = dst.len() - from;
        dst.index_mut(from..)
            .memset(0, len)
            .expect("Failed to zero device buffer tail");
    }

    fn write_combined_public_segment(
        _: <Self::State as MpcState>::PartyID,
        public_values: &DeviceSlice<F>,
        dst: &mut DeviceSlice<F>,
    ) {
        // Every party writes the same real values: see the trait docs.
        dst.copy(public_values)
            .expect("Failed to write public segment of combined MSM scalars");
    }

    fn distribute_powers_and_mul_by_const(
        coeffs: &mut Self::DeviceShares,
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

    fn add_assign_point_public<C: CurveGroup>(
        _: <Self::State as MpcState>::PartyID,
        acc: &mut C,
        point: &C,
    ) {
        // Adding a public point at *every* party is a valid Shamir operation: the open
        // lagrange coefficients interpolate a constant to itself, i.e. they sum to one.
        *acc += point;
    }

    fn fft_in_place(input: &mut Self::DeviceShares, stream: &IcicleStream, coset_gen: Option<F>) {
        fft_inplace(input, stream, coset_gen);
    }

    fn ifft_in_place(input: &mut Self::DeviceShares, stream: &IcicleStream, coset_gen: Option<F>) {
        ifft_inplace(input, stream, coset_gen);
    }

    fn alloc_device_shares(len: usize) -> Self::DeviceShares {
        DeviceVec::device_malloc(len).expect("Failed to allocate device vector")
    }

    fn shares_to_device_into<
        B: ArkIcicleBridge<IcicleScalarField = F>,
        T: co_groth16::CircomGroth16Prover<B::ArkPairing> + 'static,
    >(
        shares: &[T::ArithmeticShare],
        dst: &mut Self::DeviceShares,
        start: usize,
    ) {
        if std::any::TypeId::of::<T>()
            != std::any::TypeId::of::<co_groth16::mpc::ShamirGroth16Driver>()
        {
            panic!("Invalid driver: expected ShamirGroth16Driver");
        }

        // SAFETY: At this point we know T::ArithmeticShare = ShamirPrimeFieldShare<B::ArkScalarField>,
        // which is repr(transparent) over B::ArkScalarField.
        let shares = unsafe { transmute::<&[T::ArithmeticShare], &[B::ArkScalarField]>(shares) };
        ark_scalars_to_device_into_at(shares, dst, start);
    }

    fn half_shares_to_device_into<
        B: ArkIcicleBridge<IcicleScalarField = F>,
        T: co_groth16::CircomGroth16Prover<B::ArkPairing> + 'static,
    >(
        shares: &[T::ArithmeticHalfShare],
        dst: &mut DeviceVec<F>,
        start: usize,
    ) {
        if std::any::TypeId::of::<T>()
            != std::any::TypeId::of::<co_groth16::mpc::ShamirGroth16Driver>()
        {
            panic!("Invalid driver: expected ShamirGroth16Driver");
        }

        // SAFETY: At this point we know the shares are safe to transmute
        let shares =
            unsafe { transmute::<&[T::ArithmeticHalfShare], &[B::ArkScalarField]>(shares) };
        ark_scalars_to_device_into_at(shares, dst, start);
    }

    fn shares_to_half_share_device_into<
        B: ArkIcicleBridge<IcicleScalarField = F>,
        T: co_groth16::CircomGroth16Prover<B::ArkPairing> + 'static,
    >(
        shares: &[T::ArithmeticShare],
        dst: &mut DeviceVec<F>,
    ) {
        // A degree-t Shamir share is already a valid degree-2t (half) share, so there's
        // nothing to convert.
        Self::shares_to_device_into::<B, T>(shares, dst, 0);
    }

    fn local_mul_vec<B: ArkIcicleBridge<IcicleScalarField = F>>(
        a: &Self::DeviceShares,
        b: &Self::DeviceShares,
        _: &mut Self::State,
        stream: &IcicleStream,
        result: &mut DeviceSlice<F>,
    ) {
        let mut cfg = VecOpsConfig::default();
        cfg.stream_handle = **stream;
        cfg.is_async = true;

        mul_scalars(a, b, result, &cfg).unwrap();
        stream
            .synchronize()
            .expect("Failed to synchronize local_mul_vec stream");
    }

    fn local_mul<B: ArkIcicleBridge<IcicleScalarField = F>>(
        a: &Self::ArithmeticShare,
        b: &Self::ArithmeticShare,
        _: &mut Self::State,
    ) -> F {
        *a * *b
    }

    fn rand<N: Network, B: ArkIcicleBridge<IcicleScalarField = F>>(
        net: &N,
        state: &mut Self::State,
    ) -> eyre::Result<Self::ArithmeticShare> {
        let state = cast_state::<Fr, B::ArkScalarField>(state);
        let res = state.rand(net)?;
        Ok(ark_to_icicle_scalar(res.inner()))
    }

    fn open_two_half_points_g1<N: Network, B: ArkIcicleBridge<IcicleScalarField = F>>(
        a: B::ArkG1,
        b: B::ArkG1,
        net: &N,
        state: &mut Self::State,
    ) -> eyre::Result<(B::ArkG1, B::ArkG1)> {
        let state = cast_state::<Fr, B::ArkScalarField>(state);
        open_two_half_points(a, b, net, state)
    }

    fn open_two_half_points_g1g2<N: Network, B: ArkIcicleBridge<IcicleScalarField = F>>(
        a: B::ArkG1,
        b: B::ArkG2,
        net: &N,
        state: &mut Self::State,
    ) -> eyre::Result<(B::ArkG1, B::ArkG2)> {
        let state = cast_state::<Fr, B::ArkScalarField>(state);
        open_two_half_points(a, b, net, state)
    }

    fn open_device_shares<N: Network, B: ArkIcicleBridge<IcicleScalarField = F>>(
        shares: &Self::DeviceShares,
        net: &N,
        state: &mut Self::State,
    ) -> eyre::Result<Vec<B::ArkScalarField>> {
        let host_a = to_host_vec_icicle_scalar(shares)
            .into_par_iter()
            .with_min_len(1024)
            .map(icicle_to_ark_scalar::<B::ArkScalarField, _>)
            .collect::<Vec<_>>();

        let shares = ShamirPrimeFieldShare::convert_vec_rev(host_a);
        let state = cast_state::<Fr, B::ArkScalarField>(state);
        let opened = arithmetic::open_vec(&shares, net, state)?;

        Ok(opened)
    }

    fn open_device_half_shares<N: Network, B: ArkIcicleBridge<IcicleScalarField = F>>(
        shares: &DeviceVec<F>,
        net: &N,
        state: &mut Self::State,
    ) -> eyre::Result<Vec<B::ArkScalarField>> {
        let host_a = to_host_vec_icicle_scalar(shares)
            .into_par_iter()
            .with_min_len(1024)
            .map(icicle_to_ark_scalar::<B::ArkScalarField, _>)
            .collect::<Vec<_>>();

        let state = cast_state::<Fr, B::ArkScalarField>(state);

        // Values passed through `local_mul_vec`/`local_mul` are degree-2t shares, so opening
        // them requires 2t+1 shares and the corresponding Lagrange coefficients, unlike
        // `open_device_shares` which opens ordinary degree-t shares.
        let rcv = net.broadcast_next(state.num_parties, 2 * state.threshold + 1, host_a)?;

        // Reconstruct each element as a dot product over the (few, `2t+1`-sized) received rows,
        // parallelizing over `len` (typically domain- or witness-sized) rather than over the
        // rows, and without ever materializing a `len`-sized transpose.
        let len = rcv.first().map_or(0, Vec::len);
        let result = (0..len)
            .into_par_iter()
            .with_min_len(1024)
            .map(|i| {
                rcv.iter()
                    .zip(state.open_lagrange_2t.iter())
                    .map(|(row, coeff)| row[i] * coeff)
                    .sum()
            })
            .collect();

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::{RECONSTRUCT_TABLE_LEN, reconstruct_point};
    use ark_ec::CurveGroup;
    use ark_ff::{UniformRand, Zero};
    use rand::thread_rng;

    /// The reference implementation from `mpc_core`: one full scalar multiplication per share.
    fn reconstruct_point_naive<C: CurveGroup>(shares: &[C], lagrange: &[C::ScalarField]) -> C {
        let mut res = C::zero();
        for (s, l) in shares.iter().zip(lagrange.iter()) {
            res += *s * l;
        }
        res
    }

    fn check_matches_naive<C: CurveGroup>(num_shares: usize) {
        let mut rng = thread_rng();
        for _ in 0..20 {
            let shares = (0..num_shares)
                .map(|_| C::rand(&mut rng))
                .collect::<Vec<_>>();
            let lagrange = (0..num_shares)
                .map(|_| C::ScalarField::rand(&mut rng))
                .collect::<Vec<_>>();
            assert_eq!(
                reconstruct_point(&shares, &lagrange),
                reconstruct_point_naive(&shares, &lagrange)
            );
        }
    }

    #[test]
    fn reconstruct_matches_naive_g1() {
        // 2t + 1 shares for t = 1 (the REP3-equivalent setting) and t = 2.
        check_matches_naive::<ark_bn254::G1Projective>(3);
        check_matches_naive::<ark_bn254::G1Projective>(5);
    }

    #[test]
    fn reconstruct_matches_naive_g2() {
        check_matches_naive::<ark_bn254::G2Projective>(3);
    }

    #[test]
    fn reconstruct_handles_zero_scalars_and_identity_points() {
        let mut rng = thread_rng();
        type G = ark_bn254::G1Projective;
        let cases: [(Vec<G>, Vec<ark_bn254::Fr>); 4] = [
            (vec![G::zero(); 3], vec![ark_bn254::Fr::rand(&mut rng); 3]),
            (
                (0..3).map(|_| G::rand(&mut rng)).collect(),
                vec![ark_bn254::Fr::from(0u64); 3],
            ),
            (
                vec![G::zero(), G::rand(&mut rng), G::zero()],
                vec![
                    ark_bn254::Fr::from(0u64),
                    ark_bn254::Fr::from(1u64),
                    ark_bn254::Fr::rand(&mut rng),
                ],
            ),
            // A scalar whose top window is non-zero exercises the very first iteration.
            (
                (0..3).map(|_| G::rand(&mut rng)).collect(),
                vec![-ark_bn254::Fr::from(1u64); 3],
            ),
        ];
        for (shares, lagrange) in cases {
            assert_eq!(
                reconstruct_point(&shares, &lagrange),
                reconstruct_point_naive(&shares, &lagrange)
            );
        }
    }

    /// Not an assertion of speed, just a printed comparison so the choice of algorithm can
    /// be re-checked on the target machine: `cargo test --release compare_reconstruct -- --nocapture`.
    #[test]
    fn compare_reconstruct_strategies() {
        use std::time::Instant;

        fn bench<C: CurveGroup>(label: &str, iters: usize) {
            use std::hint::black_box;

            let mut rng = thread_rng();
            let shares = (0..3).map(|_| C::rand(&mut rng)).collect::<Vec<_>>();
            let lagrange = (0..3)
                .map(|_| C::ScalarField::rand(&mut rng))
                .collect::<Vec<_>>();

            // `black_box` on the inputs, otherwise the loop-invariant candidates that do
            // not allocate get hoisted out of their timing loop and look free.
            let t = Instant::now();
            let mut sink = C::zero();
            for _ in 0..iters {
                sink += reconstruct_point_naive(black_box(&shares), black_box(&lagrange));
            }
            let naive = t.elapsed() / iters as u32;

            let t = Instant::now();
            for _ in 0..iters {
                let bases = C::normalize_batch(black_box(&shares));
                sink += C::msm_unchecked(black_box(&bases), black_box(&lagrange));
            }
            let ark_msm = t.elapsed() / iters as u32;

            let t = Instant::now();
            for _ in 0..iters {
                sink += reconstruct_point(black_box(&shares), black_box(&lagrange));
            }
            let interleaved = t.elapsed() / iters as u32;

            assert_ne!(sink, C::zero());
            println!(
                "{label}: naive {naive:?} | ark msm {ark_msm:?} | interleaved {interleaved:?} \
                 (speedup vs naive: {:.2}x)",
                naive.as_secs_f64() / interleaved.as_secs_f64()
            );
        }

        assert_eq!(RECONSTRUCT_TABLE_LEN, 15);
        bench::<ark_bn254::G1Projective>("G1", 2000);
        bench::<ark_bn254::G2Projective>("G2", 1000);
    }
}
