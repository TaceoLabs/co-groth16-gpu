use std::{marker::PhantomData, mem::transmute, ops::IndexMut};

use ark_ec::CurveGroup;
use ark_ff::PrimeField;
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
        ShamirPrimeFieldShare, ShamirState, arithmetic, network::ShamirNetworkExt, pointshare,
    },
};
use mpc_net::Network;
use rayon::prelude::*;

use crate::{
    bridges::{
        ArkIcicleBridge, ark_to_icicle_affine, ark_to_icicle_scalar, ark_to_icicle_scalars,
        icicle_to_ark_scalar,
    },
    gpu_utils::{fft_inplace, from_host_slice, ifft_inplace, to_host_vec_icicle_scalar},
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

    fn to_half_share_vec(a: Self::DeviceShares) -> DeviceVec<F> {
        // A degree-t Shamir share is already a valid degree-2t (half) share, so there's
        // nothing to convert.
        a
    }

    fn promote_to_trivial_shares(
        _: <Self::State as MpcState>::PartyID,
        public_values: &DeviceSlice<F>,
    ) -> Self::DeviceShares {
        let mut result = DeviceVec::device_malloc(public_values.len())
            .expect("Failed to allocate device vector");
        result.copy(public_values).unwrap();
        result
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

    fn add_assign_points_public_hs<C: Curve<ScalarField = F>>(
        _: <Self::State as MpcState>::PartyID,
        a: &mut Affine<C>,
        b: &Affine<C>,
    ) {
        *a = (a.to_projective() + b.to_projective()).into();
    }

    fn fft_in_place(input: &mut Self::DeviceShares, stream: &IcicleStream, coset_gen: Option<F>) {
        fft_inplace(input, stream, coset_gen);
    }

    fn ifft_in_place(input: &mut Self::DeviceShares, stream: &IcicleStream, coset_gen: Option<F>) {
        ifft_inplace(input, stream, coset_gen);
    }

    fn copy_to_device_shares(
        src: &Self::DeviceShares,
        dst: &mut Self::DeviceShares,
        start: usize,
        end: usize,
    ) {
        dst.index_mut(start..end).copy(src).unwrap();
    }

    fn shares_to_device<
        B: ArkIcicleBridge<IcicleScalarField = F>,
        T: co_groth16::CircomGroth16Prover<B::ArkPairing> + 'static,
    >(
        shares: &[T::ArithmeticShare],
    ) -> Self::DeviceShares {
        if std::any::TypeId::of::<T>()
            != std::any::TypeId::of::<co_groth16::mpc::ShamirGroth16Driver>()
        {
            panic!("Invalid driver: expected ShamirGroth16Driver");
        }

        // SAFETY: At this point we know T::ArithmeticShare = ShamirPrimeFieldShare<B::ArkScalarField>,
        // which is repr(transparent) over B::ArkScalarField.
        let shares = unsafe { transmute::<&[T::ArithmeticShare], &[B::ArkScalarField]>(shares) };

        let shares_icicle = from_host_slice(shares);
        ark_to_icicle_scalars(shares_icicle).unwrap()
    }

    fn half_shares_to_device<
        B: ArkIcicleBridge<IcicleScalarField = F>,
        T: co_groth16::CircomGroth16Prover<B::ArkPairing> + 'static,
    >(
        shares: &[T::ArithmeticHalfShare],
    ) -> DeviceVec<F> {
        if std::any::TypeId::of::<T>()
            != std::any::TypeId::of::<co_groth16::mpc::ShamirGroth16Driver>()
        {
            panic!("Invalid driver: expected ShamirGroth16Driver");
        }

        // SAFETY: At this point we know the shares are safe to transmute
        let shares =
            unsafe { transmute::<&[T::ArithmeticHalfShare], &[B::ArkScalarField]>(shares) };

        let shares_icicle = from_host_slice(shares);
        ark_to_icicle_scalars(shares_icicle).unwrap()
    }

    fn local_mul_vec<B: ArkIcicleBridge<IcicleScalarField = F>>(
        a: &Self::DeviceShares,
        b: &Self::DeviceShares,
        _: &mut Self::State,
        stream: &IcicleStream,
    ) -> DeviceVec<F> {
        let mut result = DeviceVec::device_malloc_async(a.len(), stream)
            .expect("Failed to allocate device vector");
        let mut cfg = VecOpsConfig::default();
        cfg.stream_handle = **stream;
        cfg.is_async = true;

        mul_scalars(a, b, result.as_mut_slice(), &cfg).unwrap();
        stream
            .synchronize()
            .expect("Failed to synchronize local_mul_vec stream");
        result
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
        a: Affine<B::IcicleG1>,
        b: Affine<B::IcicleG1>,
        net: &N,
        state: &mut Self::State,
    ) -> eyre::Result<(Affine<B::IcicleG1>, Affine<B::IcicleG1>)> {
        let ark_a = B::icicle_to_ark_g1(a);
        let ark_b = B::icicle_to_ark_g1(b);
        let state = cast_state::<Fr, B::ArkScalarField>(state);
        let (open_a, open_b) =
            pointshare::open_two_half_points(ark_a.into(), ark_b.into(), net, state)?;
        Ok((
            ark_to_icicle_affine(&open_a.into_affine()),
            ark_to_icicle_affine(&open_b.into_affine()),
        ))
    }

    fn open_two_half_points_g1g2<N: Network, B: ArkIcicleBridge<IcicleScalarField = F>>(
        a: Affine<B::IcicleG1>,
        b: Affine<B::IcicleG2>,
        net: &N,
        state: &mut Self::State,
    ) -> eyre::Result<(Affine<B::IcicleG1>, Affine<B::IcicleG2>)> {
        let ark_a = B::icicle_to_ark_g1(a);
        let ark_b = B::icicle_to_ark_g2(b);
        let state = cast_state::<Fr, B::ArkScalarField>(state);
        let (open_a, open_b) =
            pointshare::open_two_half_points(ark_a.into(), ark_b.into(), net, state)?;
        Ok((
            ark_to_icicle_affine(&open_a.into_affine()),
            ark_to_icicle_affine(&open_b.into_affine()),
        ))
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
