use ark_ff::UniformRand;
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
use mpc_core::MpcState;
use mpc_net::Network;
use rand::thread_rng;
use rayon::prelude::*;
use std::{mem::transmute, ops::IndexMut};

use crate::{
    bridges::{ArkIcicleBridge, ark_to_icicle_scalar, ark_to_icicle_scalars, icicle_to_ark_scalar},
    gpu_utils::{fft_inplace, from_host_slice_async, ifft_inplace, to_host_vec_icicle_scalar},
};

use super::CircomGroth16Prover;

/// A plain Groth16 driver
pub struct PlainGroth16Driver;

impl<F: FieldImpl<Config: VecOps<F> + NTT<F, F>> + Arithmetic + MontgomeryConvertible>
    CircomGroth16Prover<F> for PlainGroth16Driver
{
    type ArithmeticShare = F;

    type DeviceShares = DeviceVec<F>;
    type DevicePointShares<C: Curve<ScalarField = F>> = DeviceVec<Affine<C>>;

    type State = ();
    fn to_half_share(a: &Self::ArithmeticShare) -> F {
        *a
    }

    fn to_half_share_vec(a: Self::DeviceShares) -> DeviceVec<F> {
        // A plain share already *is* its half share, so there's nothing to convert.
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
        stream: &IcicleStream,
    ) -> Self::DeviceShares {
        if std::any::TypeId::of::<T>()
            != std::any::TypeId::of::<co_groth16::mpc::PlainGroth16Driver>()
        {
            panic!("Invalid driver: expected PlainGroth16Driver");
        }

        // SAFETY: At this point we know the shares are safe to transmute
        let shares = unsafe { transmute::<&[T::ArithmeticShare], &[B::ArkScalarField]>(shares) };

        let shares_icicle = from_host_slice_async(shares, stream);
        ark_to_icicle_scalars(shares_icicle, stream).unwrap()
    }

    fn half_shares_to_device<
        B: ArkIcicleBridge<IcicleScalarField = F>,
        T: co_groth16::CircomGroth16Prover<B::ArkPairing> + 'static,
    >(
        shares: &[T::ArithmeticHalfShare],
        stream: &IcicleStream,
    ) -> Self::DeviceShares {
        if std::any::TypeId::of::<T>()
            != std::any::TypeId::of::<co_groth16::mpc::PlainGroth16Driver>()
        {
            panic!("Invalid driver: expected PlainGroth16Driver");
        }

        // SAFETY: At this point we know the shares are safe to transmute
        let shares =
            unsafe { transmute::<&[T::ArithmeticHalfShare], &[B::ArkScalarField]>(shares) };

        let shares_icicle = from_host_slice_async(shares, stream);
        ark_to_icicle_scalars(shares_icicle, stream).unwrap()
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
        // This synchronize is load-bearing, not redundant: `a`/`b` are read here on `stream`,
        // but the caller (`reduction.rs`) immediately follows this call with in-place writes to
        // the same buffers on *different* streams (`ifft_in_place` on `stream_a`/`stream_b`).
        // Without blocking here, that write can be issued to the GPU before this read has
        // actually completed — CUDA gives no ordering guarantee across streams — corrupting the
        // input. See git history: `14f5ac7 fix: missing synchronization checkpoint`.
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
        _: &N,
        _: &mut Self::State,
    ) -> eyre::Result<Self::ArithmeticShare> {
        let mut rng = thread_rng();
        let res = B::ArkScalarField::rand(&mut rng);
        Ok(ark_to_icicle_scalar(res))
    }

    fn open_two_half_points_g1<N: Network, B: ArkIcicleBridge<IcicleScalarField = F>>(
        a: Affine<B::IcicleG1>,
        b: Affine<B::IcicleG1>,
        _: &N,
        _: &mut Self::State,
    ) -> eyre::Result<(Affine<B::IcicleG1>, Affine<B::IcicleG1>)> {
        Ok((a, b))
    }

    fn open_two_half_points_g1g2<N: Network, B: ArkIcicleBridge<IcicleScalarField = F>>(
        a: Affine<B::IcicleG1>,
        b: Affine<B::IcicleG2>,
        _: &N,
        _: &mut Self::State,
    ) -> eyre::Result<(Affine<B::IcicleG1>, Affine<B::IcicleG2>)> {
        Ok((a, b))
    }

    fn open_device_shares<N: Network, B: ArkIcicleBridge<IcicleScalarField = F>>(
        shares: &Self::DeviceShares,
        _: &N,
        _: &mut Self::State,
    ) -> eyre::Result<Vec<B::ArkScalarField>> {
        Ok(to_host_vec_icicle_scalar(shares)
            .into_par_iter()
            .with_min_len(1024)
            .map(icicle_to_ark_scalar)
            .collect::<Vec<_>>())
    }

    fn open_device_half_shares<N: Network, B: ArkIcicleBridge<IcicleScalarField = F>>(
        shares: &DeviceVec<F>,
        _: &N,
        _: &mut Self::State,
    ) -> eyre::Result<Vec<B::ArkScalarField>> {
        Ok(to_host_vec_icicle_scalar(shares)
            .into_par_iter()
            .with_min_len(1024)
            .map(icicle_to_ark_scalar)
            .collect::<Vec<_>>())
    }
}
