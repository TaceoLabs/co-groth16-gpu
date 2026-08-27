use std::{mem::transmute, ops::IndexMut};

use ark_ec::CurveGroup;
use icicle_core::{
    curve::{Affine, Curve},
    ntt::NTT,
    traits::{Arithmetic, FieldImpl, MontgomeryConvertible},
    vec_ops::{VecOps, VecOpsConfig, add_scalars, mul_scalars},
};
use icicle_runtime::{
    memory::{DeviceSlice, DeviceVec, HostOrDeviceSlice},
    stream::IcicleStream,
};
use mpc_core::{
    MpcState,
    protocols::rep3::{Rep3PrimeFieldShare, Rep3State, arithmetic, id::PartyID, pointshare},
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
use mpc_core::protocols::rep3::network::Rep3NetworkExt;

use super::CircomGroth16Prover;

/// A Groth16 driver for REP3 secret sharing
pub struct Rep3Groth16Driver;

pub struct Rep3IcicleShare<T> {
    a: T,
    b: T,
}

pub struct Rep3IcicleShares<T> {
    a: DeviceVec<T>,
    b: DeviceVec<T>,
}
impl<F: FieldImpl<Config: VecOps<F> + NTT<F, F>> + Arithmetic + MontgomeryConvertible>
    CircomGroth16Prover<F> for Rep3Groth16Driver
{
    type ArithmeticShare = Rep3IcicleShare<F>;

    type DeviceShares = Rep3IcicleShares<F>;
    type DevicePointShares<C: Curve<ScalarField = F>> = Rep3IcicleShares<Affine<C>>;

    type State = Rep3State;

    fn to_half_share(a: &Self::ArithmeticShare) -> F {
        a.a
    }

    fn to_half_share_vec(a: Self::DeviceShares) -> DeviceVec<F> {
        // The `a` component alone is already a valid half share; move it out instead of
        // allocating a fresh device buffer and copying into it.
        a.a
    }

    fn promote_to_trivial_shares(
        id: <Self::State as MpcState>::PartyID,
        public_values: &DeviceSlice<F>,
    ) -> Self::DeviceShares {
        let mut a = DeviceVec::device_malloc(public_values.len())
            .expect("Failed to allocate device vector");
        let mut b = DeviceVec::device_malloc(public_values.len())
            .expect("Failed to allocate device vector");

        match id {
            PartyID::ID0 => {
                a.copy(public_values).unwrap();
                b.memset(0, public_values.len()).unwrap();
            }
            PartyID::ID1 => {
                a.memset(0, public_values.len()).unwrap();
                b.copy(public_values).unwrap();
            }
            PartyID::ID2 => {
                a.memset(0, public_values.len()).unwrap();
                b.memset(0, public_values.len()).unwrap();
            }
        }

        Self::DeviceShares { a, b }
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
        let a_in: &DeviceSlice<F> = unsafe { &*(&*coeffs.a as *const DeviceSlice<F>) };
        mul_scalars(a_in, roots, coeffs.a.as_mut_slice(), &cfg).unwrap();
        // SAFETY: as above
        let b_in: &DeviceSlice<F> = unsafe { &*(&*coeffs.b as *const DeviceSlice<F>) };
        mul_scalars(b_in, roots, coeffs.b.as_mut_slice(), &cfg).unwrap();
    }

    fn add_assign_points_public_hs<C: Curve<ScalarField = F>>(
        id: <Self::State as MpcState>::PartyID,
        a: &mut Affine<C>,
        b: &Affine<C>,
    ) {
        if matches!(id, PartyID::ID0) {
            *a = (a.to_projective() + b.to_projective()).into();
        }
    }

    fn fft_in_place(input: &mut Self::DeviceShares, stream: &IcicleStream, coset_gen: Option<F>) {
        fft_inplace(&mut input.a, stream, coset_gen);
        fft_inplace(&mut input.b, stream, coset_gen);
    }

    fn ifft_in_place(input: &mut Self::DeviceShares, stream: &IcicleStream, coset_gen: Option<F>) {
        ifft_inplace(&mut input.a, stream, coset_gen);
        ifft_inplace(&mut input.b, stream, coset_gen);
    }

    fn copy_to_device_shares(
        src: &Self::DeviceShares,
        dst: &mut Self::DeviceShares,
        start: usize,
        end: usize,
    ) {
        dst.a.index_mut(start..end).copy(&src.a).unwrap();
        dst.b.index_mut(start..end).copy(&src.b).unwrap();
    }

    fn shares_to_device<
        B: ArkIcicleBridge<IcicleScalarField = F>,
        T: co_groth16::CircomGroth16Prover<B::ArkPairing> + 'static,
    >(
        shares: &[T::ArithmeticShare],
    ) -> Self::DeviceShares {
        if std::any::TypeId::of::<T>()
            != std::any::TypeId::of::<co_groth16::mpc::Rep3Groth16Driver>()
        {
            panic!("Invalid driver: expected Rep3Groth16Driver");
        }

        // SAFETY: At this point we know the shares are safe to transmute
        let shares = unsafe {
            transmute::<&[T::ArithmeticShare], &[Rep3PrimeFieldShare<B::ArkScalarField>]>(shares)
        };

        let (shares_a, shares_b): (Vec<B::ArkScalarField>, Vec<B::ArkScalarField>) =
            shares.iter().map(|s| (s.a, s.b)).unzip();

        let shares_a = from_host_slice(&shares_a);
        let shares_b = from_host_slice(&shares_b);

        let a = ark_to_icicle_scalars(shares_a).unwrap();
        let b = ark_to_icicle_scalars(shares_b).unwrap();

        Self::DeviceShares { a, b }
    }

    fn half_shares_to_device<
        B: ArkIcicleBridge<IcicleScalarField = F>,
        T: co_groth16::CircomGroth16Prover<B::ArkPairing> + 'static,
    >(
        shares: &[T::ArithmeticHalfShare],
    ) -> DeviceVec<F> {
        if std::any::TypeId::of::<T>()
            != std::any::TypeId::of::<co_groth16::mpc::Rep3Groth16Driver>()
        {
            panic!("Invalid driver: expected Rep3Groth16Driver");
        }

        // SAFETY: At this point we know the shares are safe to transmute
        let shares =
            unsafe { transmute::<&[T::ArithmeticHalfShare], &[B::ArkScalarField]>(shares) };

        let shares = from_host_slice(shares);

        ark_to_icicle_scalars(shares).unwrap()
    }

    fn local_mul_vec<B: ArkIcicleBridge<IcicleScalarField = F>>(
        a: &Self::DeviceShares,
        b: &Self::DeviceShares,
        state: &mut Self::State,
        stream: &IcicleStream,
    ) -> DeviceVec<F> {
        let masking_fes = state
            .rngs
            .rand
            .masking_field_elements_vec::<B::ArkScalarField>(a.a.len());
        let masking_fes = from_host_slice(&masking_fes);
        let masking_fes: DeviceVec<F> =
            ark_to_icicle_scalars::<B::ArkScalarField, F>(masking_fes).unwrap();

        let mut tmp0 = DeviceVec::device_malloc_async(a.a.len(), stream)
            .expect("Failed to allocate device vector");
        let mut tmp1 = DeviceVec::device_malloc_async(a.a.len(), stream)
            .expect("Failed to allocate device vector");
        let mut tmp2 = DeviceVec::device_malloc_async(a.a.len(), stream)
            .expect("Failed to allocate device vector");

        let mut cfg = VecOpsConfig::default();
        cfg.stream_handle = **stream;
        cfg.is_async = true;
        mul_scalars(&a.a, &b.a, tmp0.as_mut_slice(), &cfg).unwrap();
        mul_scalars(&a.a, &b.b, tmp1.as_mut_slice(), &cfg).unwrap();
        mul_scalars(&a.b, &b.a, tmp2.as_mut_slice(), &cfg).unwrap();

        let mut result = DeviceVec::device_malloc_async(a.b.len(), stream)
            .expect("Failed to allocate device vector");

        add_scalars(&tmp0, &tmp1, result.as_mut_slice(), &cfg).unwrap();
        add_scalars(&tmp2, &result, tmp0.as_mut_slice(), &cfg).unwrap();
        add_scalars(&tmp0, &masking_fes, result.as_mut_slice(), &cfg).unwrap();
        stream
            .synchronize()
            .expect("Failed to synchronize local_mul_vec stream");

        result
    }

    fn local_mul<B: ArkIcicleBridge<IcicleScalarField = F>>(
        a: &Self::ArithmeticShare,
        b: &Self::ArithmeticShare,
        state: &mut Self::State,
    ) -> F {
        let masking_fe = state.rngs.rand.masking_field_element::<B::ArkScalarField>();
        let masking_fe = ark_to_icicle_scalar(masking_fe);
        a.a * b.a + a.a * b.b + a.b * b.a + masking_fe
    }

    fn rand<N: Network, B: ArkIcicleBridge<IcicleScalarField = F>>(
        _: &N,
        state: &mut Self::State,
    ) -> eyre::Result<Self::ArithmeticShare> {
        let res = arithmetic::rand::<B::ArkScalarField>(state);
        Ok(Self::ArithmeticShare {
            a: ark_to_icicle_scalar(res.a),
            b: ark_to_icicle_scalar(res.b),
        })
    }

    fn open_two_half_points_g1<N: Network, B: ArkIcicleBridge<IcicleScalarField = F>>(
        a: Affine<B::IcicleG1>,
        b: Affine<B::IcicleG1>,
        net: &N,
        _: &mut Self::State,
    ) -> eyre::Result<(Affine<B::IcicleG1>, Affine<B::IcicleG1>)> {
        let ark_a = B::icicle_to_ark_g1(a);
        let ark_b = B::icicle_to_ark_g1(b);
        let (open_a, open_b) = pointshare::open_two_half_points(ark_a.into(), ark_b.into(), net)?;
        Ok((
            ark_to_icicle_affine(&open_a.into_affine()),
            ark_to_icicle_affine(&open_b.into_affine()),
        ))
    }

    fn open_two_half_points_g1g2<N: Network, B: ArkIcicleBridge<IcicleScalarField = F>>(
        a: Affine<B::IcicleG1>,
        b: Affine<B::IcicleG2>,
        net: &N,
        _: &mut Self::State,
    ) -> eyre::Result<(Affine<B::IcicleG1>, Affine<B::IcicleG2>)> {
        let ark_a = B::icicle_to_ark_g1(a);
        let ark_b = B::icicle_to_ark_g2(b);
        let (open_a, open_b) = pointshare::open_two_half_points(ark_a.into(), ark_b.into(), net)?;
        Ok((
            ark_to_icicle_affine(&open_a.into_affine()),
            ark_to_icicle_affine(&open_b.into_affine()),
        ))
    }

    // TODO CESAR: remove
    fn open_device_shares<N: Network, B: ArkIcicleBridge<IcicleScalarField = F>>(
        shares: &Self::DeviceShares,
        net: &N,
        _: &mut Self::State,
    ) -> eyre::Result<Vec<B::ArkScalarField>> {
        let host_a = to_host_vec_icicle_scalar(&shares.a)
            .into_par_iter()
            .with_min_len(1024)
            .map(icicle_to_ark_scalar::<B::ArkScalarField, _>)
            .collect::<Vec<_>>();
        let host_b = to_host_vec_icicle_scalar(&shares.b)
            .into_par_iter()
            .with_min_len(1024)
            .map(icicle_to_ark_scalar::<B::ArkScalarField, _>)
            .collect::<Vec<_>>();

        let shares = host_a
            .into_par_iter()
            .with_min_len(1024)
            .zip(host_b)
            .map(|(a, b)| Rep3PrimeFieldShare { a, b })
            .collect::<Vec<_>>();
        let opened = arithmetic::open_vec(&shares, net).unwrap();

        Ok(opened)
    }

    fn open_device_half_shares<N: Network, B: ArkIcicleBridge<IcicleScalarField = F>>(
        shares: &DeviceVec<F>,
        net: &N,
        _: &mut Self::State,
    ) -> eyre::Result<Vec<B::ArkScalarField>> {
        let host_a = to_host_vec_icicle_scalar(shares)
            .into_par_iter()
            .with_min_len(1024)
            .map(icicle_to_ark_scalar::<B::ArkScalarField, _>)
            .collect::<Vec<_>>();

        let (b, c) = net.broadcast_many(&host_a)?;

        Ok(host_a
            .into_par_iter()
            .with_min_len(1024)
            .zip(b)
            .zip(c)
            .map(|((a, b), c)| a + b + c)
            .collect::<Vec<_>>())
    }
}
