//! The witness map NTTs of [`CircomReduction`], on the device: the inverse NTT leaves the
//! coefficients bit-reversed, the forward one takes them so, and the coset scaling happens in
//! between, so there is no reordering. Kernels in `src/cuda/ntt.cu`.
//!
//! [`CircomReduction`]: crate::groth16_gpu::reduction::CircomReduction

use ark_ff::{FftField, Field, PrimeField};
use icicle_core::traits::{FieldImpl, MontgomeryConvertible};
use icicle_runtime::{
    memory::{DeviceSlice, DeviceVec, HostOrDeviceSlice, HostSlice},
    stream::IcicleStream,
};

use crate::{
    bridges::ark_to_icicle_scalar,
    cuda::{Args, Module, device_ptr, device_ptr_mut},
};

static NTT: Module = Module::new(
    concat!(include_str!("cuda/ntt.ptx"), "\0"),
    &[
        c"ntt_inverse_pass",
        c"ntt_forward_pass",
        c"ntt_coset_pass",
        c"ntt_mul",
        c"ntt_mul_sub",
    ],
);
const INVERSE_PASS: usize = 0;
const FORWARD_PASS: usize = 1;
const COSET_PASS: usize = 2;
const MUL: usize = 3;
const MUL_SUB: usize = 4;

/// As in `ntt.cu`.
const TILE_LOG: u32 = 10;
const TW_LOG: u32 = 12;
const THREADS: u32 = 256;
const MAX_LOG_N: u32 = 2 * TW_LOG;

/// `base^t` for `t < 2^TW_LOG` and `base^(t * 2^TW_LOG)` for `t < 2^TW_LOG`, scaled by `scale`.
fn split_table<T: Field>(base: T, scale: T) -> [Vec<T>; 2] {
    let lo = std::iter::successors(Some(scale), |x| Some(*x * base))
        .take(1 << TW_LOG)
        .collect();
    let step = base.pow([1u64 << TW_LOG]);
    let hi = std::iter::successors(Some(T::one()), |x| Some(*x * step))
        .take(1 << TW_LOG)
        .collect();
    [lo, hi]
}

fn upload<T: PrimeField, F: FieldImpl + MontgomeryConvertible>(host: &[T]) -> DeviceVec<F> {
    let host = host
        .iter()
        .map(|x| ark_to_icicle_scalar::<T, F>(*x))
        .collect::<Vec<_>>();
    let mut dev = DeviceVec::device_malloc(host.len()).expect("Failed to allocate device vector");
    dev.copy_from_host(HostSlice::from_slice(&host))
        .expect("Failed to upload twiddles");
    dev
}

/// The witness map NTTs for one domain size, with their twiddles.
pub(crate) struct CosetNtt<F> {
    log_n: u32,
    /// The bits of the index of every pass, from the top; the last one is the coset pass.
    passes: Vec<u32>,
    /// Inverse and forward twiddles.
    sub: [DeviceVec<F>; 2],
    tw: [[DeviceVec<F>; 2]; 2],
    /// `g^m / N`.
    coset: [DeviceVec<F>; 2],
}

impl<F: FieldImpl + MontgomeryConvertible> CosetNtt<F> {
    /// For the domain of size `2^log_n` with root `coset_gen^2`; `None` if it is too large.
    pub(crate) fn new<T: PrimeField + FftField>(log_n: u32, coset_gen: T) -> Option<Self> {
        if !(1..=MAX_LOG_N).contains(&log_n) {
            return None;
        }
        let omega = coset_gen.square();
        let omega_inv = omega.inverse().unwrap();
        // Powers of the 2^TILE_LOG-th root of unity (those the kernels read, for small domains).
        let sub = |root: T| {
            (0..1u64 << (TILE_LOG - 1))
                .map(|e| {
                    if log_n >= TILE_LOG {
                        root.pow([e << (log_n - TILE_LOG)])
                    } else if e % (1 << (TILE_LOG - log_n)) == 0 {
                        root.pow([e >> (TILE_LOG - log_n)])
                    } else {
                        T::zero()
                    }
                })
                .collect::<Vec<_>>()
        };
        let n_inv = T::from(1u64 << log_n).inverse().unwrap();
        let passes = log_n.div_ceil(TILE_LOG);
        let passes = (0..passes)
            .map(|i| log_n / passes + u32::from(i < log_n % passes))
            .collect();
        let upload2 = |[lo, hi]: [Vec<T>; 2]| [upload(&lo), upload(&hi)];
        Some(Self {
            log_n,
            passes,
            sub: [upload(&sub(omega_inv)), upload(&sub(omega))],
            tw: [
                upload2(split_table(omega_inv, T::one())),
                upload2(split_table(omega, T::one())),
            ],
            coset: upload2(split_table(coset_gen, n_inv)),
        })
    }

    /// `h = a * b - c` on the coset, from `a`, `b` and `c = a * b` on the domain. Overwrites `a`
    /// and `b`; `c` is scratch.
    pub(crate) fn witness_map(
        &self,
        a: &mut DeviceSlice<F>,
        b: &mut DeviceSlice<F>,
        c: &mut DeviceSlice<F>,
        h: &mut DeviceSlice<F>,
        stream: &IcicleStream,
    ) {
        let n = 1u32 << self.log_n;
        for v in [&*a, &*b, &*c, &*h] {
            assert_eq!(v.len(), n as usize, "length mismatch");
        }
        let module = NTT.get();
        let ptr = |v: &DeviceVec<F>| device_ptr(v);
        let polys = [device_ptr_mut(a), device_ptr_mut(b), device_ptr_mut(c)];
        let h = device_ptr_mut(h);
        let with_polys = || polys.iter().fold(Args::new(), |args, p| args.arg(*p));
        let log_n = self.log_n;
        let elementwise = (n.div_ceil(THREADS), 1);
        // SAFETY: the arguments match the kernel signatures in `ntt.cu`, and every kernel stays
        // within the buffers, which all have n elements.
        let launch = |kernel: usize, grid: (u32, u32), args: Args| unsafe {
            module.launch_args(kernel, grid, THREADS, stream, args)
        };
        // The pass over bits [lo, lo + r), in direction `dir` (0: inverse, 1: forward).
        let strided = |kernel: usize, dir: usize, i: usize| {
            let r = self.passes[i];
            let lo: u32 = self.passes[i + 1..].iter().sum();
            let log_g = (TILE_LOG - r).min(lo);
            launch(
                kernel,
                (1 << (log_n - r - log_g), 3),
                with_polys()
                    .arg(log_n)
                    .arg(lo)
                    .arg(r)
                    .arg(ptr(&self.sub[dir]))
                    .arg(ptr(&self.tw[dir][0]))
                    .arg(ptr(&self.tw[dir][1])),
            );
        };

        launch(MUL, elementwise, with_polys().arg(n));
        let last = self.passes.len() - 1;
        for i in 0..last {
            strided(INVERSE_PASS, 0, i);
        }
        launch(
            COSET_PASS,
            (n >> log_n.min(TILE_LOG), 3),
            with_polys()
                .arg(log_n)
                .arg(self.passes[last])
                .arg(ptr(&self.sub[0]))
                .arg(ptr(&self.sub[1]))
                .arg(ptr(&self.coset[0]))
                .arg(ptr(&self.coset[1])),
        );
        for i in (0..last).rev() {
            strided(FORWARD_PASS, 1, i);
        }
        launch(MUL_SUB, elementwise, with_polys().arg(h).arg(n));
    }
}

#[cfg(test)]
mod tests {
    use ark_bn254::Fr;
    use ark_ff::{BigInteger, PrimeField, UniformRand};
    use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};
    use icicle_bn254::curve::ScalarField;
    use icicle_core::traits::FieldImpl;
    use icicle_runtime::{
        memory::{DeviceVec, HostSlice},
        stream::IcicleStream,
    };
    use rayon::prelude::*;

    use super::CosetNtt;
    use crate::{bridges::ark_scalars_to_device_into, utils::root_of_unity_for_groth16};

    #[test]
    #[ignore = "needs a GPU"]
    fn witness_map_matches_host() {
        crate::load_backend_from_env_and_set_device(0);
        let mut rng = rand::thread_rng();
        let stream = IcicleStream::create().unwrap();
        for log_n in [1u32, 2, 5, 10, 11, 16, 17, 20, 21] {
            let n = 1usize << log_n;
            let mut domain = GeneralEvaluationDomain::<Fr>::new(n).unwrap();
            let g = root_of_unity_for_groth16(log_n as usize, &mut domain);
            let a = (0..n).map(|_| Fr::rand(&mut rng)).collect::<Vec<_>>();
            let b = (0..n).map(|_| Fr::rand(&mut rng)).collect::<Vec<_>>();
            let on_coset = |mut v: Vec<Fr>| {
                domain.ifft_in_place(&mut v);
                let mut x = Fr::from(1u64);
                for v in v.iter_mut() {
                    *v *= x;
                    x *= g;
                }
                domain.fft_in_place(&mut v);
                v
            };
            let c = a.par_iter().zip(&b).map(|(a, b)| *a * b).collect();
            let (ac, bc, cc) = (on_coset(a.clone()), on_coset(b.clone()), on_coset(c));

            let ntt = CosetNtt::<ScalarField>::new(log_n, g).unwrap();
            let alloc = || DeviceVec::<ScalarField>::device_malloc(n).unwrap();
            let (mut da, mut db, mut dc, mut dh) = (alloc(), alloc(), alloc(), alloc());
            ark_scalars_to_device_into(&a, &mut da);
            ark_scalars_to_device_into(&b, &mut db);
            ntt.witness_map(&mut da, &mut db, &mut dc, &mut dh, &stream);
            let mut h = vec![ScalarField::zero(); n];
            dh.copy_to_host(HostSlice::from_mut_slice(&mut h)).unwrap();
            assert!(
                (0..n)
                    .all(|i| h[i].to_bytes_le()
                        == (ac[i] * bc[i] - cc[i]).into_bigint().to_bytes_le()),
                "mismatch for log_n = {log_n}"
            );
        }
    }
}
