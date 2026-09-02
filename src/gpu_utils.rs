use std::ops::IndexMut;

use ark_ff::FftField;
use ark_poly::EvaluationDomain;
use ark_poly::GeneralEvaluationDomain;
use icicle_core::curve::Affine;
use icicle_core::vec_ops::VecOps;
use icicle_core::{
    curve::{Curve, Projective},
    msm::{CUDA_MSM_LARGE_BUCKET_FACTOR, MSM, MSMConfig, msm, precompute_bases},
    ntt::{self, NTT, NTTConfig, NTTDir, NTTDomain, ntt_inplace},
    traits::{Arithmetic, FieldImpl, MontgomeryConvertible},
};
use icicle_runtime::{
    memory::{DeviceSlice, DeviceVec, HostOrDeviceSlice, HostSlice},
    stream::IcicleStream,
};

pub const PRECOMPUTE_FACTOR_G1: i32 = 8;
pub const PRECOMPUTE_FACTOR_G2: i32 = 8;
pub const C: i32 = 0; // Lets icicle auto-pick
pub const LARGE_BUCKET_FACTOR: i32 = 5;

#[macro_export]
macro_rules! rayon_join_5 {
    ($f1:expr, $f2:expr, $f3:expr, $f4:expr, $f5:expr $(,)?) => {{
        use rayon::join;

        let ((a, b), ((c, d), e)) = join(
            || join(|| $f1(), || $f2()),
            || join(|| join(|| $f3(), || $f4()), || $f5()),
        );

        (a, b, c, d, e)
    }};
}

use crate::bridges::{ArkIcicleBridge, ark_to_icicle_affine, ark_to_icicle_scalar};
use crate::utils::root_of_unity_for_groth16;

fn upload_points_async<C: Curve + MSM<C>>(
    points: &[Affine<C>],
    stream: &IcicleStream,
    precompute_factor: i32,
) -> DeviceVec<Affine<C>> {
    assert!(
        !points.is_empty(),
        "MSM query slice cannot be empty for this prover flow"
    );

    let precompute_factor = precompute_factor.max(1);
    if precompute_factor == 1 {
        return from_host_slice_async(points, stream);
    }

    let mut cfg = MSMConfig::default();
    cfg.stream_handle = **stream;
    cfg.is_async = true;
    cfg.precompute_factor = precompute_factor;
    cfg.c = C;
    cfg.ext
        .set_int(CUDA_MSM_LARGE_BUCKET_FACTOR, LARGE_BUCKET_FACTOR);
    let mut precomputed =
        DeviceVec::device_malloc_async(points.len() * (precompute_factor as usize), stream)
            .expect("Failed to allocate precomputed MSM bases");
    precompute_bases::<C>(
        HostSlice::from_slice(points),
        &cfg,
        precomputed.as_mut_slice(),
    )
    .expect("Failed to precompute MSM bases");
    precomputed
}

/// Returns the cached device buffer, (re)allocating it if it is missing or its
/// length does not match `len`.
pub(crate) fn get_or_alloc<T>(buf: &mut Option<DeviceVec<T>>, len: usize) -> &mut DeviceVec<T> {
    if buf.as_ref().is_none_or(|b| b.len() != len) {
        *buf = Some(DeviceVec::device_malloc(len).expect("Failed to allocate device vector"));
    }
    buf.as_mut().unwrap()
}

pub fn from_host_slice<T>(slice: &[T]) -> DeviceVec<T> {
    let count = slice.len();
    let mut result = DeviceVec::device_malloc(count).expect("Failed to allocate device vector");
    result
        .copy_from_host(HostSlice::from_slice(slice))
        .expect("Failed to copy data from host to device");
    result
}

pub fn from_host_slice_async<T>(slice: &[T], stream: &IcicleStream) -> DeviceVec<T> {
    let count = slice.len();
    let mut result =
        DeviceVec::device_malloc_async(count, stream).expect("Failed to allocate device vector");
    result
        .copy_from_host_async(HostSlice::from_slice(slice), stream)
        .expect("Failed to copy data from host to device");
    result
}

pub fn to_host_vec_icicle_scalar<F: FieldImpl>(slice: &DeviceSlice<F>) -> Vec<F> {
    let mut host_vec = vec![F::zero(); slice.len()];
    let host_slice = HostSlice::from_mut_slice(&mut host_vec);
    slice.copy_to_host(host_slice).unwrap();
    host_vec
}

pub fn get_first<C: Curve>(vec: &DeviceVec<Projective<C>>) -> Projective<C> {
    let mut result = [Projective::<C>::zero(); 1];
    let host_slice = HostSlice::from_mut_slice(&mut result);
    vec.copy_to_host(host_slice)
        .expect("Failed to copy data from device to host");
    result[0]
}

pub(crate) struct Proof<
    F: FieldImpl<Config: VecOps<F> + NTT<F, F>>,
    C1: Curve<ScalarField = F>,
    C2: Curve<ScalarField = F>,
> {
    /// The `A` element in `G1`.
    pub a: Affine<C1>,
    /// The `B` element in `G2`.
    pub b: Affine<C2>,
    /// The `C` element in `G1`.
    pub c: Affine<C1>,
}

impl<
    F: FieldImpl<Config: VecOps<F> + NTT<F, F>>,
    C1: Curve<ScalarField = F>,
    C2: Curve<ScalarField = F>,
> Proof<F, C1, C2>
{
    pub(crate) fn to_ark<
        B: ArkIcicleBridge<IcicleG1 = C1, IcicleG2 = C2, IcicleScalarField = F>,
    >(
        &self,
    ) -> ark_groth16::Proof<B::ArkPairing> {
        ark_groth16::Proof {
            a: B::icicle_to_ark_g1(self.a),
            b: B::icicle_to_ark_g2(self.b),
            c: B::icicle_to_ark_g1(self.c),
        }
    }
}

pub(crate) struct VerifyingKey<
    F: FieldImpl<Config: VecOps<F> + NTT<F, F>>,
    C1: Curve<ScalarField = F>,
    C2: Curve<ScalarField = F>,
> {
    /// The `alpha * G`, where `G` is the generator of `E::G1`.
    pub(crate) alpha_g1: Affine<C1>,
    /// The `alpha * H`, where `H` is the generator of `E::G2`.
    pub(crate) beta_g2: Affine<C2>,
    /// The `delta * H`, where `H` is the generator of `E::G2`.
    pub(crate) delta_g2: Affine<C2>,
}

pub struct ProvingKey<
    F: FieldImpl<Config: VecOps<F> + NTT<F, F>>,
    C1: Curve<ScalarField = F>,
    C2: Curve<ScalarField = F>,
> {
    /// The underlying verification key.
    pub(crate) vk: VerifyingKey<F, C1, C2>,
    /// The element `beta * G` in `E::G1`.
    pub(crate) beta_g1: Affine<C1>,
    /// The element `delta * G` in `E::G1`.
    pub(crate) delta_g1: Affine<C1>,
    /// The first `a_i * G` query element used in A commitment.
    pub(crate) a_query_first: Affine<C1>,
    /// The first `b_i * G` query element used in B(G1) commitment.
    pub(crate) b_g1_query_first: Affine<C1>,
    /// The first `b_i * H` query element used in B(G2) commitment.
    pub(crate) b_g2_query_first: Affine<C2>,
    /// The public slice of `a_query` excluding index 0.
    pub(crate) a_query_pub: DeviceVec<Affine<C1>>,
    /// The private-witness slice of `a_query`.
    pub(crate) a_query_priv: DeviceVec<Affine<C1>>,
    /// The public slice of `b_g1_query` excluding index 0.
    pub(crate) b_g1_query_pub: DeviceVec<Affine<C1>>,
    /// The private-witness slice of `b_g1_query`.
    pub(crate) b_g1_query_priv: DeviceVec<Affine<C1>>,
    /// The public slice of `b_g2_query` excluding index 0.
    pub(crate) b_g2_query_pub: DeviceVec<Affine<C2>>,
    /// The private-witness slice of `b_g2_query`.
    pub(crate) b_g2_query_priv: DeviceVec<Affine<C2>>,
    /// The elements `h_i * G` in `E::G1`.
    pub(crate) h_query: DeviceVec<Affine<C1>>,
    /// The elements `l_i * G` in `E::G1`.
    pub(crate) l_query: DeviceVec<Affine<C1>>,
    pub(crate) domain_size: usize,
    pub(crate) precomputed_roots: DeviceVec<F>,
    pub(crate) num_constraints: usize,
}

pub struct ProofStreams {
    pub g1: IcicleStream,
    pub g2: IcicleStream,
}

impl ProofStreams {
    pub(crate) fn new() -> Self {
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

impl<
    F: FieldImpl<Config: VecOps<F> + NTT<F, F>> + Arithmetic + MontgomeryConvertible + 'static,
    C1: Curve<ScalarField = F> + MSM<C1>,
    C2: Curve<ScalarField = F> + MSM<C2>,
> ProvingKey<F, C1, C2>
{
    pub(crate) fn from_ark<P: ark_ec::pairing::Pairing>(
        pk: &ark_groth16::ProvingKey<P>,
        num_constraints: usize,
        num_instance_variables: usize,
        eval_c: bool,
    ) -> Self {
        let alpha_g1 = ark_to_icicle_affine(&pk.vk.alpha_g1);
        let beta_g2 = ark_to_icicle_affine(&pk.vk.beta_g2);
        let delta_g2 = ark_to_icicle_affine(&pk.vk.delta_g2);

        let beta_g1 = ark_to_icicle_affine(&pk.beta_g1);
        let delta_g1 = ark_to_icicle_affine(&pk.delta_g1);

        let (a_query, b_g1_query, b_g2_query, h_query_host, l_query_host) = rayon_join_5!(
            || pk
                .a_query
                .iter()
                .map(ark_to_icicle_affine)
                .collect::<Vec<_>>(),
            || pk
                .b_g1_query
                .iter()
                .map(ark_to_icicle_affine)
                .collect::<Vec<_>>(),
            || pk
                .b_g2_query
                .iter()
                .map(ark_to_icicle_affine)
                .collect::<Vec<_>>(),
            || pk
                .h_query
                .iter()
                .map(ark_to_icicle_affine)
                .collect::<Vec<_>>(),
            || pk
                .l_query
                .iter()
                .map(ark_to_icicle_affine)
                .collect::<Vec<_>>(),
        );

        assert!(
            num_instance_variables > 0,
            "num_instance_variables must be > 0"
        );
        assert!(
            num_instance_variables <= a_query.len()
                && num_instance_variables <= b_g1_query.len()
                && num_instance_variables <= b_g2_query.len(),
            "Proving key query vectors are shorter than num_instance_variables"
        );

        let a_query_first = a_query[0];
        let b_g1_query_first = b_g1_query[0];
        let b_g2_query_first = b_g2_query[0];

        let a_query_pub_host = a_query[1..num_instance_variables].to_vec();
        let a_query_priv_host = a_query[num_instance_variables..].to_vec();
        let b_g1_query_pub_host = b_g1_query[1..num_instance_variables].to_vec();
        let b_g1_query_priv_host = b_g1_query[num_instance_variables..].to_vec();
        let b_g2_query_pub_host = b_g2_query[1..num_instance_variables].to_vec();
        let b_g2_query_priv_host = b_g2_query[num_instance_variables..].to_vec();

        let mut streams = (0..8)
            .map(|_| IcicleStream::create().unwrap())
            .collect::<Vec<_>>();

        let a_query_pub = upload_points_async(&a_query_pub_host, &streams[0], PRECOMPUTE_FACTOR_G1);
        let a_query_priv =
            upload_points_async(&a_query_priv_host, &streams[1], PRECOMPUTE_FACTOR_G1);
        let b_g1_query_pub =
            upload_points_async(&b_g1_query_pub_host, &streams[2], PRECOMPUTE_FACTOR_G1);
        let b_g1_query_priv =
            upload_points_async(&b_g1_query_priv_host, &streams[3], PRECOMPUTE_FACTOR_G1);
        let b_g2_query_pub =
            upload_points_async(&b_g2_query_pub_host, &streams[4], PRECOMPUTE_FACTOR_G2);
        let b_g2_query_priv =
            upload_points_async(&b_g2_query_priv_host, &streams[5], PRECOMPUTE_FACTOR_G2);
        let h_query = upload_points_async(&h_query_host, &streams[6], PRECOMPUTE_FACTOR_G1);
        let l_query = upload_points_async(&l_query_host, &streams[7], PRECOMPUTE_FACTOR_G1);

        streams.iter_mut().for_each(|stream| {
            stream.synchronize().unwrap();
            stream.destroy().unwrap();
        });

        let mut domain = GeneralEvaluationDomain::<P::ScalarField>::new(
            num_constraints + num_instance_variables,
        )
        .unwrap();
        let domain_size = domain.size();
        initialize_domain::<F>(domain_size);
        let power = domain_size.ilog2() as usize;

        let root_of_unity = if eval_c {
            P::ScalarField::GENERATOR
        } else {
            root_of_unity_for_groth16::<P::ScalarField>(power, &mut domain)
        };
        let root_of_unity = ark_to_icicle_scalar(root_of_unity);
        let mut roots = Vec::with_capacity(domain_size);
        let mut c = F::one();
        for _ in 0..domain_size {
            roots.push(c);
            c = c * root_of_unity;
        }
        let precomputed_roots = from_host_slice(&roots);

        Self {
            vk: VerifyingKey {
                alpha_g1,
                beta_g2,
                delta_g2,
            },
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
            h_query,
            l_query,
            domain_size,
            precomputed_roots,
            num_constraints,
        }
    }
}

pub(crate) fn initialize_domain<F: FieldImpl<Config: NTTDomain<F>> + 'static>(max_size: usize) {
    let res = ntt::release_domain::<F>();
    match res {
        Ok(_) => (),
        Err(e) => {
            tracing::info!("Warning: Failed to release existing NTT domain: {e}");
        }
    }

    ntt::initialize_domain(
        ntt::get_root_of_unity::<F>(max_size.try_into().unwrap()),
        &ntt::NTTInitDomainConfig::default(),
    )
    .unwrap();
}

pub(crate) fn fft_inplace<F: FieldImpl<Config: VecOps<F> + NTT<F, F>>>(
    input: &mut DeviceSlice<F>,
    stream: &IcicleStream,
    coset_gen: Option<F>,
) {
    let mut ntt_config = NTTConfig::<F>::default();
    ntt_config.stream_handle = **stream;
    ntt_config.is_async = true;
    ntt_config.coset_gen = coset_gen.unwrap_or_else(|| F::one());

    ntt_inplace(input, NTTDir::kForward, &ntt_config).expect("Failed to compute FFT in place");
}

pub(crate) fn ifft_inplace<F: FieldImpl<Config: VecOps<F> + NTT<F, F>>>(
    input: &mut DeviceSlice<F>,
    stream: &IcicleStream,
    coset_gen: Option<F>,
) {
    let mut ntt_config = NTTConfig::<F>::default();
    ntt_config.stream_handle = **stream;
    ntt_config.is_async = true;
    ntt_config.coset_gen = coset_gen.unwrap_or_else(|| F::one());

    ntt_inplace(input, NTTDir::kInverse, &ntt_config)
        .expect("Failed to compute inverse FFT in place");
}

pub(crate) fn msm_async<
    F: FieldImpl<Config: VecOps<F> + NTT<F, F>>,
    C: Curve<ScalarField = F> + MSM<C>,
>(
    points: &DeviceSlice<Affine<C>>,
    scalars: &DeviceSlice<F>,
    stream: &IcicleStream,
    precompute_factor: i32,
) -> DeviceVec<Projective<C>> {
    let mut results: DeviceVec<Projective<C>> =
        DeviceVec::device_malloc_async(1, stream).expect("Failed to allocate device vector");
    let mut cfg = MSMConfig::default();
    cfg.stream_handle = **stream;
    cfg.is_async = true;
    cfg.precompute_factor = precompute_factor.max(1);
    cfg.c = C;
    cfg.ext
        .set_int(CUDA_MSM_LARGE_BUCKET_FACTOR, LARGE_BUCKET_FACTOR);

    msm::<C>(scalars, points, &cfg, results.index_mut(..)).expect("Failed to compute MSM");
    results
}
