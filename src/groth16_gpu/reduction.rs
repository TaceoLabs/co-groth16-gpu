use eyre::Result;
use icicle_core::vec_ops::{VecOpsConfig, mul_scalars, sub_scalars};
use icicle_runtime::{
    memory::{DeviceSlice, DeviceVec, HostOrDeviceSlice, HostSlice},
    stream::IcicleStream,
};
use mpc_core::MpcState;
use tracing::instrument;

use crate::{
    bridges::{ArkIcicleBridge, ark_to_icicle_scalar},
    gpu_utils,
    mpc::CircomGroth16Prover,
};

use ark_ff::{FftField, Field};
use num_traits::One;

/// This trait is used to convert the secret-shared witness into a secret-shared QAP witness as part of a collaborative Groth16 proof.
/// Refer to <https://docs.rs/ark-groth16/latest/ark_groth16/r1cs_to_qap/trait.R1CSToQAP.html> for more details on the plain version.
/// We do not implement the other methods of the arkworks trait, as we do not need them during proof generation.
pub trait R1CSToQAP {
    /// Computes a QAP witness corresponding to the R1CS witness defined by `private_witness`, using the provided `ConstraintMatrices`.
    /// The provided `driver` is used to perform the necessary operations on the secret-shared witness.
    fn witness_map_from_r1cs_eval<
        B: ArkIcicleBridge,
        T: CircomGroth16Prover<B::IcicleScalarField>,
    >(
        state: &mut T::State,
        eval_a: &mut T::DeviceShares,
        eval_b: &mut T::DeviceShares,
        eval_c: Option<&mut DeviceVec<B::IcicleScalarField>>,
        public_inputs: &DeviceSlice<B::IcicleScalarField>,
        roots_to_power_domain: &DeviceSlice<B::IcicleScalarField>,
        num_constraints: usize,
        domain_size: usize,
    ) -> Result<DeviceVec<B::IcicleScalarField>>;

    fn requires_eval_c() -> bool;
}

/// Implements the witness map used by snarkjs. The arkworks witness map calculates the
/// coefficients of H through computing (AB-C)/Z in the evaluation domain and going back to the
/// coefficients domain. snarkjs instead precomputes the Lagrange form of the powers of tau bases
/// in a domain twice as large and the witness map is computed as the odd coefficients of (AB-C)
/// in that domain. This serves as HZ when computing the C proof element.
///
/// Based on <https://github.com/arkworks-rs/circom-compat/>.
pub struct CircomReduction;

impl R1CSToQAP for CircomReduction {
    #[instrument(level = "debug", name = "witness map from matrices", skip_all)]
    fn witness_map_from_r1cs_eval<
        B: ArkIcicleBridge,
        T: CircomGroth16Prover<B::IcicleScalarField>,
    >(
        state: &mut T::State,
        eval_a: &mut T::DeviceShares,
        eval_b: &mut T::DeviceShares,
        eval_c: Option<&mut DeviceVec<B::IcicleScalarField>>,
        public_inputs: &DeviceSlice<B::IcicleScalarField>,
        roots_to_power_domain: &DeviceSlice<B::IcicleScalarField>,
        num_constraints: usize,
        domain_size: usize,
    ) -> Result<DeviceVec<B::IcicleScalarField>> {
        assert!(eval_c.is_none());

        let id = state.id();

        // Computation of a
        let promoted_public = T::promote_to_trivial_shares(id, public_inputs);
        T::copy_to_device_shares(&promoted_public, eval_a, num_constraints, domain_size);

        let mut stream_c = IcicleStream::create().unwrap();
        let mut c = T::local_mul_vec::<B>(eval_a, eval_b, state, &stream_c);

        // Computation of a
        let mut stream_a = IcicleStream::create().unwrap();
        T::ifft_in_place(eval_a, &stream_a, None);
        T::distribute_powers_and_mul_by_const(eval_a, roots_to_power_domain, &stream_a);
        T::fft_in_place(eval_a, &stream_a, None);

        // Computation of b
        let mut stream_b = IcicleStream::create().unwrap();
        T::ifft_in_place(eval_b, &stream_b, None);
        T::distribute_powers_and_mul_by_const(eval_b, roots_to_power_domain, &stream_b);
        T::fft_in_place(eval_b, &stream_b, None);

        // Computation of c
        gpu_utils::ifft_inplace(&mut c, &stream_c, None);
        T::distribute_powers_and_mul_by_const_hs(&mut c, roots_to_power_domain, &stream_c);
        gpu_utils::fft_inplace(&mut c, &stream_c, None);

        stream_b.synchronize().unwrap();

        let ab = T::local_mul_vec::<B>(eval_a, eval_b, state, &stream_a);

        stream_a.synchronize().unwrap();

        stream_a.destroy().unwrap();
        stream_b.destroy().unwrap();

        let mut result = DeviceVec::device_malloc_async(c.len(), &stream_c)
            .expect("Failed to allocate device vector");

        let mut cfg = VecOpsConfig::default();
        cfg.stream_handle = *stream_c;
        cfg.is_async = true;
        sub_scalars(&ab, &c, result.as_mut_slice(), &cfg).unwrap();

        stream_c.synchronize().unwrap();
        stream_c.destroy().unwrap();

        Ok(result)
    }

    fn requires_eval_c() -> bool {
        false
    }
}

/// Implements the witness map used by libsnark. The arkworks witness map calculates the
/// coefficients of H through computing (AB-C)/Z in the evaluation domain and going back to the
/// coefficients domain.
///
/// Based on <https://github.com/arkworks-rs/groth16/>.
pub struct LibSnarkReduction;

impl R1CSToQAP for LibSnarkReduction {
    #[instrument(level = "debug", name = "witness map from matrices", skip_all)]
    fn witness_map_from_r1cs_eval<
        B: ArkIcicleBridge,
        T: CircomGroth16Prover<B::IcicleScalarField>,
    >(
        state: &mut T::State,
        eval_a: &mut T::DeviceShares,
        eval_b: &mut T::DeviceShares,
        eval_c: Option<&mut DeviceVec<B::IcicleScalarField>>,
        public_inputs: &DeviceSlice<B::IcicleScalarField>,
        _: &DeviceSlice<B::IcicleScalarField>,
        num_constraints: usize,
        domain_size: usize,
    ) -> Result<DeviceVec<B::IcicleScalarField>> {
        assert!(eval_c.is_some());

        let c = eval_c.unwrap();

        let id = state.id();
        let coset_gen = Some(ark_to_icicle_scalar(B::ArkScalarField::GENERATOR));

        // Computation of a
        let promoted_public = T::promote_to_trivial_shares(id, public_inputs);
        T::copy_to_device_shares(&promoted_public, eval_a, num_constraints, domain_size);

        // Computation of a
        let mut stream_a = IcicleStream::create().unwrap();
        T::ifft_in_place(eval_a, &stream_a, None);
        T::fft_in_place(eval_a, &stream_a, coset_gen);

        // Computation of b
        let mut stream_b = IcicleStream::create().unwrap();
        T::ifft_in_place(eval_b, &stream_b, None);
        T::fft_in_place(eval_b, &stream_b, coset_gen);

        // Computation of c
        let mut stream_c = IcicleStream::create().unwrap();
        gpu_utils::ifft_inplace(c, &stream_c, None);
        gpu_utils::fft_inplace(c, &stream_c, coset_gen);

        stream_b.synchronize().unwrap();

        let ab = T::local_mul_vec::<B>(eval_a, eval_b, state, &stream_a);

        stream_a.synchronize().unwrap();

        stream_a.destroy().unwrap();
        stream_b.destroy().unwrap();

        // TODO CESAR
        let vanishing_polynomial_over_coset =
            (B::ArkScalarField::GENERATOR.pow(&[domain_size as u64]) - B::ArkScalarField::one())
                .inverse()
                .unwrap();

        let vanishing_polynomial_over_coset = ark_to_icicle_scalar(vanishing_polynomial_over_coset);

        let tmp = vec![vanishing_polynomial_over_coset; c.len()];

        let mut sub = DeviceVec::device_malloc_async(c.len(), &stream_c)
            .expect("Failed to allocate device vector");
        let mut result = DeviceVec::device_malloc_async(c.len(), &stream_c)
            .expect("Failed to allocate device vector");

        let mut cfg = VecOpsConfig::default();
        cfg.stream_handle = *stream_c;
        cfg.is_async = true;
        sub_scalars(&ab, c, sub.as_mut_slice(), &cfg).unwrap();
        mul_scalars(
            &sub,
            HostSlice::from_slice(&tmp),
            result.as_mut_slice(),
            &cfg,
        )
        .unwrap();

        gpu_utils::ifft_inplace(result.as_mut_slice(), &stream_c, coset_gen);

        stream_c.synchronize().unwrap();
        stream_c.destroy().unwrap();

        Ok(result)
    }

    fn requires_eval_c() -> bool {
        true
    }
}
