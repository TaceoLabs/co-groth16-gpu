use ark_ff::{FftField, LegendreSymbol, PrimeField};
use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};
use ark_relations::r1cs::Matrix;
use mpc_core::MpcState;
use num_traits::ToPrimitive;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

/// Computes the roots of unity over the provided prime field. This method
/// is equivalent with [circom's implementation](https://github.com/iden3/ffjavascript/blob/337b881579107ab74d5b2094dbe1910e33da4484/src/wasm_field1.js).
///
/// We calculate smallest quadratic non residue q (by checking q^((p-1)/2)=-1 mod p). We also calculate smallest t s.t. p-1=2^s*t, s is the two adicity.
/// We use g=q^t (this is a 2^s-th root of unity) as (some kind of) generator and compute another domain by repeatedly squaring g, should get to 1 in the s+1-th step.
/// Then if log2(\text{domain_size}) equals s we take q^2 as root of unity. Else we take the log2(\text{domain_size}) + 1-th element of the domain created above.
pub(crate) fn roots_of_unity<F: PrimeField + FftField>() -> (F, Vec<F>) {
    let mut roots = vec![F::zero(); F::TWO_ADICITY.to_usize().unwrap() + 1];
    let mut q = F::one();
    while q.legendre() != LegendreSymbol::QuadraticNonResidue {
        q += F::one();
    }
    let z = q.pow(F::TRACE);
    roots[0] = z;
    for i in 1..roots.len() {
        roots[i] = roots[i - 1].square();
    }
    roots.reverse();
    (q, roots)
}

/* old way of computing root of unity, does not work for bls12_381:
let root_of_unity = {
    let domain_size_double = 2 * domain_size;
    let domain_double =
        D::new(domain_size_double).ok_or(SynthesisError::PolynomialDegreeTooLarge)?;
    domain_double.element(1)
};
new one is computed in the same way as in snarkjs (More precisely in ffjavascript/src/wasm_field1.js)
calculate smallest quadratic non residue q (by checking q^((p-1)/2)=-1 mod p) also calculate smallest t (F::TRACE) s.t. p-1=2^s*t, s is the two_adicity
use g=q^t (this is a 2^s-th root of unity) as (some kind of) generator and compute another domain by repeatedly squaring g, should get to 1 in the s+1-th step.
then if log2(domain_size) equals s we take as root of unity q^2, and else we take the log2(domain_size) + 1-th element of the domain created above
*/
pub(crate) fn root_of_unity_for_groth16<F: PrimeField + FftField>(
    pow: usize,
    domain: &mut GeneralEvaluationDomain<F>,
) -> F {
    let (q, roots) = roots_of_unity::<F>();
    match domain {
        GeneralEvaluationDomain::Radix2(domain) => {
            domain.group_gen = roots[pow];
            domain.group_gen_inv = domain.group_gen.inverse().expect("can compute inverse");
        }
        GeneralEvaluationDomain::MixedRadix(domain) => {
            domain.group_gen = roots[pow];
            domain.group_gen_inv = domain.group_gen.inverse().expect("can compute inverse");
        }
    };
    if F::TWO_ADICITY.to_u64().unwrap() == domain.log_size_of_group() {
        q.square()
    } else {
        roots[domain.log_size_of_group().to_usize().unwrap() + 1]
    }
}

pub(crate) fn evaluate_constraint<
    P: ark_ec::pairing::Pairing,
    T: co_groth16::CircomGroth16Prover<P>,
>(
    id: <T::State as MpcState>::PartyID,
    domain_size: usize,
    matrix: &Matrix<P::ScalarField>,
    public_inputs: &[P::ScalarField],
    private_witness: &[T::ArithmeticShare],
) -> Vec<T::ArithmeticShare> {
    let mut result = matrix
        .par_iter()
        .with_min_len(256)
        .map(|x| T::evaluate_constraint(id, x, public_inputs, private_witness))
        .collect::<Vec<_>>();
    result.resize(domain_size, T::ArithmeticShare::default());
    result
}

pub fn evaluate_constraint_half_share<
    P: ark_ec::pairing::Pairing,
    T: co_groth16::CircomGroth16Prover<P>,
>(
    id: <T::State as MpcState>::PartyID,
    domain_size: usize,
    matrix: &Matrix<P::ScalarField>,
    public_inputs: &[P::ScalarField],
    private_witness: &[T::ArithmeticShare],
) -> Vec<T::ArithmeticHalfShare> {
    let mut result = matrix
        .par_iter()
        .with_min_len(256)
        .map(|x| T::evaluate_constraint_half_share(id, x, public_inputs, private_witness))
        .collect::<Vec<_>>();
    result.resize(domain_size, T::ArithmeticHalfShare::default());
    result
}
