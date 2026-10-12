//! R1CS constraint evaluation on the device: a sparse matrix-vector product with the
//! assignment, which is on the device anyway for the MSMs.
//!
//! Evaluating `A` and `B` on the host takes one field multiplication per non-zero matrix
//! entry, millions for large circuits, and sits on the critical path of every proof.
//!
//! The kernel (`src/cuda/spmv.cu`) is embedded as PTX and loaded through the CUDA driver at
//! runtime, which compiles it for the GPU at hand, so building the crate needs no CUDA
//! toolchain. Regenerate the PTX with `scripts/build-ptx.sh` after changing the kernel.

use std::collections::HashMap;

use ark_ff::PrimeField;
use icicle_runtime::{
    memory::{DeviceSlice, DeviceVec, HostOrDeviceSlice, HostSlice},
    stream::IcicleStream,
};
use rayon::prelude::*;

use crate::cuda::{Args, Module, device_ptr, device_ptr_mut};

/// Kernel `i` handles a row with `1 << i` lanes.
static SPMV: Module = Module::new(
    concat!(include_str!("cuda/spmv.ptx"), "\0"),
    &[
        c"spmv_l1",
        c"spmv_l2",
        c"spmv_l4",
        c"spmv_l8",
        c"spmv_l16",
        c"spmv_l32",
    ],
);
const THREADS_PER_BLOCK: u32 = 256;
const BINS: usize = 6;

/// The field parameters the kernel takes by value; matches `FieldParams` in `spmv.cu`.
#[repr(C)]
#[derive(Clone, Copy)]
struct FieldParams {
    modulus: [u64; 4],
    /// `-modulus^-1 mod 2^64`
    inv: u64,
}

fn upload<T>(host: &[T]) -> DeviceVec<T> {
    let mut dev = DeviceVec::device_malloc(host.len()).expect("Failed to allocate device vector");
    dev.copy_from_host(HostSlice::from_slice(host))
        .expect("Failed to upload matrix");
    dev
}

/// An R1CS matrix in CSR form, its entries indexing a table of the distinct coefficients (in
/// Montgomery form), its rows binned by length: bin `i` holds the rows of up to `4 << i` entries.
pub(crate) struct DeviceMatrix<F> {
    /// The non-empty bins.
    bins: Vec<(usize, DeviceVec<u32>)>,
    row_ptr: DeviceVec<u32>,
    cols: DeviceVec<u32>,
    coeff_idx: DeviceVec<u32>,
    table: DeviceVec<F>,
    /// The largest column index.
    max_col: usize,
    field: FieldParams,
}

impl<F> DeviceMatrix<F> {
    pub(crate) fn new<T: PrimeField>(matrix: &[Vec<(T, usize)>]) -> Self {
        assert_eq!(size_of::<T>(), 32, "only 256-bit fields are supported");
        assert_eq!(size_of::<F>(), 32, "only 256-bit fields are supported");
        let mut row_ptr = Vec::with_capacity(matrix.len() + 1);
        row_ptr.push(0u32);
        for row in matrix {
            let end = u32::try_from(*row_ptr.last().unwrap() as usize + row.len())
                .expect("too many non-zero matrix entries");
            row_ptr.push(end);
        }
        let cols = matrix
            .par_iter()
            .flat_map_iter(|row| row.iter().map(|(_, col)| *col as u32))
            .collect::<Vec<_>>();
        let mut table = Vec::new();
        let mut index = HashMap::new();
        let coeff_idx = matrix
            .iter()
            .flatten()
            .map(|(coeff, _)| {
                *index.entry(*coeff).or_insert_with(|| {
                    table.push(*coeff);
                    table.len() as u32 - 1
                })
            })
            .collect::<Vec<_>>();
        // SAFETY: both types are 32 bytes; an arkworks field element is its Montgomery-form
        // limbs, which is what the kernel expects for the coefficients.
        let table = unsafe { std::slice::from_raw_parts(table.as_ptr().cast::<F>(), table.len()) };

        let modulus: [u64; 4] = T::MODULUS.as_ref().try_into().unwrap();
        // Newton iteration for the inverse of the (odd) lowest limb modulo 2^64.
        let mut inv = 1u64;
        for _ in 0..6 {
            inv = inv.wrapping_mul(2u64.wrapping_sub(modulus[0].wrapping_mul(inv)));
        }
        let mut bins = vec![Vec::new(); BINS];
        for (row, entries) in matrix.iter().enumerate() {
            let bin = (0..BINS - 1)
                .find(|&i| entries.len() <= 4 << i)
                .unwrap_or(BINS - 1);
            bins[bin].push(row as u32);
        }
        Self {
            bins: bins
                .iter()
                .enumerate()
                .filter(|(_, rows)| !rows.is_empty())
                .map(|(bin, rows)| (bin, upload(rows)))
                .collect(),
            max_col: cols.par_iter().copied().max().unwrap_or(0) as usize,
            row_ptr: upload(&row_ptr),
            cols: upload(&cols),
            coeff_idx: upload(&coeff_idx),
            table: upload(table),
            field: FieldParams {
                modulus,
                inv: inv.wrapping_neg(),
            },
        }
    }

    pub(crate) fn rows(&self) -> usize {
        self.row_ptr.len() - 1
    }

    /// Computes `out = M * (public_inputs ++ witness)` on `stream`.
    pub(crate) fn mul_vec(
        &self,
        public_inputs: &DeviceSlice<F>,
        witness: &DeviceSlice<F>,
        out: &mut DeviceSlice<F>,
        stream: &IcicleStream,
    ) {
        assert_eq!(out.len(), self.rows(), "output length mismatch");
        assert!(
            self.max_col < public_inputs.len() + witness.len(),
            "assignment too short for the matrix"
        );
        let module = SPMV.get();
        let out = device_ptr_mut(out);
        for (bin, rows) in &self.bins {
            let threads = (rows.len() as u32) << bin;
            // SAFETY: the arguments match the kernel signature, all pointers are device
            // buffers of the sizes the kernel reads, and the row and column indices are in
            // bounds (checked above, and when binning).
            unsafe {
                module.launch_args(
                    *bin,
                    (threads.div_ceil(THREADS_PER_BLOCK), 1),
                    THREADS_PER_BLOCK,
                    stream,
                    Args::new()
                        .arg(device_ptr(rows))
                        .arg(rows.len() as u32)
                        .arg(device_ptr(&self.row_ptr))
                        .arg(device_ptr(&self.cols))
                        .arg(device_ptr(&self.coeff_idx))
                        .arg(device_ptr(&self.table))
                        .arg(device_ptr(public_inputs))
                        .arg(public_inputs.len() as u32)
                        .arg(device_ptr(witness))
                        .arg(out)
                        .arg(self.field),
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::fs::File;

    use ark_bn254::{Bn254, Fr};
    use ark_ff::{BigInteger, PrimeField, UniformRand, Zero};
    use circom_types::{CheckElement, groth16::Zkey};
    use icicle_bn254::curve::ScalarField;
    use icicle_core::traits::FieldImpl;
    use icicle_runtime::{
        memory::{DeviceVec, HostSlice},
        stream::IcicleStream,
    };

    use super::DeviceMatrix;
    use crate::bridges::ark_scalars_to_device_into;

    #[test]
    #[ignore = "needs a GPU"]
    fn spmv_matches_host() {
        crate::load_backend_from_env_and_set_device(0);
        let zkey =
            File::open("test_vectors/Groth16/bn254/transaction_batched/circuit.zkey").unwrap();
        let (matrices, _): (
            co_groth16::ConstraintMatrices<Fr>,
            co_groth16::ProvingKey<Bn254>,
        ) = Zkey::<Bn254>::from_reader(zkey, CheckElement::No)
            .unwrap()
            .into();
        let mut rng = rand::thread_rng();
        let num_public = matrices.num_instance_variables;
        let z = (0..num_public + matrices.num_witness_variables)
            .map(|_| Fr::rand(&mut rng))
            .collect::<Vec<_>>();
        let mut public = DeviceVec::<ScalarField>::device_malloc(num_public).unwrap();
        ark_scalars_to_device_into(&z[..num_public], &mut public);
        let mut witness = DeviceVec::<ScalarField>::device_malloc(z.len() - num_public).unwrap();
        ark_scalars_to_device_into(&z[num_public..], &mut witness);
        let stream = IcicleStream::create().unwrap();
        for matrix in [&matrices.a, &matrices.b] {
            let mut out = DeviceVec::<ScalarField>::device_malloc(matrix.len()).unwrap();
            DeviceMatrix::<ScalarField>::new(matrix).mul_vec(&public, &witness, &mut out, &stream);
            let mut host = vec![ScalarField::zero(); matrix.len()];
            out.copy_to_host(HostSlice::from_mut_slice(&mut host))
                .unwrap();
            for (row, got) in matrix.iter().zip(&host) {
                let expected = row.iter().fold(Fr::zero(), |acc, (c, i)| acc + *c * z[*i]);
                assert_eq!(got.to_bytes_le(), expected.into_bigint().to_bytes_le());
            }
        }
    }
}
