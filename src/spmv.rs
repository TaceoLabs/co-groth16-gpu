//! R1CS constraint evaluation on the device: a sparse matrix-vector product with the
//! assignment, which is on the device anyway for the MSMs.
//!
//! Evaluating `A` and `B` on the host takes one field multiplication per non-zero matrix
//! entry, millions for large circuits, and sits on the critical path of every proof.
//!
//! The kernel (`src/cuda/spmv.cu`) is embedded as PTX and loaded through the CUDA driver at
//! runtime, which compiles it for the GPU at hand, so building the crate needs no CUDA
//! toolchain. Regenerate the PTX with `scripts/build-ptx.sh` after changing the kernel.

use std::{
    collections::HashMap,
    ffi::{CStr, c_void},
    ptr::null_mut,
    sync::{Mutex, OnceLock},
};

use ark_ff::PrimeField;
use icicle_runtime::{
    memory::{DeviceSlice, DeviceVec, HostOrDeviceSlice, HostSlice},
    stream::IcicleStream,
};
use rayon::prelude::*;

const PTX: &str = concat!(include_str!("cuda/spmv.ptx"), "\0");
const THREADS_PER_BLOCK: u32 = 256;

/// Whether the active device is a CUDA device, i.e. can run the kernel.
pub(crate) fn available() -> bool {
    icicle_runtime::get_active_device().is_ok_and(|device| device.get_device_type() == "CUDA")
}

type CuResult = i32;
type Handle = *mut c_void;

/// The CUDA driver API functions used here, loaded from `libcuda` at runtime.
struct Driver {
    init: unsafe extern "C" fn(u32) -> CuResult,
    device_get: unsafe extern "C" fn(*mut i32, i32) -> CuResult,
    primary_ctx_retain: unsafe extern "C" fn(*mut Handle, i32) -> CuResult,
    ctx_set_current: unsafe extern "C" fn(Handle) -> CuResult,
    module_load_data: unsafe extern "C" fn(*mut Handle, *const c_void) -> CuResult,
    module_get_function: unsafe extern "C" fn(*mut Handle, Handle, *const i8) -> CuResult,
    launch_kernel: unsafe extern "C" fn(
        Handle,
        u32,
        u32,
        u32,
        u32,
        u32,
        u32,
        u32,
        Handle,
        *mut *mut c_void,
        *mut *mut c_void,
    ) -> CuResult,
}

/// Resolves `name` from `lib` as a function pointer of type `T`.
///
/// # Safety
/// `T` must be the function's signature.
unsafe fn symbol<T>(lib: *mut c_void, name: &CStr) -> T {
    assert_eq!(size_of::<T>(), size_of::<*mut c_void>());
    // SAFETY: the caller guarantees the signature.
    unsafe {
        let f = libc::dlsym(lib, name.as_ptr());
        assert!(!f.is_null(), "missing CUDA driver symbol {name:?}");
        std::mem::transmute_copy(&f)
    }
}

fn driver() -> &'static Driver {
    static DRIVER: OnceLock<Driver> = OnceLock::new();
    DRIVER.get_or_init(|| {
        // SAFETY: the symbols are given the signatures from `cuda.h`.
        unsafe {
            let lib = libc::dlopen(c"libcuda.so.1".as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL);
            assert!(
                !lib.is_null(),
                "failed to load the CUDA driver (libcuda.so.1)"
            );
            Driver {
                init: symbol(lib, c"cuInit"),
                device_get: symbol(lib, c"cuDeviceGet"),
                primary_ctx_retain: symbol(lib, c"cuDevicePrimaryCtxRetain"),
                ctx_set_current: symbol(lib, c"cuCtxSetCurrent"),
                module_load_data: symbol(lib, c"cuModuleLoadData"),
                module_get_function: symbol(lib, c"cuModuleGetFunction"),
                launch_kernel: symbol(lib, c"cuLaunchKernel"),
            }
        }
    })
}

fn check(res: CuResult, call: &str) {
    assert_eq!(res, 0, "{call} failed with CUresult {res}");
}

/// The kernel, loaded into the primary context of a device: the context the CUDA runtime,
/// and hence Icicle, works in.
#[derive(Clone, Copy)]
struct Kernel {
    ctx: usize,
    function: usize,
}

/// The kernel for `device`, loaded on first use.
fn kernel(device: i32) -> Kernel {
    static KERNELS: OnceLock<Mutex<HashMap<i32, Kernel>>> = OnceLock::new();
    let mut kernels = KERNELS.get_or_init(Default::default).lock().unwrap();
    *kernels.entry(device).or_insert_with(|| {
        let d = driver();
        let (mut dev, mut ctx, mut module, mut function) = (0, null_mut(), null_mut(), null_mut());
        // SAFETY: plain driver API calls with valid out-pointers; `PTX` is NUL-terminated.
        unsafe {
            check((d.init)(0), "cuInit");
            check((d.device_get)(&mut dev, device), "cuDeviceGet");
            check(
                (d.primary_ctx_retain)(&mut ctx, dev),
                "cuDevicePrimaryCtxRetain",
            );
            check((d.ctx_set_current)(ctx), "cuCtxSetCurrent");
            check(
                (d.module_load_data)(&mut module, PTX.as_ptr().cast()),
                "cuModuleLoadData",
            );
            check(
                (d.module_get_function)(&mut function, module, c"spmv_kernel".as_ptr()),
                "cuModuleGetFunction",
            );
        }
        Kernel {
            ctx: ctx as usize,
            function: function as usize,
        }
    })
}

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

/// An R1CS matrix in CSR form on the device, with Montgomery-form coefficients.
pub(crate) struct DeviceMatrix<F> {
    row_ptr: DeviceVec<u32>,
    cols: DeviceVec<u32>,
    coeffs: DeviceVec<F>,
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
        let coeffs = matrix
            .par_iter()
            .flat_map_iter(|row| row.iter().map(|(coeff, _)| *coeff))
            .collect::<Vec<_>>();
        // SAFETY: both types are 32 bytes; an arkworks field element is its Montgomery-form
        // limbs, which is what the kernel expects for the coefficients.
        let coeffs =
            unsafe { std::slice::from_raw_parts(coeffs.as_ptr().cast::<F>(), coeffs.len()) };

        let modulus: [u64; 4] = T::MODULUS.as_ref().try_into().unwrap();
        // Newton iteration for the inverse of the (odd) lowest limb modulo 2^64.
        let mut inv = 1u64;
        for _ in 0..6 {
            inv = inv.wrapping_mul(2u64.wrapping_sub(modulus[0].wrapping_mul(inv)));
        }
        Self {
            max_col: cols.par_iter().copied().max().unwrap_or(0) as usize,
            row_ptr: upload(&row_ptr),
            cols: upload(&cols),
            coeffs: upload(coeffs),
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
        let kernel = kernel(
            icicle_runtime::get_active_device()
                .expect("Failed to get active device")
                .id,
        );
        let d = driver();
        // SAFETY: only the device addresses are taken, nothing is dereferenced on the host.
        let (mut row_ptr, mut cols, mut coeffs, mut public, mut witness_ptr, mut out_ptr) = unsafe {
            (
                self.row_ptr.as_ptr(),
                self.cols.as_ptr(),
                self.coeffs.as_ptr(),
                public_inputs.as_ptr(),
                witness.as_ptr(),
                out.as_mut_ptr(),
            )
        };
        let mut num_public = public_inputs.len() as u32;
        let mut rows = self.rows() as u32;
        let mut field = self.field;
        let mut params: [*mut c_void; 9] = [
            (&raw mut row_ptr).cast(),
            (&raw mut cols).cast(),
            (&raw mut coeffs).cast(),
            (&raw mut public).cast(),
            (&raw mut num_public).cast(),
            (&raw mut witness_ptr).cast(),
            (&raw mut rows).cast(),
            (&raw mut out_ptr).cast(),
            (&raw mut field).cast(),
        ];
        // SAFETY: the parameters match the kernel signature, all pointers are device buffers
        // of the sizes the kernel reads, and the column indices are in bounds (checked above).
        unsafe {
            check((d.ctx_set_current)(kernel.ctx as Handle), "cuCtxSetCurrent");
            check(
                (d.launch_kernel)(
                    kernel.function as Handle,
                    rows.div_ceil(THREADS_PER_BLOCK),
                    1,
                    1,
                    THREADS_PER_BLOCK,
                    1,
                    1,
                    0,
                    stream.handle.cast(),
                    params.as_mut_ptr(),
                    null_mut(),
                ),
                "cuLaunchKernel",
            );
        }
    }
}
