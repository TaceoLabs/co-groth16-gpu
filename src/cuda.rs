//! The CUDA driver API functions the crate calls itself, loaded from `libcuda` at runtime so
//! that building the crate needs no CUDA toolchain.

use std::{
    ffi::{CStr, c_void},
    ptr::null_mut,
    sync::OnceLock,
};

pub(crate) type CuResult = i32;
pub(crate) type Handle = *mut c_void;

/// The CUDA driver API functions used by the crate.
pub(crate) struct Driver {
    pub(crate) init: unsafe extern "C" fn(u32) -> CuResult,
    pub(crate) device_get: unsafe extern "C" fn(*mut i32, i32) -> CuResult,
    pub(crate) primary_ctx_retain: unsafe extern "C" fn(*mut Handle, i32) -> CuResult,
    pub(crate) ctx_set_current: unsafe extern "C" fn(Handle) -> CuResult,
    pub(crate) module_load_data: unsafe extern "C" fn(*mut Handle, *const c_void) -> CuResult,
    pub(crate) module_get_function:
        unsafe extern "C" fn(*mut Handle, Handle, *const i8) -> CuResult,
    pub(crate) launch_kernel: unsafe extern "C" fn(
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
    pub(crate) device_get_default_mem_pool: unsafe extern "C" fn(*mut Handle, i32) -> CuResult,
    pub(crate) mem_pool_set_attribute: unsafe extern "C" fn(Handle, i32, *mut c_void) -> CuResult,
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

pub(crate) fn driver() -> &'static Driver {
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
                device_get_default_mem_pool: symbol(lib, c"cuDeviceGetDefaultMemPool"),
                mem_pool_set_attribute: symbol(lib, c"cuMemPoolSetAttribute"),
            }
        }
    })
}

pub(crate) fn check(res: CuResult, call: &str) {
    assert_eq!(res, 0, "{call} failed with CUresult {res}");
}

/// Keeps device memory freed by Icicle in the CUDA stream-ordered allocator's pool of the
/// active device, instead of returning it to the driver at every synchronization.
///
/// Icicle allocates the MSM and NTT scratch with `cudaMallocAsync` from the device's default
/// pool, whose release threshold is 0. Without this, every MSM maps its scratch memory again,
/// which is slow and makes the MSM time fluctuate. Icicle links the CUDA runtime statically,
/// so the threshold is set through the driver API (the pool is a per-device driver object).
pub(crate) fn retain_freed_device_memory() {
    const CU_MEMPOOL_ATTR_RELEASE_THRESHOLD: i32 = 4;
    let device = icicle_runtime::get_active_device().expect("Failed to get active device");
    if device.get_device_type() != "CUDA" {
        return;
    }
    let d = driver();
    let (mut cu_device, mut pool) = (0, null_mut());
    let mut threshold = u64::MAX;
    // SAFETY: plain driver API calls with valid out-pointers; the attribute value is a u64.
    unsafe {
        check((d.init)(0), "cuInit");
        check((d.device_get)(&mut cu_device, device.id), "cuDeviceGet");
        check(
            (d.device_get_default_mem_pool)(&mut pool, cu_device),
            "cuDeviceGetDefaultMemPool",
        );
        check(
            (d.mem_pool_set_attribute)(
                pool,
                CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                (&raw mut threshold).cast(),
            ),
            "cuMemPoolSetAttribute",
        );
    }
}
