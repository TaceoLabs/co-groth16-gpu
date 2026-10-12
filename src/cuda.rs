//! The CUDA driver API functions the crate calls itself, loaded from `libcuda` at runtime so
//! that building the crate needs no CUDA toolchain.

use std::{
    ffi::{CStr, c_void},
    ptr::null_mut,
    sync::{Mutex, OnceLock},
};

use icicle_runtime::{
    memory::{DeviceSlice, HostOrDeviceSlice},
    stream::IcicleStream,
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
    pub(crate) event_create: unsafe extern "C" fn(*mut Handle, u32) -> CuResult,
    pub(crate) event_record: unsafe extern "C" fn(Handle, Handle) -> CuResult,
    pub(crate) event_destroy: unsafe extern "C" fn(Handle) -> CuResult,
    pub(crate) stream_wait_event: unsafe extern "C" fn(Handle, Handle, u32) -> CuResult,
    pub(crate) ctx_get_stream_priority_range: unsafe extern "C" fn(*mut i32, *mut i32) -> CuResult,
    pub(crate) stream_create_with_priority: unsafe extern "C" fn(*mut Handle, u32, i32) -> CuResult,
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
                event_create: symbol(lib, c"cuEventCreate"),
                event_record: symbol(lib, c"cuEventRecord"),
                event_destroy: symbol(lib, c"cuEventDestroy_v2"),
                stream_wait_event: symbol(lib, c"cuStreamWaitEvent"),
                ctx_get_stream_priority_range: symbol(lib, c"cuCtxGetStreamPriorityRange"),
                stream_create_with_priority: symbol(lib, c"cuStreamCreateWithPriority"),
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

/// Makes the primary context of `device`, which Icicle works in, current.
fn primary_context(device: i32) -> Handle {
    let d = driver();
    let (mut cu_device, mut ctx) = (0, null_mut());
    // SAFETY: plain driver API calls with valid out-pointers.
    unsafe {
        check((d.init)(0), "cuInit");
        check((d.device_get)(&mut cu_device, device), "cuDeviceGet");
        check(
            (d.primary_ctx_retain)(&mut ctx, cu_device),
            "cuDevicePrimaryCtxRetain",
        );
        check((d.ctx_set_current)(ctx), "cuCtxSetCurrent");
    }
    ctx
}

/// The active device, if it is a CUDA device.
fn active_cuda_device() -> Option<i32> {
    let device = icicle_runtime::get_active_device().ok()?;
    (device.get_device_type() == "CUDA").then_some(device.id)
}

/// Whether the active device is a CUDA device, i.e. can run the crate's kernels.
pub(crate) fn on_cuda_device() -> bool {
    active_cuda_device().is_some()
}

/// A stream of the highest priority, an ordinary one on a non-CUDA device.
pub(crate) fn high_priority_stream() -> IcicleStream {
    let Some(device) = active_cuda_device() else {
        return IcicleStream::create().expect("Failed to create stream");
    };
    primary_context(device);
    let d = driver();
    let mut stream = null_mut();
    let (mut least, mut greatest) = (0, 0);
    // SAFETY: plain driver API calls with valid out-pointers.
    unsafe {
        check(
            (d.ctx_get_stream_priority_range)(&mut least, &mut greatest),
            "cuCtxGetStreamPriorityRange",
        );
        check(
            (d.stream_create_with_priority)(&mut stream, 0, greatest),
            "cuStreamCreateWithPriority",
        );
    }
    IcicleStream { handle: stream }
}

/// A PTX module, loaded on each device on first use.
pub(crate) struct Module {
    /// NUL-terminated PTX.
    ptx: &'static str,
    kernels: &'static [&'static CStr],
    loaded: Mutex<Vec<(i32, Loaded)>>,
}

/// A [`Module`] loaded on one device.
#[derive(Clone)]
pub(crate) struct Loaded {
    ctx: usize,
    functions: Vec<usize>,
}

impl Module {
    pub(crate) const fn new(ptx: &'static str, kernels: &'static [&'static CStr]) -> Self {
        Self {
            ptx,
            kernels,
            loaded: Mutex::new(Vec::new()),
        }
    }

    /// The module on the active device.
    pub(crate) fn get(&self) -> Loaded {
        let device = icicle_runtime::get_active_device()
            .expect("Failed to get active device")
            .id;
        let mut loaded = self.loaded.lock().unwrap();
        if let Some((_, l)) = loaded.iter().find(|(d, _)| *d == device) {
            return l.clone();
        }
        let d = driver();
        let ctx = primary_context(device);
        let mut module = null_mut();
        // SAFETY: plain driver API calls with valid out-pointers; the PTX is NUL-terminated.
        let functions = unsafe {
            check(
                (d.module_load_data)(&mut module, self.ptx.as_ptr().cast()),
                "cuModuleLoadData",
            );
            self.kernels
                .iter()
                .map(|name| {
                    let mut function = null_mut();
                    check(
                        (d.module_get_function)(&mut function, module, name.as_ptr()),
                        "cuModuleGetFunction",
                    );
                    function as usize
                })
                .collect()
        };
        let l = Loaded {
            ctx: ctx as usize,
            functions,
        };
        loaded.push((device, l.clone()));
        l
    }
}

impl Loaded {
    /// Launches kernel `kernel` (an index into the module's kernel list).
    ///
    /// # Safety
    /// `params` must point to values matching the kernel's parameters, and the kernel must
    /// only access memory that is valid for the given grid.
    pub(crate) unsafe fn launch(
        &self,
        kernel: usize,
        (blocks_x, blocks_y): (u32, u32),
        threads_per_block: u32,
        stream: &IcicleStream,
        params: &mut [*mut c_void],
    ) {
        let d = driver();
        // SAFETY: guaranteed by the caller.
        unsafe {
            check((d.ctx_set_current)(self.ctx as Handle), "cuCtxSetCurrent");
            check(
                (d.launch_kernel)(
                    self.functions[kernel] as Handle,
                    blocks_x,
                    blocks_y,
                    1,
                    threads_per_block,
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
    /// # Safety
    /// As for [`Loaded::launch`].
    pub(crate) unsafe fn launch_args(
        &self,
        kernel: usize,
        grid: (u32, u32),
        threads_per_block: u32,
        stream: &IcicleStream,
        mut args: Args,
    ) {
        let base = args.slots.as_mut_ptr();
        // SAFETY: the offsets are within the slots.
        let mut params = args
            .offsets
            .iter()
            .map(|&o| unsafe { base.add(o) }.cast::<c_void>())
            .collect::<Vec<_>>();
        // SAFETY: guaranteed by the caller.
        unsafe { self.launch(kernel, grid, threads_per_block, stream, &mut params) }
    }
}

/// `v`'s address, as a kernel argument.
pub(crate) fn device_ptr<T>(v: &DeviceSlice<T>) -> u64 {
    // SAFETY: only the address is taken, nothing is dereferenced on the host.
    unsafe { v.as_ptr() as u64 }
}

/// `v`'s address, as a kernel argument the kernel writes to.
pub(crate) fn device_ptr_mut<T>(v: &mut DeviceSlice<T>) -> u64 {
    // SAFETY: only the address is taken, nothing is dereferenced on the host.
    unsafe { v.as_mut_ptr() as u64 }
}

/// Kernel arguments.
#[derive(Default)]
pub(crate) struct Args {
    slots: Vec<u64>,
    offsets: Vec<usize>,
}

impl Args {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn arg<T: Copy>(mut self, value: T) -> Self {
        let offset = self.slots.len();
        self.slots.resize(offset + size_of::<T>().div_ceil(8), 0);
        // SAFETY: the slots just added have room for a `T`.
        unsafe {
            self.slots
                .as_mut_ptr()
                .add(offset)
                .cast::<T>()
                .write_unaligned(value)
        };
        self.offsets.push(offset);
        self
    }
}

/// A CUDA event, to order work across streams.
pub(crate) struct Event(Handle);

// SAFETY: a CUDA event handle may be used from any thread.
unsafe impl Send for Event {}
unsafe impl Sync for Event {}

impl Event {
    /// An event on the active device, without timing.
    pub(crate) fn new() -> Self {
        primary_context(active_cuda_device().expect("not a CUDA device"));
        let mut event = null_mut();
        // SAFETY: a plain driver API call with a valid out-pointer.
        unsafe { check((driver().event_create)(&mut event, 2), "cuEventCreate") };
        Self(event)
    }

    /// Makes `waiter` wait for the work submitted to `stream` so far.
    pub(crate) fn order(&self, stream: &IcicleStream, waiter: &IcicleStream) {
        let d = driver();
        // SAFETY: valid event and stream handles.
        unsafe {
            check(
                (d.event_record)(self.0, stream.handle.cast()),
                "cuEventRecord",
            );
            check(
                (d.stream_wait_event)(waiter.handle.cast(), self.0, 0),
                "cuStreamWaitEvent",
            );
        }
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        // SAFETY: the event is not used after this.
        unsafe { (driver().event_destroy)(self.0) };
    }
}
