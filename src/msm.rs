//! MSMs on Icicle's precomputed bases, which hold `2^(c * k) * P_i` at `i * pf + k`: digit `k`
//! of scalar `i` adds that base to bucket `digit`, and an MSM is `sum_b b * bucket_b`. Kernels in
//! `src/cuda/msm_*.cu`.

use icicle_core::curve::{Affine, Curve, Projective};
use icicle_runtime::{
    memory::{DeviceSlice, DeviceVec, HostOrDeviceSlice},
    stream::IcicleStream,
};

use crate::cuda::{
    Args, Event, Module, device_ptr as ptr, device_ptr_mut as ptr_mut, high_priority_stream,
};

static SORT: Module = Module::new(
    concat!(include_str!("cuda/msm_sort.ptx"), "\0"),
    &[
        c"msm_zero",
        c"msm_count",
        c"msm_scan_tiles",
        c"msm_scan_top",
        c"msm_scan_apply",
        c"msm_scatter",
        c"msm_order",
    ],
);
const ZERO: usize = 0;
const COUNT: usize = 1;
const SCAN_TILES: usize = 2;
const SCAN_TOP: usize = 3;
const SCAN_APPLY: usize = 4;
const SCATTER: usize = 5;
const ORDER: usize = 6;

const CURVE_KERNELS: &[&std::ffi::CStr] = &[
    c"msm_accumulate",
    c"msm_large_reduce",
    c"msm_reduce_first",
    c"msm_reduce_pairs",
];
static G1: Module = Module::new(
    concat!(include_str!("cuda/msm_g1.ptx"), "\0"),
    CURVE_KERNELS,
);
static G2: Module = Module::new(
    concat!(include_str!("cuda/msm_g2.ptx"), "\0"),
    CURVE_KERNELS,
);
const ACCUMULATE: usize = 0;
const LARGE_REDUCE: usize = 1;
const REDUCE_FIRST: usize = 2;
const REDUCE_PAIRS: usize = 3;

const SCALAR_BITS: u32 = 254;
/// `SCAN_THREADS` and `SCAN_TILE` in `msm_sort.cu`.
const SCAN_THREADS: u32 = 1024;
/// `LARGE_THREADS` in `msm_curve.cuh`.
const LARGE_THREADS: u32 = 128;
const THREADS: u32 = 256;
const ACCUMULATE_THREADS: u32 = 128;
const REDUCE_THREADS: u32 = 64;

#[derive(Clone, Copy)]
pub(crate) enum Group {
    G1,
    G2,
}

impl Group {
    fn module(self) -> &'static Module {
        match self {
            Group::G1 => &G1,
            Group::G2 => &G2,
        }
    }
}

fn alloc<T>(len: usize) -> DeviceVec<T> {
    DeviceVec::device_malloc(len.max(1)).expect("Failed to allocate device vector")
}

/// The (digit, base) pairs of `n` scalars sorted by digit, shared by the MSMs over them.
pub(crate) struct MsmSort {
    n: u32,
    c: u32,
    digits: u32,
    pf: u32,
    /// Buckets of more pairs are split into tasks of at most this many.
    large: u32,
    max_large_tasks: u32,
    max_large_buckets: u32,
    counts: DeviceVec<u32>,
    offsets: DeviceVec<u32>,
    large_task_offsets: DeviceVec<u32>,
    tile_sums: DeviceVec<[u32; 2]>,
    hist: DeviceVec<u32>,
    class_cursors: DeviceVec<u32>,
    pairs: DeviceVec<u32>,
    work: DeviceVec<[u32; 4]>,
    large_list: DeviceVec<u32>,
    info: DeviceVec<u32>,
}

impl MsmSort {
    /// `None` if the precomputed bases do not cover all digits.
    pub(crate) fn new(n: usize, c: u32, pf: u32) -> Option<Self> {
        let digits = SCALAR_BITS.div_ceil(c);
        if digits > pf || !(10..=24).contains(&c) {
            return None;
        }
        let n = u32::try_from(n).ok()?;
        let buckets = 1usize << c;
        let max_pairs = n as usize * digits as usize;
        u32::try_from(n as usize * pf as usize).ok()?;
        let large = (4 * max_pairs / buckets).clamp(32, SCAN_THREADS as usize - 1) as u32;
        let max_large_buckets = (max_pairs / (large as usize + 1)).min(buckets) as u32;
        let max_large_tasks = (2 * max_pairs / large as usize) as u32;
        Some(Self {
            n,
            c,
            digits,
            pf,
            large,
            max_large_tasks,
            max_large_buckets,
            counts: alloc(buckets),
            offsets: alloc(buckets + 1),
            large_task_offsets: alloc(buckets + 1),
            tile_sums: alloc(buckets / SCAN_THREADS as usize),
            hist: alloc(SCAN_THREADS as usize),
            class_cursors: alloc(SCAN_THREADS as usize),
            pairs: alloc(max_pairs),
            work: alloc(max_large_tasks as usize + buckets),
            large_list: alloc(max_large_buckets as usize),
            info: alloc(4),
        })
    }

    fn buckets(&self) -> u32 {
        1 << self.c
    }

    pub(crate) fn sort<F>(&mut self, scalars: &DeviceSlice<F>, stream: &IcicleStream) {
        assert_eq!(size_of::<F>(), 32, "BN254 scalars expected");
        assert_eq!(scalars.len(), self.n as usize, "scalar count mismatch");
        let module = SORT.get();
        let buckets = self.buckets();
        let per_scalar = (self.n.div_ceil(THREADS).max(1), 1);
        let per_bucket = (buckets.div_ceil(THREADS), 1);
        // SAFETY: the arguments match the kernels, and the buffers have the sizes they index.
        unsafe {
            module.launch_args(
                ZERO,
                per_bucket,
                THREADS,
                stream,
                Args::new().arg(ptr_mut(&mut self.counts)).arg(buckets),
            );
            module.launch_args(
                COUNT,
                per_scalar,
                THREADS,
                stream,
                Args::new()
                    .arg(ptr(scalars))
                    .arg(self.n)
                    .arg(self.c)
                    .arg(self.digits)
                    .arg(ptr_mut(&mut self.counts)),
            );
            module.launch_args(
                ZERO,
                (SCAN_THREADS / THREADS, 1),
                THREADS,
                stream,
                Args::new().arg(ptr_mut(&mut self.hist)).arg(SCAN_THREADS),
            );
            let tiles = buckets / SCAN_THREADS;
            module.launch_args(
                SCAN_TILES,
                (tiles, 1),
                SCAN_THREADS,
                stream,
                Args::new()
                    .arg(ptr(&self.counts))
                    .arg(self.large)
                    .arg(ptr_mut(&mut self.tile_sums))
                    .arg(ptr_mut(&mut self.hist)),
            );
            module.launch_args(
                SCAN_TOP,
                (1, 1),
                SCAN_THREADS,
                stream,
                Args::new()
                    .arg(ptr_mut(&mut self.tile_sums))
                    .arg(tiles)
                    .arg(self.large)
                    .arg(ptr(&self.hist))
                    .arg(ptr_mut(&mut self.class_cursors))
                    .arg(ptr_mut(&mut self.info)),
            );
            module.launch_args(
                SCAN_APPLY,
                (tiles, 1),
                SCAN_THREADS,
                stream,
                Args::new()
                    .arg(ptr_mut(&mut self.counts))
                    .arg(self.large)
                    .arg(ptr(&self.tile_sums))
                    .arg(ptr_mut(&mut self.offsets))
                    .arg(ptr_mut(&mut self.large_task_offsets))
                    .arg(buckets),
            );
            module.launch_args(
                SCATTER,
                per_scalar,
                THREADS,
                stream,
                Args::new()
                    .arg(ptr(scalars))
                    .arg(self.n)
                    .arg(self.c)
                    .arg(self.digits)
                    .arg(self.pf)
                    .arg(ptr_mut(&mut self.counts))
                    .arg(ptr_mut(&mut self.pairs)),
            );
            module.launch_args(
                ORDER,
                per_bucket,
                THREADS,
                stream,
                Args::new()
                    .arg(ptr(&self.offsets))
                    .arg(buckets)
                    .arg(self.large)
                    .arg(ptr(&self.large_task_offsets))
                    .arg(ptr_mut(&mut self.class_cursors))
                    .arg(ptr_mut(&mut self.work))
                    .arg(ptr_mut(&mut self.large_list))
                    .arg(ptr_mut(&mut self.info)),
            );
        }
    }
}

/// A level of the bucket reduction (see `msm_curve.cuh`); the last one writes the results.
struct Level<P> {
    n_out: u32,
    seg_log: u32,
    t_z: Option<(DeviceVec<P>, DeviceVec<P>)>,
}

/// Scratch for a batch of MSMs over one [`MsmSort`].
pub(crate) struct MsmScratch<P> {
    batch: u32,
    sums: DeviceVec<P>,
    partials: DeviceVec<P>,
    levels: Vec<Level<P>>,
    /// The latency-bound bucket reduction runs here, ahead of other MSMs' accumulation.
    tail: IcicleStream,
    accumulated: Event,
    reduced: Event,
}

impl<P> Drop for MsmScratch<P> {
    fn drop(&mut self) {
        let _ = self.tail.destroy();
    }
}

impl<P> MsmScratch<P> {
    pub(crate) fn new(sort: &MsmSort, group: Group, batch: usize) -> Self {
        let first = match group {
            Group::G1 => 4,
            Group::G2 => 3,
        };
        let mut levels = Vec::new();
        let mut log_n = sort.c;
        while log_n > 0 {
            let seg_log = log_n.min(if levels.is_empty() { first } else { 1 });
            log_n -= seg_log;
            levels.push(Level {
                n_out: 1 << log_n,
                seg_log,
                t_z: (log_n > 0).then(|| (alloc(batch << log_n), alloc(batch << log_n))),
            });
        }
        Self {
            batch: batch as u32,
            sums: alloc(batch * sort.buckets() as usize),
            partials: alloc(batch * sort.max_large_tasks as usize),
            levels,
            tail: high_priority_stream(),
            accumulated: Event::new(),
            reduced: Event::new(),
        }
    }
}

/// `results[m]` = the MSM of the scalars of `sort` with the bases at `bases[m * bases_stride..]`.
pub(crate) fn msm<C: Curve>(
    group: Group,
    sort: &MsmSort,
    bases: &DeviceSlice<Affine<C>>,
    bases_stride: usize,
    scratch: &mut MsmScratch<Projective<C>>,
    results: &mut DeviceSlice<Projective<C>>,
    stream: &IcicleStream,
) {
    let batch = scratch.batch;
    assert_eq!(results.len(), batch as usize, "result count mismatch");
    let points = sort.n as usize * sort.pf as usize;
    assert!(
        bases.len() >= (batch as usize - 1) * bases_stride + points,
        "too few bases"
    );
    let module = group.module().get();
    let buckets = sort.buckets();
    let max_work = sort.max_large_tasks + buckets;
    // SAFETY: the arguments match the kernels, and the buffers have the sizes they index.
    unsafe {
        module.launch_args(
            ACCUMULATE,
            (max_work.div_ceil(ACCUMULATE_THREADS), batch),
            ACCUMULATE_THREADS,
            stream,
            Args::new()
                .arg(ptr(bases))
                .arg(bases_stride as u64)
                .arg(ptr(&sort.pairs))
                .arg(ptr(&sort.work))
                .arg(ptr(&sort.info))
                .arg(ptr_mut(&mut scratch.sums))
                .arg(buckets)
                .arg(ptr_mut(&mut scratch.partials))
                .arg(sort.max_large_tasks),
        );
        scratch.accumulated.order(stream, &scratch.tail);
        let stream = &scratch.tail;
        module.launch_args(
            LARGE_REDUCE,
            (sort.max_large_buckets.clamp(1, 256), batch),
            LARGE_THREADS,
            stream,
            Args::new()
                .arg(ptr(&sort.large_task_offsets))
                .arg(ptr(&sort.large_list))
                .arg(ptr(&sort.info))
                .arg(ptr(&scratch.partials))
                .arg(sort.max_large_tasks)
                .arg(ptr_mut(&mut scratch.sums))
                .arg(buckets),
        );
        let (mut t_in, mut z_in) = (ptr(&scratch.sums), 0u64);
        let results = ptr_mut(results);
        for (k, level) in scratch.levels.iter_mut().enumerate() {
            let (t_out, z_out) = level
                .t_z
                .as_mut()
                .map_or((0, results), |(t, z)| (ptr_mut(t), ptr_mut(z)));
            let grid = (level.n_out.div_ceil(REDUCE_THREADS), batch);
            if k == 0 {
                module.launch_args(
                    REDUCE_FIRST,
                    grid,
                    REDUCE_THREADS,
                    stream,
                    Args::new()
                        .arg(t_in)
                        .arg(ptr(&sort.offsets))
                        .arg(level.n_out)
                        .arg(level.seg_log)
                        .arg(t_out)
                        .arg(z_out),
                );
            } else {
                assert_eq!(level.seg_log, 1);
                module.launch_args(
                    REDUCE_PAIRS,
                    grid,
                    REDUCE_THREADS,
                    stream,
                    Args::new()
                        .arg(t_in)
                        .arg(z_in)
                        .arg(level.n_out)
                        .arg(t_out)
                        .arg(z_out),
                );
            }
            (t_in, z_in) = (t_out, z_out);
        }
    }
    scratch.reduced.order(&scratch.tail, stream);
}

#[cfg(test)]
mod tests {
    use ark_bn254::{Fr, G1Projective, G2Projective};
    use ark_ec::CurveGroup;
    use ark_ff::UniformRand;
    use icicle_bn254::curve::{CurveCfg, G2CurveCfg, ScalarField};
    use icicle_core::{
        curve::{Affine, Curve, Projective},
        msm::MSM,
    };
    use icicle_runtime::{
        memory::{DeviceSlice, DeviceVec, HostSlice},
        stream::IcicleStream,
    };
    use rayon::prelude::*;

    use super::{Group, MsmScratch, MsmSort, msm};
    use crate::{
        bridges::{ark_scalars_to_device_into, ark_to_icicle_affine},
        gpu_utils::{msm_into, msm_window, upload_points_async},
    };

    const PF: i32 = 16;

    fn bases<C: Curve + MSM<C>, G: CurveGroup>(
        n: usize,
        stream: &IcicleStream,
    ) -> DeviceVec<Affine<C>> {
        let mut rng = rand::thread_rng();
        let base = (0..64).map(|_| G::rand(&mut rng)).collect::<Vec<_>>();
        let points = (0..n)
            .into_par_iter()
            .map(|i| base[i % 64] * G::ScalarField::from((i / 64 + 1) as u64))
            .collect::<Vec<_>>();
        let points = G::normalize_batch(&points)
            .iter()
            .map(ark_to_icicle_affine::<_, C>)
            .collect::<Vec<_>>();
        upload_points_async(&points, n, stream, PF)
    }

    fn icicle<C: Curve<ScalarField = ScalarField> + MSM<C>>(
        bases: &DeviceSlice<Affine<C>>,
        scalars: &DeviceSlice<ScalarField>,
        stream: &IcicleStream,
    ) -> Projective<C> {
        let mut out = DeviceVec::device_malloc(1).unwrap();
        msm_into(bases, scalars, &mut out, stream, PF);
        let mut host = [Projective::<C>::zero()];
        out.copy_to_host(HostSlice::from_mut_slice(&mut host))
            .unwrap();
        host[0]
    }

    /// The device MSMs (three G1 ones and a G2 one over the same scalars) against Icicle's.
    #[test]
    #[ignore = "needs a GPU"]
    fn msm_matches_icicle() {
        crate::load_backend_from_env_and_set_device(0);
        let stream = IcicleStream::create().unwrap();
        let mut rng = rand::thread_rng();
        for n in [40_000, 300_000] {
            // Small scalars and repeated ones make large buckets.
            let scalars = (0..n)
                .map(|i| match i % 4 {
                    0 => Fr::from((i % 5) as u64),
                    1 => -Fr::from(1 + (i % 3) as u64),
                    _ => Fr::rand(&mut rng),
                })
                .collect::<Vec<_>>();
            let mut dev_scalars = DeviceVec::<ScalarField>::device_malloc(n).unwrap();
            ark_scalars_to_device_into(&scalars, &mut dev_scalars);
            let g1 = bases::<CurveCfg, G1Projective>(3 * n, &stream);
            let g2 = bases::<G2CurveCfg, G2Projective>(n, &stream);
            stream.synchronize().unwrap();

            let mut sort = MsmSort::new(n, msm_window(n) as u32, PF as u32).unwrap();
            let mut g1_scratch = MsmScratch::new(&sort, Group::G1, 3);
            let mut g2_scratch = MsmScratch::new(&sort, Group::G2, 1);
            let mut g1_out = DeviceVec::device_malloc(3).unwrap();
            let mut g2_out = DeviceVec::device_malloc(1).unwrap();
            sort.sort(&dev_scalars, &stream);
            let stride = n * PF as usize;
            msm(
                Group::G1,
                &sort,
                &g1,
                stride,
                &mut g1_scratch,
                &mut g1_out,
                &stream,
            );
            msm(
                Group::G2,
                &sort,
                &g2,
                0,
                &mut g2_scratch,
                &mut g2_out,
                &stream,
            );
            let mut got_g1 = [Projective::<CurveCfg>::zero(); 3];
            g1_out
                .copy_to_host(HostSlice::from_mut_slice(&mut got_g1))
                .unwrap();
            let mut got_g2 = [Projective::<G2CurveCfg>::zero()];
            g2_out
                .copy_to_host(HostSlice::from_mut_slice(&mut got_g2))
                .unwrap();

            for (m, got) in got_g1.iter().enumerate() {
                let expected = icicle(&g1[m * stride..(m + 1) * stride], &dev_scalars, &stream);
                assert!(*got == expected, "G1 MSM {m} mismatch for n = {n}");
            }
            assert!(
                got_g2[0] == icicle(&g2, &dev_scalars, &stream),
                "G2 mismatch for n = {n}"
            );
        }
    }
}
