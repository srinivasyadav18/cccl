# PR Summary: Run-to-Run Deterministic Warpspeed Scan

**Commit:** `3c98b6dbf` — *warpspeed run_to_run deterministic using fixed order scheduling + 32 way tile aggregation for prefix sum*

**Author:** Srinivas Yadav Singanaboina

## What it does

Adds **run-to-run bit-reproducible** support to the CUB `warpspeed` device scan path (sm_100+). Previously, `run_to_run` / `gpu_to_gpu` determinism was only available for integral types with known operators. This PR extends `run_to_run` determinism to **floating-point types with `plus`** (where reduction order matters for bit-exact reproducibility), while leaving `gpu_to_gpu` integral-only.

## How it achieves determinism

Two cooperating mechanisms in `kernel_scan_warpspeed.cuh` and `look_ahead.cuh`:

1. **Fixed-order scheduling** (`kernelBody`, `dispatch_scan.cuh`)
   - Drops the dynamic `clusterlaunchcontrol` "next tile" scheduler used in the non-deterministic path.
   - Launches a grid of `min(sm_count, num_tiles)` CTAs; each CTA strides through tiles by `gridDim.x` in a fixed pattern, so any given tile is always assigned to the same CTA across runs.
   - `_PDL_GRID_DEPENDENCY_SYNC` / PDL launch is disabled when deterministic (`use_pdl = !RunToRunDeterministic`).

2. **32-way fixed-shape tile aggregation** (`warpIncrementalLookback`)
   - Non-deterministic mode reduces only the *rightmost contiguous run* of already-visible tile aggregates — a variable count, which produces a different reduction tree shape (and different FP rounding) across runs.
   - Deterministic mode **waits** until all 32 expected predecessors in the batch are visible (or the tail count at the end), then runs a fixed-width warp reduction. The reduction tree is therefore identical across runs.

## API / template surface

`scan_impl_determinism`, `dispatch`, `dispatch_with_accum`, `DispatchScan`, `DeviceScanKernelSource`, `DeviceScanKernel`, `device_scan_lookahead_body`, and `kernelBody` are all extended with a new `bool RunToRunDeterministic = false` template parameter (replacing the prior `__determinism_holder_t` plumbing in `scan_impl_determinism`).

Static asserts in `device_scan.cuh`:
- `run_to_run` ⇒ integral-with-known-op **or** floating-point + `plus` (warpspeed sm_100+).
- `gpu_to_gpu` ⇒ integral-with-known-op only.

## Supporting changes in `look_ahead.cuh`

Because the deterministic grid uses fewer CTAs than there are tiles, `gridDim.x` is no longer a safe upper bound for tile-state indices. `storeTileAggregate`, `loadTileAggregate`, `warpLoadLookback`, and `warpIncrementalLookback` now take an explicit `num_tiles` argument used in their bounds assertions and lookback termination. `numTiles` is computed once at the top of `kernelBody`.

The temp-storage sizing and the init kernel still allocate one tile-state slot per *tile* (`num_tiles`), independent of `scan_grid_dim`.

## Tests / benchmarks added

- `cub/test/catch2_test_device_scan_deterministic.cu` — Catch2 test that runs `ExclusiveScan` with `cuda::execution::determinism::run_to_run` for `float` and `double` across a range of input sizes (1, 10, 1337, 3000, 31·128, 10000, plus randomized and min/max-items cases), repeats the scan 2/5/10 times, and asserts every run produces output bit-identical to the first.
- `cub/benchmarks/bench/scan/deterministic/exclusive.cu` — new nvbench benchmark for the deterministic exclusive-scan path, guarded on `__CUDA_ARCH_LIST__ >= 1000` and `__cccl_ptx_isa >= 860`, with a `Det` axis to compare deterministic vs. non-deterministic throughput.

## Files touched

| File | Δ |
|---|---|
| `cub/cub/detail/warpspeed/look_ahead.cuh` | +73 / −45 |
| `cub/cub/device/dispatch/kernels/kernel_scan_warpspeed.cuh` | +41 / −11 |
| `cub/cub/device/dispatch/dispatch_scan.cuh` | +39 / −18 |
| `cub/cub/device/device_scan.cuh` | +24 / −15 |
| `cub/cub/device/dispatch/kernels/kernel_scan.cuh` | +9 / −6 |
| `cub/test/catch2_test_device_scan_deterministic.cu` | +64 (new) |
| `cub/benchmarks/bench/scan/deterministic/exclusive.cu` | +146 (new) |

Net: **+393 / −98** across 7 files.

