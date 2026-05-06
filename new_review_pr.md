# PR Review: `cuco::static_map` → `cudax/__cuco/` migration

Branch: `cuco_static_map` (6 commits, +7001/−8 across 23 files)
Baseline for comparison: `cucollections/include/cuco/` (upstream)
Target: `cccl/cudax/include/cuda/experimental/__cuco/`

Reviewer focus: API surface, correctness, migration fidelity, testability, and CCCL-style conformance.

---

## Commit sequence

| # | SHA | Subject | Notes |
|---|-----|---------|-------|
| 1 | `9a6f714b` | initial migration of OA and static_map | Bulk port |
| 2 | `874285cc` | cleanups | |
| 3 | `a1337e97` | temporary WAR for call to ~buffer from cuda::counting_iterator[] | Test-only `nv_diag_suppress 20011` |
| 4 | `f2820a6a` | use shared memory buffer flushing in kernels;cleanups | Perf-critical shmem path |
| 5 | `b0d0702a` | refactor extent's to use size_type like std::span | Template API change |
| 6 | `1194f024` | simlify primes usage | Miller-Rabin replaces static table |

---

## Overall assessment

**Strong port.** The migration preserves cuco's open-addressing design, lifts the public API into CCCL conventions (double-underscore internals, `::cuda::std::`, `_CCCL_*` macros, public PascalCase template params), and makes two non-trivial improvements over upstream:

1. **Span-style capacity (commit 5).** `_Capacity = dynamic_extent` as a `size_t` NTTP replaces the old `extents<size_t,N>` wrapping. Users write `static_map<int,int,512>` instead of `static_map<int,int,extents<size_t,512>>`. Semantically equivalent, ergonomically much better, and consistent with `std::span`.
2. **Primality via Miller-Rabin (commit 6).** The multi-megabyte `__primes[]` lookup table and its `__lower_bound`-based probe is replaced with a deterministic constexpr Miller-Rabin, usable for any `uint64_t`. Binary size drops and there's no implicit upper bound on requested capacity.

The migration is largely ready; the items below are tightening, not blockers.

---

## Major (should address before merge)

### M1. Disabled storage-extent static_assert
`open_addressing_ref_impl.cuh:140`:
```cpp
// static_assert(is_bucket_extent_v<typename StorageRef::extent_type>, ...);
// TODO: how to re-enable this check?
```
This invariant (storage-ref carries a bucket extent type) is preserved in upstream cuco and silently disabled here. Either re-enable with the current type machinery, delete the comment, or convert to a concept/trait on the ref.

### M2. NVCC diagnostic suppression in tests (`nv_diag_suppress 20011`)
`test/cuco/static_map/test_static_map.cu:11`:
```cpp
#if defined(__CUDACC__)
#  pragma nv_diag_suppress 20011
#endif
```
Labelled as a temporary WAR in the commit message, but with no tracking link and no mention in code of when it can be removed. Needs an issue reference inline or a note in the MR description. (Root cause per project memory: `~buffer` call via `cuda::counting_iterator::operator[]`.)

### M3. Thrust iterator usage in test despite deprecation
`test_static_map.cu` still uses `thrust::device_vector`, `thrust::counting_iterator<int>`, `thrust::sequence`, `thrust::transform`. The surrounding CCCL has already deprecated `thrust::constant_iterator` / `thrust::counting_iterator` (commits `1dac745604`, `4eb027ce47`) — the same break that forced this PR to replace them internally. Tests will hit `-Werror` + `#1444-D` as soon as CCCL main drops the deprecation shims. Migrate test to `cuda::counting_iterator` / raw `device_vector`-equivalents now.

### M4. Bucket / CG-size / capacity alignment is silent
For static `_Capacity`, `compute_capacity<N>()` rounds up to `ceil(N/stride)*stride` (linear probing) or `next_prime(ceil(N/stride))*stride` (double hashing), where `stride = cg_size * bucket_size`. If a user writes `static_map<K,V, 10, ..., double_hashing<2,H>, 3>`, the *reported* `capacity_v` is the adjusted one, but the adjustment is invisible at the call site. A `@note` in the class docblock explaining how the requested `_Capacity` differs from `capacity_v` — with a small example — would avoid confusion.

---

## Minor

### API / documentation

- **Constructor count.** `static_map` has 5 ctors with 7 parameters each (via SFINAE). Consider an options struct or a named-argument helper for the optional params (`mr`, `stream`, `probing_scheme`, `pred`). Current form is functional but verbose.
- **`ref()` has no doxygen.** Every other accessor does — add one-liner describing lifetime semantics (borrows from the owning map).
- **`compute_capacity<N>()` vs `compute_capacity(size_type)`.** Same name, different callsite syntax (`::template` needed for the NTTP form in dependent contexts). Worth a `@note` showing both forms.
- **`sizeof(_Tp) == 4 || sizeof(_Tp) == 8` assertion** in `static_map_ref` silently restricts the mapped type. Mention this in the class docblock, not just as a static_assert.

### Code quality

- **Repeated `// TODO atomic_ref::load if insert operator is present` (6×)** in `open_addressing_ref_impl.cuh`. Either consolidate behind a helper or file one ticket and reference it.
- **Duplicate sentinel-handling** flagged at `open_addressing_ref_impl.cuh:2138`. Author already noted this — fine as out-of-scope, but track it.
- **`// TODO include` at `open_addressing_ref_impl.cuh:1222, 1277`** is cryptic — rewrite the comment to describe the missing include or delete it.
- **`mutable _MemoryResource __memory_resource`** in `open_addressing_impl.cuh:110` is correct (const observer methods need scratch-alloc for counters / CUB temp storage), but deserves a one-line comment so future readers don't assume it's a mistake.

### Tests (`test_static_map.cu`, 8 cases, 385 lines)

Coverage gaps, in rough priority order:

| Area | Status |
|------|--------|
| Device ref APIs (contains, find, insert_or_assign, insert_or_apply) | ✓ covered by `[container]` test |
| Static / dynamic capacity, `capacity_v`, `compute_capacity` | ✓ |
| Static-capacity shmem sizing via `capacity_v` | ✓ |
| Erasure on static capacity | ✓ |
| Load-factor constructor | ✓ |
| `insert_and_find` / `insert_if` / `insert_if_async` | **missing** |
| `retrieve` / `retrieve_outer` / `retrieve_all` | **missing** — APIs exist on `static_map` with no test coverage |
| `rehash` / `rehash_async` | **missing** |
| `for_each` / `for_each_async` | **missing** |
| `find_if` / `contains_if` with custom predicate | **missing** |
| Custom `_ProbeEqual` / `_ProbeHash` via `rebind_*` | **missing** |
| `_BucketSize > 1` path | **missing** |
| `double_hashing` probing scheme | **missing** — only `linear_probing<1>` exercised |
| Thread scopes other than `thread_scope_device` | **missing** |
| Exceptions thrown from `__make_valid_extent` | **missing** |

No benchmarks. cucollections has nvbench suites for each of these ops — porting at least a smoke benchmark would catch perf regressions in the shmem-flushing path (commit 4) and in the hot insert/find loops.

### Build / portability

- `test_static_map.cu` uses `-Werror` diagnostic suppression globally for `20011` rather than per-call — narrower scoping would catch unintended suppressions.
- `_CCCL_HAS_INT128()` fallback in `prime.hpp` is verified to work in constexpr context, but untested on MSVC (where `__int128` is unavailable). Worth a CI matrix check.

---

## Nitpicks

- Commit subject `simlify primes usage` → `simplify` typo.
- `probing_scheme_impl.cuh` uses `class _Extent` in template signatures. Unrelated to the user-facing `_Extent` that was replaced — but the name collision across the codebase is confusing. Consider `_IndexExtent` or `_ProbeExtent`.
- `__utility/strong_type.cuh` had a +14/−0 change but no obvious new functionality. Confirm the delta is intended.
- `CMakeLists.txt` for tests shows `+4` lines — the single test target. Consider grouping future `static_map_*.cu` variants under a glob.
- Header include orderings are consistent CCCL style (system first, then cuda/std, then cudax) across new files — well done.

---

## Security / correctness spot-checks

No concerns found in the Miller-Rabin implementation — standard bases, correct decomposition, overflow-safe `__mod_mul` under both `__int128` and Russian-peasant paths. `__next_prime(0|1|2)` returns `2` as documented.

The overflow guard in `__make_valid_extent_double_hash` (`__size > numeric_limits<_SizeType>::max() / __stride`) is correct and meaningful, unlike the previous `__size > numeric_limits<_SizeType>::max()` which was always false.

Kernel-attribute macro usage is consistent (`_CCCL_KERNEL_ATTRIBUTES` throughout — no stale `CCCL_DETAIL_KERNEL_ATTRIBUTES` post-rebase).

---

## Recommendations for the MR description

1. Mention that this port intentionally **changes the template signature** of `static_map` and `static_map_ref` vs. upstream cuco (span-style `_Capacity` instead of `extents<>`).
2. Call out the two enhancements over upstream: span-style capacity, Miller-Rabin primes.
3. Explicitly list deferred items (the 6× `atomic_ref::load` TODO, bucket-extent static_assert, diagnostic suppression) with tracking links.
4. Note GPU test coverage gaps vs cuco's upstream suite — list what's in/out so reviewers don't have to diff.

---

## Verdict

**Ready to merge with minor fixups.** M1–M4 and the test-coverage gaps are the only items I'd want resolved before shipping. Everything else can follow in a cleanup PR.
