# Zero-overlap geometry and spatial-map churn qualification

<!-- Copyright (c) 2026 Tuomas Pietilä. SPDX-License-Identifier: CC-BY-4.0. -->

This continuation preserves the accepted 2M source at local tag
`streaming-r0-2m-frozen-95e80bb` and records its source/archive/receipt hashes in
[the freeze manifest](../../evidence/geometry_2m_freeze.json). All prior PASS and FAIL receipts,
including smaller-window queue failures, remain unchanged. Nothing is pushed or published.
The original independently evaluated small FRACTAL U-Net, its weights, TensorRT engine,
quality evidence and accepted R4 artifacts remain frozen. These measurements apply to this
model and the named replay corpora; larger models and other datasets are unqualified.

## Measured outcome

**The optional direct geometry path sustains nominal 3M freshly classified points/s plus
map observation updates on scenes A and B, each repeated for 60 measured seconds. Scene C
passes two 60-second trials at 2.5M and FAILs at 3M.** The unchanged reference passes 2.5M on
scene A and FAILs at 3M. These are bounded offline-emulated replay results for the frozen small
trained U-Net, not a general 3M hardware qualification.

True spatial churn exposes a separate limit: a 4,096-cell map passes 50k points/s twice for
60 seconds; a 65,536-cell map passes 2,500 points/s twice for 120 seconds, including over 76
seconds after eviction starts. Full-capacity 65,536-cell churn FAILs at 2M, 50k, 25k, 10k and
5k; stop phases and losses remain in the receipts. No latency, queue or memory gate was relaxed.

The [31-trial JSON](../../evidence/geometry_rtx_trials.json) contains 24 PASS and seven FAIL
trials; the [per-trial CSV](../../evidence/geometry_trial_table.csv) contains rates, latency,
coverage, loss, queue depths, ages, memory, GPU utilization/power, map operations and output
hashes for every trial. Campaign completion status means the controller completed and preserved
its trials; it does not turn individual FAILs into PASS. Original 2M, point-only 1M and prior
map/pinned-staging measurements remain in the frozen [previous report](SUSTAINED_RTX_REPORT.md).

## Independent reference/direct/flat comparison

All six scene-A A/B trials process the same 120,012,800 measured source points over 60.0064
virtual seconds, following five seconds of warmup. The actual trained TensorRT model executes
freshly for every window. Point classifications/semantic IDs/source metadata, map delta digest
and final full snapshot digest agree across all six trials:

- Point digest: `11966976976787136549`
- Map delta digest: `1921173779110773649`
- Snapshot SHA256: `6e922f057ee53ac929976c20e89bef9cc27fc561093cb4c2128aa1598e2d9b66`

| Geometry strategy | Trials | CPU geometry, mean active s | Worker window, mean active s | Result at nominal 2M |
|---|---:|---:|---:|---|
| reference | 2 × 60s | 17.508 | 42.474 | PASS, 100% coverage |
| direct | 2 × 60s | 8.906 | 33.727 | PASS, 100% coverage |
| flat | 2 × 60s | 9.494 | 33.939 | PASS, 100% coverage |

Mean CPU geometry falls 49.1% (1.97× active geometry speed), and mean worker-window time falls
20.6%. These are measured service-region changes; neither is an end-to-end throughput multiplier.
The single binary retains the original reference algorithm and selects the strategy explicitly.
Dynamic clocks are unchanged, not locked: sampled SM clocks include idle and boost states. There
were no competing compute processes; thermal/clock variation remains a measurement limit.

For the first paired 2M trial, direct geometry divides into minimum reduction 0.289s, assignment
3.015s, tree grouping 0.421s and member sorting 5.209s. Flat grouping costs 0.133s, but global
record sorting costs 6.071s and total geometry rises to 9.454s versus direct's 8.938s. Flat's tiny
worker-total advantage in one trial is not repeatable evidence of better sustained throughput;
it was not selected for the rate sweep. The CPU microbenchmark separately records 655,360 tree
lookups and 2,120 member-capacity growths for direct, versus zero of each for flat; sorting costs
outweigh those savings on this input. Reference counters are uninstrumented zeros, not proof of
zero allocation/tree work. See the [CPU qualification](../../evidence/geometry_cpu_qualification.json).

| Measured service region, active seconds | Reference 2M t1 | Direct 2M t1 | Direct 3M scene A t1 |
|---|---:|---:|---:|
| CPU geometry | 17.445 | 8.938 | 13.204 |
| CPU point conversion | 2.536 | 1.359 | 1.971 |
| H2D and zeroing (CUDA events) | 6.350 | 6.339 | 11.885 |
| CUDA tensor preparation | 0.963 | 0.963 | 1.452 |
| TensorRT execution | 9.108 | 9.183 | 13.928 |
| GPU point reconstruction/scatter | 0.378 | 0.379 | 0.568 |
| D2H | 1.907 | 1.908 | 2.932 |
| Host adapter/orchestration/sync residual | 6.208 | 6.210 | 9.243 |
| Worker window total | 42.358 | 33.920 | 53.212 |
| Semantic map, including subset validation | 12.602 | 12.605 | 18.586 |
| Map validation subset (already in map time) | 2.497 | 2.508 | 3.547 |
| Independent full map oracle | 2.354 | 2.355 | 3.686 |
| Point identity/class digest validation | 3.320 | 3.345 | 5.601 |
| Input queue waits | 0.249 | 0.212 | 4.946 |
| Output queue waits | 0.212 | 0.214 | 0.384 |

The 3M column processes 180,011,008 measured points, so compare per point when ranking rates.
TensorRT is about 77ns/source point in that column; this is an active inference service ratio,
not a new sustained model-only throughput claim. On the worker, TensorRT (13.93s), CPU geometry
(13.20s), H2D/zeroing (11.88s) and host synchronization/orchestration (9.24s) dominate. Map service
(18.59s) runs in the separate sink, concurrently with the 53.21s worker. Queue waits and the
acceptance-to-completion aggregate include overlapping residence time and must not be added
to worker/map regions as wall latency. CUDA events measure the same CUDA stream's regions;
the host residual is a subtraction estimate, not a separate GPU kernel. Optional JSON/file I/O
is zero in these trials. A full Nsight system trace is NOT_RUN; existing host/CUDA event timing
is retained without inventing concurrency timelines.

Model initialization across the six A/B processes takes 0.318–0.376s. First-window service is
5.72–7.47ms; persistent mean worker service is approximately 5.80ms reference and 4.60ms direct.
Initialization is measured before replay and excluded from sustained rates. No cold disk/cache
reset was performed, so this is process startup, not a power-on cold-storage measurement.

## Per-trial sustained results

Scenes A/B/C are respectively `TEST-0447_6390-002819414.laz`,
`TEST-0437_6398-002899556.laz` and `TEST-0944_6564-005900112.laz` from the licensed public
FRACTAL test split (IGN, Etalab 2.0). The source hashes are recorded per trial.

Fresh points/s and map observation-update deltas/s are independently counted and equal for each
PASS in this table. This map emits one observation delta per point, including repeated cells;
these are not unique newly created voxels. Actual rates include drain time. Source latency uses
point-weighted histogram upper bounds with at most 3.125% bin width. Every PASS has full planned
measurement duration, 100% classification coverage, zero dropped/duplicate/corrupt/delayed
points, zero reused labels, two bounded queues that drain to zero, and a PASS independent map
oracle. Queue columns show separate input/output high waters; total in-flight frames can exceed
two because worker/sink are separate from each queue. All models use fresh successful TensorRT
enqueue and reconstructed-point counts reconciled with completed frames, including warmup.

| Trial (JSON ID suffix) | PASS/FAIL | Fresh/map updates per s | Source p50/p95/p99 ms | Queue in/out | Peak RSS MiB | GPU mean % / peak MiB |
|---|---|---:|---|---|---:|---|
| ab-reference-2000000-t1 | PASS | 1999709 | 13.11/16.78/17.83 | 1/1 | 345.3 | 19.8 / 524 |
| ab-reference-2000000-t2 | PASS | 1999697 | 13.11/16.78/17.83 | 1/1 | 345.7 | 19.7 / 524 |
| ab-direct-2000000-t1 | PASS | 1999719 | 11.80/15.47/16.25 | 1/1 | 345.9 | 19.8 / 524 |
| ab-direct-2000000-t2 | PASS | 1999739 | 11.53/15.47/16.25 | 1/1 | 346.2 | 19.9 / 524 |
| ab-flat-2000000-t1 | PASS | 1999753 | 11.80/15.47/16.25 | 1/1 | 345.7 | 19.6 / 524 |
| ab-flat-2000000-t2 | PASS | 1999727 | 11.80/15.47/16.25 | 1/1 | 345.8 | 19.9 / 524 |
| sweep-reference-2250000-t1 | PASS | 2249680 | 12.58/15.99/16.78 | 1/1 | 346.2 | 22.5 / 524 |
| sweep-reference-2250000-t2 | PASS | 2249638 | 12.58/15.99/16.78 | 1/1 | 345.8 | 22.3 / 524 |
| sweep-reference-2500000-t1 | PASS | 2499640 | 12.06/15.47/16.52 | 1/1 | 345.9 | 24.9 / 524 |
| sweep-reference-2500000-t2 | PASS | 2499487 | 12.85/18.35/22.02 | 2/1 | 345.9 | 24.9 / 524 |
| sweep-reference-3000000-t1 | FAIL (measured) | 2977745 | 10.49/15.73/19.92 | 2/1 | 344.9 | 12.0 / 524 |
| sweep-direct-2250000-t1 | PASS | 2249668 | 11.53/14.94/15.73 | 1/1 | 345.9 | 22.5 / 524 |
| sweep-direct-2250000-t2 | PASS | 2249667 | 11.53/14.94/15.73 | 1/1 | 346.2 | 22.5 / 524 |
| sweep-direct-2500000-t1 | PASS | 2499584 | 11.53/14.68/15.99 | 1/1 | 346.4 | 24.8 / 524 |
| sweep-direct-2500000-t2 | PASS | 2499672 | 11.27/14.42/15.99 | 1/1 | 346.3 | 24.9 / 524 |
| sweep-direct-3000000-t1 | PASS | 2999592 | 11.01/14.42/15.73 | 2/1 | 346.4 | 29.8 / 524 |
| sweep-direct-3000000-t2 | PASS | 2999575 | 11.27/14.68/16.25 | 2/2 | 346.4 | 30.0 / 524 |
| scene1-direct-3000000-t1 | PASS | 2999625 | 11.01/14.42/15.99 | 1/2 | 343.4 | 30.3 / 524 |
| scene1-direct-3000000-t2 | PASS | 2999612 | 10.75/14.16/15.47 | 2/1 | 343.4 | 29.9 / 524 |
| scene2-direct-3000000-t1 | FAIL (measured) | 2996005 | 12.06/17.30/20.97 | 2/2 | 407.4 | 26.9 / 524 |
| churn-4096-50000-t1 | PASS | 49996 | 6.68/9.70/10.75 | 1/1 | 339.6 | 13.7 / 524 |
| churn-4096-50000-t2 | PASS | 49997 | 6.55/9.44/10.75 | 1/1 | 339.8 | 13.7 / 524 |
| churn-65536-50000-t1 | FAIL (warmup) | 0 | 0.00/0.00/0.00 | 2/2 | 498.9 | 4.4 / 524 |
| churn-65536-25000-t1 | FAIL (measured) | 23593 | 6.42/11.27/60.82 | 2/2 | 499.0 | 3.0 / 524 |
| churn-65536-10000-t1 | FAIL (measured) | 9859 | 15.20/26.74/88.08 | 2/2 | 499.3 | 2.3 / 524 |
| churn-65536-2m-16384 | FAIL (warmup) | 0 | 0.00/0.00/0.00 | 2/2 | 502.8 | 2.8 / 524 |
| sceneC-2500000-t1 | PASS | 2499667 | 12.06/16.25/18.35 | 1/2 | 408.0 | 27.0 / 524 |
| sceneC-2500000-t2 | PASS | 2499655 | 12.06/15.73/17.83 | 1/1 | 408.1 | 26.9 / 524 |
| churn65536-5000-64-t1 | FAIL (measured) | 4986 | 11.01/44.04/63.96 | 2/2 | 499.5 | 4.0 / 524 |
| churn65536-2500-32-t1 | PASS | 2500 | 12.85/20.97/22.54 | 1/1 | 500.5 | 4.1 / 524 |
| churn65536-2500-32-t2 | PASS | 2500 | 12.85/20.97/22.54 | 1/1 | 500.7 | 4.1 / 524 |

All seven runtime FAILs stop on `bounded_input_queue_full`; cancellation reports the resulting
undelivered points explicitly. A FAIL's apparent near-target processing rate or low latency is
not a sustainable result. Warmup FAILs have zero measured points and zero measured-stage totals;
these zeros are not timing successes. The CSV/JSON retain each loss count, coverage fraction,
phase and completed duration. No second trial was run after an unsupported rate failed.

Scene A and B reach only 4,821 and 3,929 live global map cells; C reaches 24,458, all below the
65,536-cell cap. Scene C has about 160.26 source points/m² versus A's 20.04 and B's 15.17, different
spatial occupancy and a 38.393m height range. All seven semantic IDs are preserved; individual
scenes need not predict every class. Distinct-scene input hashes, occupancy distributions,
semantic histograms and stage timings are in each trial. Across the three representative scenes,
2.5M is the highest rate with repeated PASS evidence; rates between 2.5M and 3M were not swept.
GPU device memory peaks at 524MiB including desktop use; GPU utilization is sampled whole-process,
including initialization and warmup, not kernel occupancy. RSS and device samples are observed
peaks, not proof that every transient allocation was captured.

## Implementation and correctness boundaries

`native/geometry.cpp` remains byte-identical to the authoritative R4 implementation.
The two-argument `geometry()` API and default session behavior continue to use it. The optional
`GeometryMode::direct` and `GeometryMode::flat` strategies are in `native/geometry_fast.cpp`.
Every nonzero-overlap contract falls back to the complete original function. The strategies
use the contract's dimensions, voxel sizes and extents; no 32³/1m assumptions are hard-coded.
The frozen U-Net uses that specific configuration with exactly zero overlap.

For unambiguous interior points, direct indexing computes the owning block and original
half-open voxel index. A conservative 64-epsilon guard covers quotient/subtraction/product and
translated-origin rounding. Boundary or poorly resolved coordinates use the original candidate
search per point, preserving its possible multiple memberships and failure behavior. No boundary
is nudged, clamped into another block or reassigned by a new policy. Block origins, source indices,
voxel indices, block traversal and member ordering remain exact. Feature reductions and CUDA
preparation/scatter kernels are unchanged.

The direct strategy stages bounded membership records and uses one tree lookup per membership,
then the original voxel/source-index member sort. The flat strategy sorts all records by block,
voxel and source index, then reserves each block's exact member capacity. Its eliminated tree
lookups and member-vector growth must be weighed against its global sort cost. The initial
prototype was slightly slower than the reference; its [negative CPU receipt](../../evidence/geometry_prototype_cpu.json)
is preserved. Avoiding temporary error-message construction on successful checks improved the
optional strategies without removing validation or changing failure conditions.

CPU tests cover negative coordinates, signed zero, LAS integer/scale reconstruction, offsets
through 1e15, exact/adjacent floating-point block boundaries, coincident points, mixed occupancy,
sparse empty regions, arbitrary grids and voxel sizes, nonzero-overlap fallback, invalid points
and coordinate-range rejection. There are 398 successful mode comparisons and eight matching
unsafe-input rejections. Origins, ordered memberships and complete float32 feature tensors are
compared byte for byte. The unchanged stream schema limits XYZ to ±1e9: the initial 1e12 GPU
fixture FAILed at serialization before inference. Its [FAIL is retained](../../evidence/geometry_initial_gpu_fixture_failure.json);
stream/GPU fixtures use 1e8, while larger offsets remain in CPU geometry tests.

The full-window CPU oracle compares 12,000 deterministic 16,384-point windows from **each**
real scene, covering every stationary window used by the timed 5s+60s trials through 3M.
Further 25,000-window runs at both 256 and 32 points include the 64m/frame translation used
in churn tests. Together these cover 597,024,000 point observations, exact memberships/order
and all feature bytes.
These are CPU geometry/feature checks, never a CPU neural throughput claim. Actual GPU tests
separately compare native persistent inference with the unchanged offline R4 path, using both
frozen models in the same executable. Timed runs reconcile actual successful TensorRT enqueues,
completed windows and freshly reconstructed source points.

## Spatial map and independent validation

The production semantic-map algorithm is unchanged apart from counters. Its ordered map and
oldest-revision eviction scan remain the reference behavior. Counters distinguish observations,
existing-cell updates, insertions, removals, late rejections, emitted deltas, maximum cell count
and cells examined by eviction. A hash-table/list oracle independently checks every consumed
point's cell key, semantic ID, timestamp and eviction, the exact delta sequence, final full
snapshot and classification ages. It never supplies labels to inference. Its runtime is reported
separately and included in source-to-output latency. Confidence remains explicitly unknown.

The moving fixture translates actual licensed scene records by 64m in X per frame, in a fixed
virtual world frame. The original tile is about 50m wide, so successive patches create genuinely
new cells. This is a deterministic spatial stress trajectory, not a physically captured moving
survey or a claim about plausible flight speed. Original record IDs, emulated acquisition times
and benchmark frame boundaries are retained and checked. Every emitted point still executes
fresh trained-model inference. No map-label reuse or geometry-only input is counted as neural
classification.

A separate CPU-only full-capacity service probe first fills the map, then measures new-cell
insertions/removals and existing-cell updates independently. Dummy classes are explicit;
[its rates](../../evidence/geometry_cpu_eviction_service.json) are map-only active service rates,
not sustained neural classification. Moving-map CPU fixtures also check late observations,
class changes, age reporting, repeated observations and coordinate mismatch rejection.

## Full spatial-churn results and next optimization

| Capacity / packet | Requested points/s | Trials / measured duration | Result | Full-capacity eviction period |
|---|---:|---|---|---|
| 4,096 / 256 | 50,000 | 2 × 60s | PASS | All measured frames; >64s including warmup |
| 65,536 / 16,384 | 2,000,000 | Stops in warmup | FAIL | Fills then queue overflows |
| 65,536 / 256 | 50,000 | Stops in warmup | FAIL | Fills then queue overflows |
| 65,536 / 256 | 25,000; 10,000 | Stops in measured interval | FAIL | Not a sustained full-map run |
| 65,536 / 64 | 5,000 | Stops after ~29 measured seconds | FAIL | Only 8.79s after first eviction |
| 65,536 / 32 | 2,500 | 2 × 120s | PASS | 76.16s after first eviction each |

The 4,096-cell first PASS processes 3,000,064 measured points: 19,851 new-cell insertions/s,
30,145 existing-cell updates/s and 19,851 removals/s (drain-inclusive rates). It performs 1,286,525
removals including warmup and examines 5.27 billion cells for eviction. The 65,536-cell first
PASS processes 300,000 measured points: 1,337 insertions/s, 1,162 existing updates/s and 848
removals/s averaged over the entire measured interval, including its remaining fill phase.
Both repeats have identical point/delta/snapshot digests and operation counts. Each removes
101,781 cells in its full-capacity phase and examines 6.67 billion eviction candidates. Maximum
size is exactly the configured cap; snapshot ages and every removal are independently checked.
The first-eviction timestamp is observed after the first evicting frame, so the reported duration
is a conservative full-capacity exposure including drain. At 2,500 points/s, over 76 measured
seconds occur after that timestamp; this is not merely a fast fill-and-stop benchmark.

The CPU-only [RTX-host eviction probe](../../evidence/geometry_cpu_eviction_rtx.json) isolates
new-cell replacement from existing-cell updates with the same production map, oracle outside
the timer. At capacities 1,024 / 4,096 / 16,384 / 65,536 it measures approximately 336k / 80.7k /
12.4k / 2.96k evicting updates per active second. Existing-cell service remains about 9.11M/s at
65,536 cells. These short dummy-class probes quantify a CPU algorithmic cost; they are not
trained-model or sustained incoming-point rates. The 65,536-cell full-pipeline PASS's map stage
consumes 43.41 active seconds over the first 120-second measured interval; its independent
validation timings remain separately recorded in the trial receipt.

**Next highest-value change: replace the full-map oldest-revision scan with a bounded CPU
revision/eviction index**, keeping the current implementation and independent list/hash oracle
as references. Preserve deterministic tie ordering, repeated-observation refresh, late-point
rejection, all delta timestamps/IDs and age semantics, and benchmark fill/existing/eviction
phases separately. The measured capacity scaling warrants this bounded CPU change. It does not
yet warrant a CUDA semantic-map rewrite. For repeated-scene high-rate classification, measure
whether stable voxel/source ordering or reusable bounded scratch storage can reduce the remaining
member sort and H2D/zeroing costs before choosing another optimization. No such further change
is included in this qualified source.

## Evidence and remaining boundaries

[Final gates](../../evidence/geometry_final_gates.json) separate accepted scopes, rejected rates
and NOT_RUN targets. The [remote source/model seal](../../evidence/geometry_remote_final.json)
compares every archived source file with the actual qualified tree, records both binary hashes
and frozen package-file hashes, and confirms the RTX is idle after regression tests. Both models
pass reference/direct/flat real inference; [exact trace comparisons](../../evidence/geometry_gpu_exact.json)
check 48 classified events and 1,684 input/logit trace files byte for byte, including source
metadata and classes. No CPU label approximation substitutes for those model executions.
Full-window CPU tensor parity extends input/preprocessing correctness to every stationary
benchmark window; timed point validation independently checks identities, counts, class ranges
and map outputs, while the A/B full-output digests prove equality on identical streams. This is
not a claim that a second independent neural engine recomputed each of the billions of timed
predictions. R4 GPU regressions pass all five selected tests. CPU tests, sanitizer checks,
licensing and frozen-evidence hashes are sealed separately.

Live sensor transport, asynchronous LAZ/COPC throughput, larger models, arbitrary datasets,
ARM64/SM87, Thor/SM110 and a general 3M rate under 65,536-cell eviction are NOT_RUN or explicitly
FAIL as appropriate. Optional export remains outside this timing path. The legacy single-commit
R0 history checker is inapplicable to a continued Track B branch; the scoped new-object audit
is recorded without altering that checker or rewriting existing Git history.

## Reproduction

The core timed source is `3f97a490860529505b1ebfad03e2726e2081dbe2`; its archive SHA256 is
`56e14e26298ca2ac3f1ec26a46863f0ef2bb350d83406d721451eac62f1c5c25`.
The reference/direct/flat trials use one binary with explicit runtime selection, identical
models, deterministic inputs/times, two frames per queue, 16,384 points/window, 65,536 map
cells, output validation and unchanged dynamic clock/power settings. This controls build
and instrumentation differences. The frozen pre-optimization source/binaries remain separate.
No runtime is rebuilt between the synthetic model and trained U-Net.

Use the existing dependencies and [native build quickstart](QUICKSTART.md); RTX qualification
uses CUDA 13.2.86, TensorRT 10.16.1, SM89, CMake Release, laz-perf 3.4.0, nlohmann/json 3.11.3
and OpenSSL 3. The same CMake source builds the CPU-only oracle without the CUDA SDK.

```bash
cmake -S . -B "$BUILD" -DCMAKE_BUILD_TYPE=Release \
  -DLAZPERF_SOURCE_DIR="$LAZPERF_SOURCE" -DNLOHMANN_JSON_INCLUDE_DIR="$JSON_INCLUDE" \
  -DSYNREX_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DCMAKE_CUDA_COMPILER="$CUDA/bin/nvcc" -DTENSORRT_ROOT="$TENSORRT"
cmake --build "$BUILD" -j 4
python3 tools/geometry_campaign.py --native "$BUILD/synrex-sustain" \
  --model "$UNET" --sources "$SCENE_A" "$SCENE_B" "$SCENE_C" \
  --source-commit "$QUALIFIED_SOURCE" --out "$NEW_CAMPAIGN"
build/synrex-geometry-tests examples/unet/dense_voxel_asprs_safe.json "$NEW_CPU_RECEIPT"
build/synrex-geometry-replay-tests "$UNET" "$SCENE_A" 12000 16384 0 "$NEW_WINDOW_RECEIPT"
build/synrex-map-churn-tests "$NEW_MAP_TEST_RECEIPT"
build/synrex-map-eviction-profile "$NEW_EVICTION_RECEIPT"
python3 tools/geometry_followup.py --native "$BUILD/synrex-sustain" \
  --model "$UNET" --sources "$SCENE_A" "$SCENE_B" "$SCENE_C" \
  --skip-scenes --churn-rate 2500 --churn-points 32 --out "$NEW_LOW_CHURN"
python3 tools/verify_geometry_traces.py --root "$TRACE_ROOT" --receipt "$NEW_TRACE_RECEIPT"
```

The C++ API selects `SessionOptions::geometry`; the native benchmark exposes
`--geometry reference|direct|flat`. Defaults remain reference. `--verify-map 1` enables the
independent delta/snapshot oracle; `--churn-mm 64000` advances the virtual world trajectory.
All destinations must be new. The benchmark uses typed classified-point callbacks and map
deltas; JSON/file encoding and live sensor transport remain separate output costs.

Original gates remain: ≥60 measured seconds for accepted trials after five seconds warmup,
100% coverage, zero drops/duplicates, p99 source age ≤250ms, schedule lag ≤50ms, two slots per
queue, oldest pending source ≤500ms, RSS ≤1GiB, measured RSS range ≤64MiB and sampled device
usage ≤4GiB. GPU sampling uses a 250ms sleep plus query overhead. CPU and GPU events measure
separate service regions; producer, worker and sink overlap, so concurrent sums are not wall
latency. Individual failed trials stop visibly. Churn capacity/window differences are reported
explicitly and do not establish a 3M full-map-churn claim.

AGX/Thor were not accessed. ARM64/SM87 and SM110 remain physically NOT_RUN, with the separate
[gated continuation plan](SUSTAINED_RTX_REPORT.md#gated-arm64-continuation) unchanged. RTX engines
are not portable target qualification. No sparse adapter, GPU rewrite or new model architecture
is included.
