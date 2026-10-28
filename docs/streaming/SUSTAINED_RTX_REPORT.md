# Streaming R0 sustained RTX qualification

<!-- Copyright (c) 2026 Tuomas Pietilä. SPDX-License-Identifier: CC-BY-4.0. -->

This report covers bounded offline-emulated source-point replay on the RTX 4090. It is
not a live sensor transport or general-purpose throughput qualification. Source windows,
map capacity, scene distribution and output contracts are part of every result.

The accepted 4,096-point baseline remains two map-enabled 1.5M trials; the earlier two
1M point-only trials remain independent evidence. All queue failures remain FAIL, including
the original 2M cases and subsequent failures at other window sizes. No threshold, queue
slot count, model weight or preprocessing implementation was relaxed.

## Results and accepted scope

**PASS: 2M incoming source points/s with a bounded semantic map at 16,384 points/window.**
All three distinct licensed scenes passed two measured 30-second trials. Two further 60-second
trials on scene A and one 60-second trial on scene C passed. Every counted source observation
received a fresh trained-U-Net classification; map-label reuse and geometry-only counts were zero.
The highest tested passing configured input rate is 2M; rates above 2M were **NOT_RUN**.

The table reports actual fully classified/emitted throughput including drain. Incoming and
freshly classified rates agree for these complete-coverage trials. Semantic-map delta rates
also agree because each timestamped observation changes or refreshes a cell. These are update
events, including repeated updates of the same voxel; unique map-cell creation is counted separately.
All listed trials have 100% coverage, zero dropped/duplicated/delayed points, input/output queue
high-water marks of one frame each, and zero queue depth after drain. Sampled whole-device GPU
memory peaked at 524 MiB, including the desktop. CPU RSS is the process high-water mark.

| Scene/trial; 16,384 points/window | Measured seconds | Fresh emitted Mpoints/s | Source age p50 / p95 / p99 ms | CPU peak MiB | GPU mean utilization¹ | Gate |
|---|---:|---:|---|---:|---:|---|
| A, 30s #1 | 30.007 | 1.999405 | 12.32 / 16.25 / 17.30 | 342.6 | 18.6% | PASS |
| A, 30s #2 | 30.007 | 1.999419 | 12.32 / 16.25 / 17.30 | 343.2 | 18.6% | PASS |
| B, 30s #1 | 30.007 | 1.999467 | 12.32 / 15.99 / 16.78 | 339.0 | 18.6% | PASS |
| B, 30s #2 | 30.007 | 1.999461 | 12.32 / 15.99 / 17.30 | 339.0 | 18.7% | PASS |
| C, 30s #1 | 30.007 | 1.999358 | 13.11 / 17.30 / 18.35 | 357.7 | 20.1% | PASS |
| C, 30s #2 | 30.007 | 1.999313 | 12.85 / 17.30 / 18.35 | 357.5 | 20.1% | PASS |
| A, 60s #1 | 60.006 | 1.999711 | 12.58 / 16.25 / 17.30 | 343.2 | 19.8% | PASS |
| A, 60s #2 | 60.006 | 1.999734 | 12.32 / 15.99 / 16.78 | 343.0 | 19.9% | PASS |
| C, 60s #1 | 60.006 | 1.999747 | 13.11 / 17.30 / 18.87 | 357.7 | 21.6% | PASS |

¹ Whole-process samples include initialization and warmup, with a 250 ms polling sleep plus
query overhead. Exact sample times and dynamic SM/memory clocks remain in the raw receipts.
No clock or power setting was changed. The three 60-second trials' maximum scheduling lags
were 0.280, 0.316 and 0.262 ms; all were below 50 ms. Their p99 source ages were 16.78–18.87 ms,
well below 250 ms. Each classified 120,012,800 measured points. Fresh-execution reconciliation
passed: the scene-A trials each issued 31,227 actual successful enqueues including warmup;
scene C issued 35,746. Those equal the observed inference-block totals. Successfully reconstructed
backend point counts equal service fresh-classification counts, with one package load/context.

[All 79 trial receipts](../../archive/evidence/sustained_rtx_trials.json) and the
[78 point-stream trial rows](../../archive/evidence/sustained_trial_table.csv) retain every PASS/FAIL,
latency quantile, incoming/fresh/map rate, coverage/drop count, queue bound, memory result,
stage sum, semantic histogram, source/model/binary identity and canonical raw-trial hash.
The remaining trial is the explicitly separated model-only benchmark.

## Window size, failure limits and scene distribution

| Window points | Requested map-enabled rate | Result |
|---:|---:|---|
| 4,096 | 500k | Original two 30s PASS trials; optimized identical-input PASS retained |
| 4,096 | 600k / 700k / 750k | Optimized two 30s PASS trials at each; new reference attempts FAIL |
| 4,096 | 1.5M | Original two accepted 30s PASS trials retained |
| 4,096 | 1.6M | FAIL: bounded input queue filled; higher rates stopped for this sweep |
| 4,096 | 2M | Original unpinned and pinned FAILs retained |
| 8,192 | 1.5M / 2M | Two 1.5M PASS; 2M FAIL during measured interval |
| 16,384 | 1.5M / 1.6M / 1.75M / 1.9M / 2M | Repeated 30s PASS at every rate |
| 32,768 | 1.5M / 2M | Repeated 30s PASS; 2M p99 ages 29.88 / 30.41 ms |

At 8,192 points and 2M, overload was visible after 8.860 measured seconds. The inference
worker averaged 3.362 ms/window (2.437M points per **active worker second**), while its
p99 was 6.685 ms against a 4.096 ms arrival interval. The input queue reached two frames;
32,768 points were dropped including cancellation cleanup. Scheduling lag was only 0.598 ms.
The output sink averaged 1.035 ms/window, p99 1.343 ms, and its queue high-water mark was one.
Thus input-worker service variation exceeded the two-slot buffering allowance. The mean
active service estimate is not a sustainable rate. CPU geometry consumed 32.4% of measured
worker time, TensorRT 24.3%, and H2D/zeroing 18.5%. The FAIL remains a FAIL despite low
end-to-end p99 and high average throughput. The successful larger-window configurations
reduce block/model-call overhead per point and provide more points per two-frame queue.

| Scene | Licensed source basename | Records | Nominal records/m²² | Global 1m occupied voxels | Mean occupied voxels/window³ | Mean inference blocks/window³ |
|---|---|---:|---:|---:|---:|---:|
| A | `TEST-0447_6390-002819414.laz` | 50,096 | 20.04 | 4,821 | 2949.5 | 3.934 |
| B | `TEST-0437_6398-002899556.laz` | 37,913 | 15.17 | 3,929 | 3056.0 | 4.000 |
| C | `TEST-0944_6564-005900112.laz` | 400,654 | 160.26 | 24,458 | 4610.6 | 4.502 |

² Record count divided by XY bounding-box area; an original-corpus density descriptor, separate
from incoming points/s. ³ 16,384-point 2M trials, dynamic per-window R4 block origins. Scene C
has a larger height span, much denser points and lower intensity values. Its source distribution
and model semantic histogram differ substantially from A/B. All seven model IDs occurred in
each measured 2M scene trial; this is channel coverage, not a new semantic-accuracy assessment.
[Corpus hashes, raw source-class histograms and density definitions](../../archive/evidence/streaming_scene_corpora.json)
remain separate from predicted semantic IDs. Raw source classes are never silently remapped.

Two 30s scene-C 1M trials with 250 mm sinusoidal X motion, alternating 25% rank-X occlusion,
50% spatial overlap replay and up to 1 ms arrival jitter passed, with full fresh coverage.
Missing-frame/duplicate/out-of-order/sequence-gap tests reject visibly. Requested cancellation
remains an explicit FAIL with counted incomplete coverage/drops. Original and corrected
out-of-order receipts are preserved; the original erroneous four-late-point assertion was not
used to weaken the assembler. Short duration is an explicit final-gate FAIL, not a sustained PASS.

## Optimization A/B and bottlenecks

The semantic-map validation change removed redundant classified-event JSON construction and
kept the same scalar correctness checks. On identical 15,003,648-point input, map processing
fell from **21.9044 s to 1.7448 s (12.55× for that stage)**. Point digest
`11635222839647314507` and delta digest `5574432551988241481` match exactly. This is a
CPU map-processing improvement, not neural inference acceleration. Full snapshots, deltas,
point fields, timestamp/identity metadata, eviction, repeated observations, class changes,
late observations, aging and coordinate mismatch rejection passed the old/new map oracle.

Pinned staging preserved classifications, source coverage, input/logit bytes, point/map digests
and tracked allocation cleanup. It did not raise the ceiling: 4,096-point 2M still FAILed.
At identical 1.5M input, total worker time increased from 23.657 s to 25.341 s; H2D/zeroing
increased from 2.789 s to 5.485 s, while D2H fell from 2.259 s to 0.738 s. It remains an
experimental option **disabled by default**. Dynamic clock settings were unchanged; exact
instantaneous clock identity is not asserted. No large CUDA voxelization or map rewrite began.

The following is the first 30s scene-A 2M/16,384 trial. Worker stages are serial within one
window; the map sink and producer overlap the worker. Queue waits are separately observed
latencies and must not be added to stage sums to claim wall-clock time.

| Stage | Cumulative seconds | Interpretation |
|---|---:|---|
| CPU geometry | 8.5553 | 41.0% of worker window time; largest worker stage |
| TensorRT inference | 4.5310 | CUDA stream elapsed region |
| H2D and tensor zeroing | 3.1116 | Includes stream submission gaps |
| Host adapter / orchestration / sync residual | 3.0411 | Includes grouping, result reconstruction and host allocations |
| D2H | 0.9423 | Labels, semantic IDs and error flag |
| CPU point conversion | 0.7497 | Outside backend window timer |
| CUDA tensor preparation | 0.4754 | Dense feature preparation |
| GPU point reconstruction | 0.1854 | Fresh voxel logits scattered to original source indices |
| Worker window total | 20.8422 | 20.842 s active service over 30.007 s source duration |
| Semantic map | 6.1582 | Parallel sink; includes validation subset below |
| Map scalar validation subset | 1.2102 | Subset of map time, not an additional stage |
| Output identity/digest validation | 1.3779 | Parallel sink, outside map timer |
| Input queue wait | 0.1157 | Cumulative waiting across windows |
| Output queue wait | 0.1065 | Cumulative waiting across windows |

The source-to-output latency is measured directly (p99 17.30 ms in this trial).
[The 163-window host/CUDA timeline](../../archive/evidence/streaming_host_cuda_timeline.json) records
source, worker and map-sink overlap. GPU utilization averaged 18.6% over the process; low
utilization alone does not establish a saturated-device ceiling. Aggregate stage measurements
identify CPU geometry and per-block host/model overhead as higher-value targets than CUDA
feature preparation or semantic-map relocation.

Model-only: actual repeated trained-U-Net execution sustained **5,847.29 enqueues/s** over
30.006 s, with 175,456 measured calls. CUDA-event per-enqueue **32-call batch averages** have
p50/p95/p99 0.143 / 0.242 / 0.246 ms. The last block represents 2,948 source points; its
17.24M point-equivalents/s exclude geometry, reconstruction, queues and output. That mode
emits **zero** freshly classified source points and is not an E2E throughput claim.

Cold/persistent: representative scene-A initialization took 293.0 ms and source decode
20.8 ms. First output arrived 382.4 ms after session-start timing, including the deliberate
50 ms replay anchor and first acquisition window. The first backend window took 6.869 ms,
versus 5.690 ms persistent mean in the measured interval. Startup and persistent rates remain
separate. The two original 1M point-only receipts remain unchanged, and two fresh-counter
reruns match digest `2288047153224766389` with 100% coverage and zero drops.

## Output contracts and next optimization

Complete fresh source events and semantic-map deltas are independently counted. Point-plus-map
2M trials maintain both simultaneously; the map stores latest semantic ID/timestamp and explicit
unknown confidence, without feeding cached labels into inference. Current 65,536-cell capacity
exceeded the tested corpora's 3,929–24,458 global occupied cells. The 64-cell oracle exercised
correct eviction, aging and coordinate guards. Sustained performance at full 65,536-cell capacity
with continuous eviction remains **NOT_RUN**; current oldest-revision eviction scans the map.
Real moving-sensor pose/CRS integration, long multi-scene surveys and calibrated uncertainty
also remain unqualified.

Optional JSON encoding runs in the sink and excludes disk I/O. The 16,384-point/50k case
completed coverage but FAILed p99 age (369.10 ms): its acquisition window alone spans 327.68 ms.
The compatible 4,096-point/50k rerun PASSed, p99 94.37 ms. The original FAIL remains recorded.
Asynchronous streaming LAZ/COPC export remains **NOT_RUN**; accepted R4 file preservation and
seven-class QGIS mapping remain unchanged and independently regression-tested.

**Recommended next optimization:** implement and A/B test a bounded CPU geometry fast path
for zero spatial overlap, retaining the current geometry implementation as the oracle.
The current generic path tests conservative neighboring blocks for each point even when only
one can own it. Preserve exact half-open boundaries, floating-point corner handling, block/member
ordering and source indices; use the generic fallback at ambiguous numeric boundaries. Geometry
is 41% of worker time in the representative 2M trial. Quantify the improvement with identical
windows and traces before considering a GPU grouping rewrite. Separately profile full-map
churn before selecting an eviction data structure. No further speedup is claimed here.

## Reproduction and evidence boundaries

Fresh source-point throughput counts every original observation reconstructed from its
window’s newly inferred voxel logits. Multiple points may share one freshly predicted voxel;
model calls, occupied voxels and source-point counts are recorded separately.

The benchmark uses the typed C++ source stream API, complete fresh classified-point callbacks,
and optionally the bounded semantic map. Each emitted point is checked against the deterministic
source for its original record ID, timestamp, geometry and intensity. Semantic IDs 0–6 and the
explicit LAS projection `[1, 2, 200, 6, 9, 17, 201]` are preserved. No map label is read back
into a model result. Each actual TensorRT enqueue and successfully reconstructed point is
independently counted and reconciled with inference blocks and completed service windows.

LAZ records are decoded once before replay. Input UNIX times and frame boundaries are emulated;
original LAS GPS times/return decoding are not claimed. The finite licensed source corpus is
cycled in source-record order, including wraparound. Distinct temporal observations have distinct
frame IDs/timestamps. Scene motion/occlusion tests are deterministic transformations of this
replay, not independently captured moving-sensor surveys. IGN/FRACTAL data retains Etalab-2.0
attribution and is not imported into Git or relicensed as MIT.

Frame size changes the model's spatial context and batching. Each configuration is compared
against its own identical-boundary R4 offline reference. Predictions between different window
sizes are not asserted equivalent. All windows retain the frozen 32³ dense tensor, features,
voxelization, zero spatial overlap and first-wins policy. Two queue slots remain two slots;
the number of points buffered per slot necessarily grows with the configured frame size.

A PASS requires at least 30 measured seconds after five seconds of warmup, complete coverage,
zero drops/duplicates, p99 source-to-output age ≤250 ms, scheduling lag ≤50 ms, two frames per
queue, oldest pending source age ≤500 ms, process RSS ≤1 GiB, measured RSS range ≤64 MiB and
sampled device-wide usage ≤4 GiB. Histograms include every point/window in bounded storage;
reported percentile upper bins have ≤3.125% relative width. Rates include output drain.
Warmup failure and measured-interval failure are reported separately.

Model initialization, source decode and first-window costs are separate from persistent rates.
CPU geometry/conversion, stream-elapsed CUDA event regions, inference, reconstruction, transfers,
map processing, scalar validation, digest checking and queue waits are separately recorded.
CUDA event regions include stream elapsed time and possible host submission gaps; they are
not exclusive kernel active time. Worker, sink and source execute concurrently. Their cumulative
service times must not be summed as wall-clock latency. The bounded host/CUDA timeline records
exact host spans and CUDA durations within host submission/synchronization brackets; it does
not assert a shared absolute CPU/GPU clock or substitute for a CUPTI kernel trace.

Hardware clocks and power settings were left unchanged. Sampled GPU usage/utilization/power
use a 250 ms polling sleep plus query overhead and include initialization and warmup; they are labeled as whole-process samples. Other reported
compute workloads cause the controller to stop its own trial. No AGX/Thor or unrelated RTX
workload was interrupted. Nothing was pushed, released or published.

Reproduce using the existing qualified dependencies and frozen packages, in fresh directories:

```bash
python3 tools/window_campaign.py --native "$BUILD/synrex-sustain" \
  --reference "$REFERENCE_BUILD/synrex-sustain" --model "$UNET" \
  --sources "$SCENE_A" "$SCENE_B" "$SCENE_C" \
  --source-commit "$SOURCE_COMMIT" --out "$NEW_CAMPAIGN"
python3 tools/summarize_sustained.py \
  --campaign reference="$REFERENCE_CAMPAIGN/receipt.json" \
  --campaign mapopt="$MAPOPT_CAMPAIGN/receipt.json" \
  --campaign pinned="$PINNED_CAMPAIGN/receipt.json" \
  --campaign windows="$NEW_CAMPAIGN/receipt.json" --out "$NEW_SUMMARY"
```

See [the protocol](SUSTAINED_PROTOCOL.md), [the exact temporal regression receipt](../../archive/evidence/sustained_out_of_order_regression.json)
and [the optimization A/B receipt](../../archive/evidence/streaming_optimization_ab.json).

## Exact source and final checks

Qualified benchmark/runtime source commits and local changes:

| Commit | Change |
|---|---|
| `7c93cc610f5eaf39ec5db7ff5d536530833dedf6` | Sustained native benchmark and full-sample timing |
| `3f933ff` | Temporal protocol regression fixtures |
| `707375b198b2bc3f27524a57bd7fecd207466f4f` | Remove redundant map-event JSON allocation |
| `8e3d97a936264e3019ed42b62ab35148e09dae33` | Optional pinned staging; remains disabled |
| `9409b8f19443883322957cdb1175af5c4c2625a5` | Fresh counts, cleanup, window matrix, exact map oracle and timelines |
| `6a4e6d103de9e0e93f844d1d299ec8af29a2dbf9` | Correct final-gate failure-phase reporting |

The window sweep used `9409b8f`; the 60-second confirmations and final regressions used
`6a4e6d1`. The latter changes only post-measurement failure-phase reporting. The model backend,
geometry and kernels are identical between those commits. Archive SHA256 values are
`08df2da0b42aebd0beb1cc760249fbcf8e43d36de81dbb0d54f95356cdd2a37a` and
`7fa9d3f641c0aab91204d7cd8e307e4d617fe3452a6cf2ffdc9f609c6c85ed8a`, respectively.
Both embedded Git IDs and all 275 extracted file contents were verified after execution.

Frozen trained engine SHA256:
`2f20d4fd17c1e1eb3399ac450f84a2f1c7737ab69e2098ae383dc1911a1455cd`.
Frozen synthetic engine SHA256:
`4dd03d4c2c6291e98145c64ec2fb7e174ba69d7b7e1728e84cf7357e51772f59`.
The training-checkpoint identity remains the accepted R4 identity
`788b65c42bb5d50d9509a56d38368f51890c4a93300d53578e290d193456c7c4`;
production executes the verified engine and does not load Python weights. All package files
were hash-checked before/after. CPU geometry and CUDA kernel source bytes remain identical
to accepted R4 `3f1f3fa`; accepted tag object `0598b83f8c190c86a5071eb0cf5592eb7788ba1b`
and the R4 final-audit lineage `c3451b6` are preserved.

Hardware: RTX 4090/SM89, 24,564 MiB device total; AMD Ryzen 7 7700X, 16 logical CPUs,
Linux under WSL2; NVIDIA driver 610.88, CUDA compiler 13.2.86/runtime 13020, TensorRT 10.16.1.
No device clock, power, access policy or unrelated dependency was changed. The resource guard
observed no competing reported compute workload. Device usage returned to 130 MiB at 0% after
tests; storage remained 357 GiB free. CPU tests: 87 pytest PASS and two CTest PASS; five GPU
skips were subsequently executed on RTX. ASan/UBSan: two CTest PASS. Final streaming regression
controller: 14 commands PASS, including the same binary for both models and no Python on PATH.
R4 CUDA/TensorRT LAZ, numerical, source-preservation, sidecar and relocation tests: five PASS.

The original reference-map setup FAIL (uploaded executable lacked execute permission) and
its successful retry are retained. The first R4 relocation invocation had four FAIL/one PASS
because the controller used nonexistent `CUDA_ROOT/lib64`. The unchanged binaries passed
all five with existing `CUDA_ROOT/lib`; no reinstall or source workaround occurred. Original
logs/hashes and exact gates are in [the final gate receipt](../../archive/evidence/streaming_final_gates.json)
and [the preserved setup receipt](../../archive/evidence/streaming_setup_failure.json).
The earlier JSON final-gate FAIL incorrectly labeled its phase as “none”; its original receipt
is retained, while the consolidated record identifies the final gate. The corrected runtime's
short-duration negative test explicitly reports FAIL / final_gate.

Exact map replay can be reproduced with an unchanged old implementation and the current test:

```bash
mkdir "$MAP_REFERENCE_SOURCE"
git archive 7c93cc6 native | tar -x -C "$MAP_REFERENCE_SOURCE"
c++ -O3 -std=c++17 -I"$MAP_REFERENCE_SOURCE/native/include" \
  -isystem "$JSON_INCLUDE" tests/native/map_replay.cpp \
  "$MAP_REFERENCE_SOURCE/native/stream.cpp" \
  "$MAP_REFERENCE_SOURCE/native/package.cpp" \
  "$MAP_REFERENCE_SOURCE/native/no_gpu.cpp" -lcrypto -pthread -o "$REFERENCE_MAP"
python3 tools/verify_stream_ab.py --default "$DEFAULT_EQUIVALENCE" \
  --pinned "$PINNED_EQUIVALENCE" --reference-map "$REFERENCE_MAP" \
  --optimized-map "$BUILD/synrex-map-replay-tests" --out "$NEW_MAP_AB"
```

The current-tree release/license/secret-pattern checks PASS. The inherited R0 `--history`
mode requires exactly one repository commit and no remotes, so its single-commit precondition
FAILs for this continued branch. Its history gate is explicitly NOT_RUN. A separate read-only
scan using the same patterns covers all 98 newly reachable objects across the seven commits
through `2a78629`: PASS, no pattern findings or model/point-cloud binary paths. See
[the scoped history receipt](../../archive/evidence/streaming_history_audit.json). The inherited checker
and accepted R4 audit remain unchanged.

All raw campaigns, complete map outputs, tensors and initial failures remain in separate
local/RTX scratch directories. No source-point clouds, model binaries, private addresses or
secrets enter Git. The committed receipt tables preserve every result and bind omitted raw
artifacts by SHA256. This source line remains unpublished; local evidence commits are separate
from the qualified executable source IDs above.

## Gated ARM64 continuation

Orin/SM87: use the existing `cmake/targets/orin.cmake` only on an available physical ARM64
Orin, with matching installed CUDA/TensorRT 10 SDKs and resource guards appropriate to that
platform. Build and test the exact qualified source commit natively. Create a new target engine
from the independently licensed frozen model export in a separate package; never overwrite the
RTX package or copy its serialized engine as qualification. Re-run CPU schema/geometry tests,
CUDA tensor/logit comparisons, seven-class/source preservation tests, cleanup tests and the
same duration/coverage/queue/latency/memory gates before reporting any target throughput.

Thor/SM110: repeat this process separately using `cmake/targets/thor.cmake` and a physically
available Thor with its matching SDKs. Verify native SM110 code generation and build a distinct
engine/package. Requalify memory sharing, pinned staging and power behavior on that machine;
RTX measurements cannot establish them. Neither platform was accessed in this campaign.
Both physical performance gates remain **NOT_RUN**. LitePT-S/TorchSparse++ remains a separate
future sparse-adapter track and is not part of this dense-adapter R0 qualification.
