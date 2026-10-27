# Spatial eviction index qualification on RTX 4090

<!-- Copyright (c) 2026 Tuomas Pietilä. SPDX-License-Identifier: CC-BY-4.0. -->

This continuation freezes the accepted geometry/churn checkpoint `958c418` before changing the
map. The [freeze manifest](../../evidence/eviction_baseline_freeze.json) preserves the complete
source identity and all historical PASS/FAIL receipts. The original R4 source, weights, model
packages, predictions, release tags and public main remain unchanged. No Track B push or release
is authorized or performed; AGX and Thor are not accessed.

**The frozen trained U-Net sustains nominal 2M freshly classified source points/s plus map
observation updates with genuine spatial churn at the full 65,536-cell capacity**, in two
60-second trials using 16,384-point windows. Actual drain-inclusive rates are 1,999,706 and
1,999,697/s. Both have 100% coverage, zero drops/duplicates/delayed points and unchanged gates.
The previous scan-based 2M warmup queue FAIL remains preserved. Repeated 100k and 1M churn
qualification also passes; the initial 100k objective and 500k stretch objective are exceeded.

[Final PASS/FAIL/NOT_RUN gates](../../evidence/eviction_final_gates.json),
[all 36 trial receipts](../../evidence/eviction_rtx_trials.json) and
[complete per-trial metrics CSV](../../evidence/eviction_trial_table.csv) contain 34 accepted
sustained PASSs and two expected cancellation FAILs. This is bounded offline-emulated replay of
this small frozen model and named licensed scenes, not a general live-sensor or larger-model
throughput guarantee. Stable 3M scene A/B and 2.5M scene C performance remains qualified.

## Implementation and actual policy

The map evicts by oldest accepted unsigned revision, not by timestamp. Existing-cell older
timestamps are rejected; equal timestamps refresh recency even when no delta is emitted. There
is no TTL expiration. The [design and complexity analysis](CPU_EVICTION_INDEX.md) explains these
rules, intrusive links, exact revision-wrap fallback, reset, thread confinement and memory bounds.
`--map-policy reference|indexed` keeps the scan selectable; indexed is the default. Ordered
lookup and snapshot traversal remain deterministic. No stale priority queue is introduced.

The exact qualified native source is `1397b033b4dc5a286928054dc75af885b3912ca2`; its archive SHA256
is `01bd7ce98f406f7f72c162c7b35af6b81f67f2263bdd47f1c5b5b8c92a1a1984`. Later auxiliary test and
controller commits do not modify native inference or the qualified map implementation. The
[provenance record](../../evidence/eviction_source_provenance.json) hashes changed source and
unchanged geometry, CUDA, TensorRT, package and session-contract files. The frozen before binary
is the accepted `3f97a49` GPU build (all its native source bytes match checkpoint `958c418`), with
SHA256 `a52d815dc923931845a85d1e080ce8d61c8ad6c9e95de95b226d263742a45a97`.

## Where the original O(N) cost occurs

The only full resident-map walk inside ordinary updates is the reference `min_element` victim
scan. In the RTX-host CPU service probe at capacity 65,536, selecting 1,024 eviction victims takes
0.379569s with the reference, versus 0.00002301s with the intrusive index. The instrumented whole
replacement probe measures about 2,693 versus 2.41M replacements per active second. These short
CPU-only dummy-class numbers are not neural or sustainable incoming-point rates.

Detailed optional timers separate key calculation, lookup, insertion, existing update, victim
selection, removal, index maintenance, delta construction and full snapshot construction. The
profile includes operation counts and cell occupancy. Snapshot construction still walks all
cells and allocates bounded JSON output; it takes about 78ms in the reference CPU probe and is
outside the sustained latency-critical loop. Its cost remains O(N). Expiration is reported as
disabled because the frozen map has no such policy. Lookup/insertion/erase remain O(log N).
Index relinking and victim lookup are O(1); the intrusive index adds 40 payload bytes per cell,
2.5MiB at capacity 65,536. The reference selector has the same enlarged cell layout, so the
original frozen binary is retained for authoritative before-performance comparisons.

See the [reference](../../evidence/eviction_rtx_cpu_reference.json) and
[indexed](../../evidence/eviction_rtx_cpu_indexed.json) detailed CPU receipts. Per-observation clocks
are disabled in sustained trials; coarse native stage timing and all output validation remain on.


| CPU profile region, capacity 65,536 | Reference active ms | Indexed active ms | Reference / indexed ns per accepted observation |
|---|---:|---:|---|
| coordinate_key | 2.7848 | 2.7126 | 41.21 / 40.14 |
| lookup | 2.1140 | 2.0295 | 31.28 / 30.03 |
| insertion | 2.9723 | 2.8673 | 43.98 / 42.43 |
| existing_update | 0.0168 | 0.0170 | 0.25 / 0.25 |
| victim_selection | 379.5690 | 0.0230 | 5616.26 / 0.34 |
| removal | 0.1354 | 0.0408 | 2.00 / 0.60 |
| index_maintenance | 0.0000 | 1.1367 | 0.00 / 16.82 |
| delta_construction | 1.6794 | 1.6253 | 24.85 / 24.05 |

This profile includes initial fill, 1,024 replacements and 1,024 existing updates: 67,584
accepted observations total. Normalized costs describe that mix; victim selection per actual
replacement is about 370,673ns reference versus 22.47ns indexed. Detailed clock overhead is
material at the smallest regions, so these are diagnostic comparisons, not precision kernel
latency claims. Full snapshots are separate from these update totals.

## Independent semantic and memory checks

The reference scan, indexed map and independent hash/list oracle agree on exact delta keys,
semantic IDs, timestamps, eviction victims/order, full snapshots, ages and counters. Tests cover
all capacities 1–64, then 257, 1,024, 4,096, 16,384 and 65,536, including a forced-collision oracle,
coincident points, negative/adjacent boundaries, large offsets, equal timestamps, class changes,
older observations, identity mismatch, reset and forced unsigned revision wrap. Production
ordered lookup has no hash-collision behavior. There are five native CTests, five ASan/UBSan
CTests and 87 Python CPU regressions; all pass. An additional explicit temporal-policy test
passes release and ASan/UBSan at five map resolutions, including out-of-order new cells, exact
expected eviction victims and non-expiring full-width ages. Five physical R4 GPU preservation
tests pass.

A strengthened memory phase accepts every one of 2,097,152 observations with advancing source
timestamps, checks exact reference deltas/snapshots, traverses all live index links and confirms
65,536 entries. Release RSS stays exactly 194,633,728 bytes before/after. ASan RSS falls from
894,525,440 to 744,542,208 bytes; allocator/sanitizer retention is distinct from index storage.
The earlier fixed-timestamp memory test mostly exercised late rejection; its original receipt
is preserved, with the gap and corrected gate explicitly recorded in the
[test-correction receipt](../../evidence/eviction_test_correction.json).

Both frozen synthetic Conv3D and trained FRACTAL U-Net run through the same native executable.
Prepared tensor bytes agree with the offline path and maximum logit error is zero. All source
classes and IDs match. Owned device and pinned allocations return to zero after session teardown.
Native execution without Python on PATH, explicit packet loss and out-of-order rejection pass.
These short diagnostic tests are [regression evidence](../../evidence/eviction_gpu_regressions.json),
not sustainable throughput measurements. Larger neural models and datasets are unqualified.

## Matched frozen/indexed sustained comparison

All runs use the unchanged trained U-Net, deterministic licensed scene A, the same per-case
source windows, acquisition times, 64m/frame world translation, map resolution/capacity, two-frame
queues and independent per-point/map oracle. One fresh frozen run reproduces each accepted
fixture; both indexed repetitions use the same inputs. The original two accepted baseline
repetitions remain preserved separately. Classifications are fresh model executions; no temporal
map labels or geometry-only input are counted.

| Churning map / packet | Incoming target | Duration | Frozen map active time | Indexed map active time | Exact point/delta/snapshot parity |
|---|---:|---:|---:|---:|---|
| 4,096 cells / 256 points | 50,000/s | 60s | 21.6273s | 0.5265 / 0.5287s | PASS |
| 65,536 cells / 32 points | 2,500/s | 120s | 42.6491s | 0.1176 / 0.1100s | PASS |

These are stage-time reductions at fixed input rates, not classification throughput multipliers.
Both cases pass full coverage, zero drops/duplicates, bounded map/queues and unchanged resource
and latency gates. The 65,536-cell case includes more than 76 seconds at full-capacity eviction.
The independent snapshot digests are respectively
`669156104aa244fef596d7ff73bfd5520027a8d8a2b15eba084d6ae029adaa1d` and
`b3c21d6ef9a62051d01960ac4c3ddbc69042cf38656f08ded61f63132f083ab3` for every matching run.

## Sustained high-churn rates collected locally

The [churn trial JSON](../../evidence/eviction_churn_trials.json) and
[per-trial CSV](../../evidence/eviction_churn_table.csv) preserve all 21 matched/sweep trials:
20 sustained PASSs and one deliberately cancelled warmup FAIL. All accepted runs have complete
source-point coverage, zero dropped/duplicate/corrupt/delayed points, fresh model execution and
a PASS independent map oracle. Every high-churn map reaches its exact configured cap. The
historical smaller-window and full-map queue FAILs remain frozen, even where indexed runs now
pass at the same or higher input rates.

| Full 65,536-cell churn window | Offered incoming points/s | Repetitions × measured seconds | Delivered fresh/map updates/s | p99 source age ms |
|---|---:|---|---|---|
| 256 | 50,000 | 2 × 60s | 49,998.2 / 49,998.3 | 8.91 / 8.91 |
| 256 | 100,000 | 2 × 120s | 99,998.9 / 99,999.0 | 4.33 / 4.33 |
| 4,096 | 100,000 | 2 × 120s | 99,996.9 / 99,997.0 | 45.09 / 45.09 |
| 4,096 | 250,000 | 2 × 60s | 249,982.9 / 249,984.1 | 20.45 / 20.45 |
| 4,096 | 500,000 | 2 × 60s | 499,970.3 / 499,971.6 | 12.32 / 12.32 |
| 4,096 | 750,000 | 2 × 60s | 749,943.9 / 749,953.7 | 10.75 / 10.75 |
| 4,096 | 1,000,000 | 2 × 60s | 999,940.7 / 999,943.0 | 10.49 / 10.75 |

Incoming rate, delivered fresh classifications and map observation-delta rates are separately
recorded in the CSV; all coincide for accepted runs, within the same drain-inclusive denominator.
Map updates include repeated observations, not just new voxels. At 1M with 4,096-point windows,
about 192,800 new voxels/s are inserted and evicted while about 807,140/s update existing cells.
The two trials each perform 11,568,766 measured insertions/removals and zero resident-cell scans.
Map processing takes 8.323/8.407s per measured minute, with the independent map oracle taking
4.819/4.947s separately. Source p50 is 6.03ms, p95 8.91/9.18ms and p99 10.49/10.75ms. These
figures include the unchanged output-validation workload; model-free probe rates are excluded.

The same trials spend 38.966/38.948s in the inference worker. In the first trial, CPU geometry
is 4.265s, point conversion 0.657s, upload/zeroing 12.122s, CUDA feature preparation 0.969s,
TensorRT 10.796s, GPU reconstruction 0.407s, D2H 3.179s and host adapter/sync residual 7.228s.
Map time and worker time overlap and must not be summed as wall latency. CUDA event upload
regions include pageable staging/command-issue gaps and zeroing; these measurements do not
establish pure PCIe bandwidth as the bottleneck. Events and device buffers are already persistent.
The next optimization should profile block upload and synchronization boundaries, then consider
bounded staging/command batching with exact block-order parity. The prior negative pinned-host
experiment remains preserved; pinning alone is not assumed to improve throughput. A GPU map
rewrite is not justified by the current map service cost.

Across locally collected accepted trials, the sampled maximum scheduling lag is 4.05ms, peak
host RSS is at most 503.8MiB, and device-used memory peaks at 524MiB including desktop use. Every
queue stays within its two-frame capacity and drains to zero. The per-trial CSV retains exact
queue high waters, oldest source age, RSS range, GPU utilization/power, operation rates and hashes.
Native 64ms cancellation deliberately FAILs during warmup with no measured points; zero measured
latency there is not a performance success. The separate post-fill cancellation is recorded below.

## Stable-scene preservation

The completed stable campaign reproduces each frozen case and runs two indexed repetitions,
all for 60 measured seconds. Every case has exact point/delta/snapshot digest and map-operation
parity. Both indexed repetitions preserve all accepted rate, latency, queue and memory gates.

| Stable case, 16,384-point windows | Map active seconds, frozen | Map active seconds, indexed t1/t2 | Source p99 ms, frozen / indexed t1 / indexed t2 |
|---|---:|---|---|
| Scene A, 2M points/s | 12.353 | 10.639 / 10.491 | 15.99 / 15.99 / 15.99 |
| Scene A, 3M points/s | 18.513 | 15.299 / 15.316 | 15.47 / 15.47 / 15.73 |
| Scene B, 3M points/s | 17.928 | 14.968 / 15.023 | 15.47 / 14.94 / 14.94 |
| Scene C, 2.5M points/s | 18.725 | 16.229 / 16.353 | 17.30 / 17.30 / 17.30 |

Across these cases, indexed map service is about 13–17% lower than the frozen run; there is
no loss of the tested stable-scene throughput. Exact per-trial latencies, queues, host/device
memory and hardware samples are retained in the complete receipts. These are observations of
this frozen small model and named scenes, not a promise of unlimited-duration or larger-model
performance. Scene C's original 3M FAIL remains frozen; this campaign reruns its accepted 2.5M
rate and does not claim a new scene-C 3M result.

The temporary [Tailscale identity-check blocker](../../evidence/eviction_access_blocker.json)
was [resolved by owner authentication](../../evidence/eviction_access_resolution.json). All completed
stable receipts have now been retrieved with verified SSH. No host-key or access-policy setting
was changed. Historical NOT_RUN entries in the initial blocker describe that temporary state.

## Full-capacity 2M revisit and cancellation

Both 16,384-point-window trials run for 60.0064 measured seconds after five seconds warmup,
with the same trained weights, geometry, 64m/frame spatial translation, map configuration,
queues and validation as the pre-index 2M failure. Each delivers 120,012,800 measured source
points and map observation deltas, inserts/evicts 21,613,729 measured cells, and scans zero
resident cells for victim selection. Peak map occupancy remains 65,536. Both include all of the
measured interval under full-capacity churn; reported post-first-eviction exposure is 64.826s
including the full part of warmup and drain. The earlier scan-based trial stops in warmup and
cannot supply a same-length output digest; it remains FAIL, never normalized into a PASS.
Each new run independently checks every map delta/victim and final snapshot against the oracle.
Both repeated output/snapshot digests and operation counts are identical.

| Full-map trial | Incoming / fresh / map observation updates per s | New cells and evictions/s | Existing-cell updates/s | Source p50/p95/p99 ms | Peak RSS / GPU MiB | GPU mean utilization / mean power |
|---|---:|---:|---:|---|---|---|
| t1 | 1,999,706.3 | 360,137.5 | 1,639,568.8 | 13.11/17.30/17.83 | 506.54 / 524 | 19.83% / 16.67W |
| t2 | 1,999,696.6 | 360,135.8 | 1,639,560.9 | 13.11/17.30/17.83 | 506.41 / 524 | 19.76% / 16.62W |

The first trial's worker spends 34.952s on 120 million points: CPU geometry 9.554s (member
sorting 5.373s and assignment 3.380s within that region), TensorRT 9.307s, upload/zeroing 6.431s,
host adapter/sync residual 6.390s, D2H 1.922s, point conversion 1.626s, CUDA preparation 0.970s
and GPU reconstruction 0.379s. The sink concurrently spends 16.295s on map processing (including
2.485s validation), 7.751s on its independent map oracle and 3.255s on point identity/digests.
Those concurrent totals must not be added as wall latency. Map active cost is about 136ns per
accepted observation for this workload, with O(1) indexed victim selection; measured operation
mix and packet size matter when comparing active costs to other rows.

**Recommended next optimization:** keep the qualified map index and target the inference worker.
At the 16,384-point configuration, CPU geometry and TensorRT are the largest individual worker
regions; member sorting alone costs 5.37s. Investigate bounded stable voxel bucketing/scratch
reuse against the unchanged sort reference, preserving exact voxel/source order. Separately
profile upload/zeroing and synchronization, which total another substantial service region;
do not infer pure transfer-bandwidth limits from CUDA event intervals containing issue gaps.
The rejected flat/global-sort and pinned-staging experiments stay preserved. No further source
optimization or 3M full-capacity churn claim is made in this campaign, and current evidence does
not warrant a GPU semantic-map rewrite or new model architecture.

A 6,500ms cancellation fires after map fill and during the measured interval. The native trial
correctly remains FAIL (`requested_cancellation`): 4,096 points are explicitly counted as dropped,
and delivered measured coverage is 97.2222%. The map remains at capacity, performs 27,466 measured
removals, and its complete accepted-output prefix and final snapshot pass the independent oracle.
This validates coherent cancellation, not sustainable throughput. Warmup and measured cancellation
failures remain separately visible in the receipts and table.

## Reproduction and boundaries

Build the exact native source with the existing quickstart/CMake dependencies: C++17 Release,
CUDA 13.2.86, TensorRT 10.16.1, SM89, laz-perf 3.4.0, nlohmann/json 3.11.3 and OpenSSL 3. Do not
rebuild engines or alter weights. The controller runs only native inference; Python supplies
standard-library orchestration and hardware sampling. Use a verified idle RTX allocation and
new output paths for every invocation:

```bash
python3 tools/eviction_campaign.py --native "$BUILD/synrex-sustain" \
  --baseline "$FROZEN_BUILD/synrex-sustain" --model "$UNET" \
  --sources "$SCENE_A" "$SCENE_B" "$SCENE_C" --source-commit "$QUALIFIED_SOURCE" \
  --phase matched --out "$NEW_MATCHED"
# Run sweep, stable, full-window and faults phases sequentially, each with a new output directory.
build/synrex-map-eviction-profile "$NEW_PROFILE" reference
build/synrex-map-index-tests "$NEW_PARITY"
```

The original gates remain p99 source-to-output ≤250ms, scheduling lag ≤50ms, two slots per queue,
oldest pending source ≤500ms, RSS ≤1GiB, measured RSS range ≤64MiB and sampled device usage ≤4GiB.
Accepted rates require two complete ≥60-second measured repetitions after five seconds warmup;
120-second cases distinguish initial fill from full-capacity churn. A failed queue or resource
gate stops the trial visibly. No burst result or warmup failure is advertised as sustainable.

The moving source translates genuine licensed IGN/FRACTAL scene points in a fixed world frame;
this is a deterministic spatial stress trajectory, not captured vehicle motion. Different
window sizes have different neural context/voxel occupancy and are reported separately; no
cross-window prediction equivalence is claimed. GPU dynamic clocks/power settings are unchanged,
not locked. CPU/GPU service regions overlap across producer/worker/sink, so summing their active
times does not produce wall latency. Full snapshot generation, final JSON encoding and optional
file exports are separate costs; LAZ/COPC export throughput is NOT_RUN in this campaign.

No public publishing, GPU geometry rewrite, new architecture, sparse adapter or target-port
qualification is included. Orin/SM87 and Thor/SM110 remain physically NOT_RUN with the existing
gated continuation plans unchanged.

## Every trial and final preservation seal

All accepted rows have 100% coverage, zero dropped/duplicate/corrupt/delayed source points,
zero reused semantic labels and zero geometry-only ingestion. Input/output queues stay within
two frames each and drain to zero. The CSV retains exact counts, operation rates, queue ages,
scheduling lag, RSS range, GPU utilization/power and point/delta/snapshot hashes for every row.
Latency values are point-weighted histogram upper bounds (maximum bin width 3.125%). GPU samples
cover initialization, warmup and measurement, at a 250ms sleep plus query overhead; memory values
are observed peaks, not an allocation trace. Full-capacity measurements are distinguished from
initial fill. A failed or cancelled trial is never advertised as sustainable from its partial rate.

| Trial (receipt ID suffix) | Native status / phase | Delivered fresh points and map updates/s | Source p50/p95/p99 ms | Queue peak in/out | Peak RSS / GPU MiB | GPU mean % |
|---|---|---:|---|---|---|---:|
| original4096-frozen | PASS | 49,996 | 6.68/9.70/10.75 | 1/1 | 339.5 / 524 | 13.7 |
| original4096-indexed-t1 | PASS | 49,998 | 4.98/7.60/8.91 | 1/1 | 339.8 / 524 | 13.7 |
| original4096-indexed-t2 | PASS | 49,998 | 4.98/7.73/8.91 | 1/1 | 339.9 / 524 | 13.7 |
| original65536-frozen | PASS | 2,500 | 12.85/20.97/22.54 | 1/1 | 500.7 / 524 | 4.1 |
| original65536-indexed-t1 | PASS | 2,500 | 8.26/14.16/14.68 | 1/1 | 502.7 / 524 | 4.1 |
| original65536-indexed-t2 | PASS | 2,500 | 8.26/14.16/14.42 | 1/1 | 502.5 / 524 | 4.1 |
| churn65536-p256-50000-t1 | PASS | 49,998 | 4.98/7.86/8.91 | 1/1 | 502.4 / 524 | 13.6 |
| churn65536-p256-50000-t2 | PASS | 49,998 | 4.85/7.60/8.91 | 1/1 | 502.2 / 524 | 13.6 |
| churn65536-p256-100000-t1 | PASS | 99,999 | 2.82/4.06/4.33 | 1/1 | 502.9 / 524 | 22.9 |
| churn65536-p256-100000-t2 | PASS | 99,999 | 2.82/4.06/4.33 | 2/1 | 502.9 / 524 | 22.8 |
| churn65536-p4096-100000-t1 | PASS | 99,997 | 25.17/44.04/45.09 | 1/1 | 503.7 / 524 | 2.6 |
| churn65536-p4096-100000-t2 | PASS | 99,997 | 24.64/42.99/45.09 | 1/1 | 503.8 / 524 | 2.6 |
| churn65536-p4096-250000-t1 | PASS | 249,983 | 12.32/19.92/20.45 | 1/1 | 503.0 / 524 | 6.1 |
| churn65536-p4096-250000-t2 | PASS | 249,984 | 12.06/19.92/20.45 | 1/1 | 503.1 / 524 | 6.1 |
| churn65536-p4096-500000-t1 | PASS | 499,970 | 8.00/11.80/12.32 | 1/1 | 503.2 / 524 | 12.0 |
| churn65536-p4096-500000-t2 | PASS | 499,972 | 8.00/11.80/12.32 | 1/1 | 503.0 / 524 | 12.1 |
| churn65536-p4096-750000-t1 | PASS | 749,944 | 6.82/9.70/10.75 | 1/1 | 502.8 / 524 | 17.9 |
| churn65536-p4096-750000-t2 | PASS | 749,954 | 6.82/9.70/10.75 | 1/1 | 503.0 / 524 | 18.1 |
| churn65536-p4096-1000000-t1 | PASS | 999,941 | 6.03/8.91/10.49 | 1/1 | 502.9 / 524 | 23.7 |
| churn65536-p4096-1000000-t2 | PASS | 999,943 | 6.03/9.18/10.75 | 1/1 | 503.0 / 524 | 23.8 |
| cancellation-visible | FAIL / warmup | 0 | 0.00/0.00/0.00 | 1/1 | 339.9 / 524 | 6.0 |
| stable-scene0-2000000-frozen | PASS | 1,999,743 | 11.80/15.47/15.99 | 1/1 | 346.4 / 524 | 19.9 |
| stable-scene0-2000000-indexed-t1 | PASS | 1,999,745 | 11.53/15.20/15.99 | 1/1 | 346.6 / 524 | 19.7 |
| stable-scene0-2000000-indexed-t2 | PASS | 1,999,759 | 11.53/15.20/15.99 | 1/1 | 346.5 / 524 | 20.0 |
| stable-scene0-3000000-frozen | PASS | 2,999,610 | 11.01/14.16/15.47 | 1/1 | 346.5 / 524 | 29.9 |
| stable-scene0-3000000-indexed-t1 | PASS | 2,999,585 | 10.75/14.16/15.47 | 1/1 | 346.5 / 524 | 29.8 |
| stable-scene0-3000000-indexed-t2 | PASS | 2,999,547 | 10.75/14.16/15.73 | 2/1 | 346.3 / 524 | 29.8 |
| stable-scene1-3000000-frozen | PASS | 2,999,587 | 10.75/14.16/15.47 | 2/1 | 343.5 / 524 | 29.9 |
| stable-scene1-3000000-indexed-t1 | PASS | 2,999,634 | 10.49/13.63/14.94 | 2/1 | 343.8 / 524 | 29.8 |
| stable-scene1-3000000-indexed-t2 | PASS | 2,999,647 | 10.49/13.63/14.94 | 1/1 | 343.8 / 524 | 30.1 |
| stable-scene2-2500000-frozen | PASS | 2,499,670 | 12.06/15.47/17.30 | 1/1 | 408.3 / 524 | 27.1 |
| stable-scene2-2500000-indexed-t1 | PASS | 2,499,669 | 11.80/15.47/17.30 | 1/1 | 409.2 / 524 | 26.7 |
| stable-scene2-2500000-indexed-t2 | PASS | 2,499,655 | 11.80/15.47/17.30 | 1/1 | 409.0 / 524 | 26.5 |
| churn65536-p16384-2000000-t1 | PASS | 1,999,706 | 13.11/17.30/17.83 | 1/1 | 506.5 / 524 | 19.8 |
| churn65536-p16384-2000000-t2 | PASS | 1,999,697 | 13.11/17.30/17.83 | 1/1 | 506.4 / 524 | 19.8 |
| post-fill-cancellation | FAIL / measured | 97,209 | 23.59/41.94/44.04 | 1/1 | 502.7 / 524 | 1.3 |

[Remote final sealing](../../evidence/eviction_remote_final.json) compares all 325 archived source
files byte for byte with the actual qualified tree, hashes every package file and binary, and
verifies model/binary preservation across all five campaigns. The trained U-Net engine remains
`2f20d4fd17c1e1eb3399ac450f84a2f1c7737ab69e2098ae383dc1911a1455cd`; the synthetic engine remains
`4dd03d4c2c6291e98145c64ec2fb7e174ba69d7b7e1728e84cf7357e51772f59`.
No source file mismatches are found. After all experiments, the GPU reports 0% utilization,
130MiB device use and no compute processes. No workloads were interrupted. The temporary
reauthentication condition was resolved before final experiments and receipt collection.

All historical geometry, map JSON, pinned-staging, point-only 1M, smaller-window queue and R4
receipts remain unchanged. The source/receipt freeze and scoped new-object history audit are
independent of new performance results. The inherited single-commit R0 history checker is not
applicable to this authorized multi-commit branch and is left unchanged. Target builds and
performance on Orin/SM87 or Thor/SM110, async file-encoding throughput, larger-model performance,
live sensor transport and a 3M full-capacity map are NOT_RUN. There are no remaining blockers to
the stated 100k initial objective or the qualified bounded 2M scene-A result.
