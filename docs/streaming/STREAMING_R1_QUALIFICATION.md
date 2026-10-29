# Streaming R1 qualification

<!-- Copyright (c) 2026 Tuomas Pietilä. SPDX-License-Identifier: CC-BY-4.0. -->

This continuation preserves accepted `fe21303` and tag `streaming-r0-eviction-2m-20261027`.
R4 source/release tags, frozen models, predictions and historical PASS/FAIL evidence remain
unchanged. New original source is MIT scoped; model/data/SDK notices retain their separate
terms. Track B is unpublished; no push, release or AGX/Thor access occurred.

## Bounded outcome

The frozen trained FRACTAL U-Net runs through a native external producer, TCP receiver,
persistent TensorRT service, bounded semantic map and independent TCP subscriber. Every
counted point receives a newly executed window's neural prediction. The map provides no
inference labels. These are deterministic software replays of genuine licensed IGN/FRACTAL
geometry/intensity, with emulated measurement times; they do not qualify a physical LiDAR,
SLAM pose source or arbitrary larger model.

Two 60-second repetitions pass at nominal **2M freshly classified points/s on scene A**.
Dense scene C and full spatial churn at both 4,096 and 65,536 cells pass two repetitions at
**1.5M/s**. Their 2M trials FAIL the bounded output-delivery queue gate. No source, latency,
queue or memory limit was relaxed. Accepted in-process R0 rates remain separately preserved:
about 3M/s on A/B, 2.5M/s on C and 2M/s at full 65,536-cell churn.

[Every transport trial](../../archive/evidence/streaming_r1_transport_trials.csv) records rate, coverage,
loss, latency, queues, memory, GPU utilization/power and hashes. The
[summary](../../archive/evidence/streaming_r1_transport_summary.json) requires at least two complete
sustained PASS repetitions and no matching failure for a qualified binary/configuration.
The original nine-trial transport campaign has five PASS and four FAIL results; the optimized
21-trial campaign has 18 PASS and three FAIL results. Short model probes are separate.

| Software transport workload, 16,384-point windows | Accepted nominal rate | Delivered fresh points and map observation deltas/s, repetitions | Validated subscriber p99 ms | Receiver peak MiB | GPU peak MiB |
| --- | ---: | --- | --- | --- | ---: |
| A, stable, 65,536 capacity | 2M | 1,999,601 / 1,999,583 | 30.93 / 31.46 | 343.5 / 344.7 | 524 |
| C, stable, 65,536 capacity | 1.5M | 1,499,722 / 1,499,725 | 34.60 / 34.60 | 382.3 / 378.5 | 524 |
| A, full churn, 4,096 capacity | 1.5M | 1,499,644 / 1,499,655 | 33.55 / 33.55 | 340.9 / 340.8 | 524 |
| A, full churn, 65,536 capacity | 1.5M | 1,499,701 / 1,499,707 | 33.03 / 33.55 | 456.9 / 456.9 | 524 |

All accepted rows have 100% independent source coverage, zero classification drops, zero failed
point deliveries, exact ordered source identities/timestamps/attributes, zero duplicated frames
and zero reused semantic labels. All three application queues have two slots and drain to zero.
Map updates here count accepted observations/deltas, not new unique voxels. The first accepted
1.5M full-65,536 trial has 90,013,696 measured observations: 16,211,279 insertions and evictions
(about 270k/s), and 73,802,417 existing-cell updates. Map occupancy stays at 65,536 and resident
victim scans remain zero. Snapshot age and confidence-null semantics are unchanged; no TTL
expiration policy is introduced.

## Geometry and exact comparison

The optional stable radix member ordering preserves block assignment/origins, voxel indices,
source order within voxels, floating-point feature reductions and reconstruction. Small blocks
retain comparison sorting; nonzero overlap retains the original reference. CPU tests cover
205 enumerated cases including dense/coincident points, sparse empty regions, LAS precision,
negative/large offsets and adjacent block boundaries. Both frozen models pass native GPU
reference comparisons; prepared tensors are byte identical and traced logit error is zero.

Sixteen sustained matched trials compare the frozen accepted binary, current direct mode and
two radix repetitions. Point/delta digests, seven-class histograms and complete snapshot hashes
match in all four configurations. [The digest receipt](../../archive/evidence/streaming_r1_geometry_digest_parity.json)
and [per-trial CSV](../../archive/evidence/streaming_r1_geometry_trials.csv) preserve the exact identities.

| Identical in-process workload | Direct geometry active s | Radix geometry active s, mean | Direct worker active s | Radix worker active s, mean |
| --- | ---: | ---: | ---: | ---: |
| A, 2M full churn | 9.627 | 4.493 | 34.672 | 30.367 |
| A, 3M stable | 13.038 | 6.575 | 54.281 | 45.972 |
| B, 3M stable | 12.712 | 6.554 | 53.653 | 46.417 |
| C, 2.5M stable | 12.279 | 5.656 | 46.661 | 39.854 |

At 2M full churn member ordering falls from 5.361s to about 0.384s. This is an active CPU
region improvement, not a throughput multiplier. All accepted R0 rates were re-established;
no higher general-purpose rate is inferred.

The first unqualified wire prototype failed subscriber delivery despite freshly classifying
all 524,288 points; that receipt remains in
[the initial probe](../../archive/evidence/streaming_r1_initial_transport.json). Wire v2 adds per-event
model identity. Subsequent optimization comparisons use identical v2 inputs/models/configuration.
[Short real-model parity](../../archive/evidence/streaming_r1_transport_smoke_parity.json) and
[sustained parity](../../archive/evidence/streaming_r1_sustained_digest_parity.json) require exact canonical
SHA256 input/output equality. Host replay/arrival/publication clocks are excluded; measurement
clock, source records, model identity, predictions and every ordered delta remain included.
Independent subscribers verify every final full snapshot, and optimized repeated snapshots
also have identical SHA256 values.

## Host/device and consumer bottlenecks

The initial v2 consumer repeated validated-record checks and maintained a redundant occupancy
tree alongside its independent hash/list oracle. R1 keeps the oracle's complete delta/eviction/
class/time checks and final snapshot comparison while removing that second tree. Decoder-owned
sequence methods fully validate each wire record once. Direct typed API input still validates.
Diagnostic strings are constructed only when a check fails. A switchable reference consumer
retains duplicate sequence validation and the occupancy tree.

The sealed CPU allocation probe decodes identical two-million-point invented fixtures using
baseline `7b32215` and optimized `f39fa29` libraries: ordinary C++ allocation requests fall from
52,458,496 to 6,297,344, and active decode/destroy time from 0.578s to 0.173s on the CPU host.
This is a codec diagnostic, not neural or sustainable throughput. Exact fixture bytes and
[probe/library/compiler identities](../../archive/evidence/streaming_r1_wire_allocation_provenance.json)
are preserved. Earlier unsealed working-tree pilot timings are retained as diagnostics, with
that identity limitation explicitly recorded.

For the first 2M stable A transport trial, measured inference-worker regions include geometry
3.974s, H2D/zeroing 6.463s, CUDA preparation 0.956s, TensorRT 8.833s, point scatter 0.378s and
D2H 1.854s, within 29.044s worker wall service. CPU source conversion is 1.306s outside that
window region; about 6.59s remains host adapter/issue/synchronization residual. The sink spends
16.726s in map processing and 24.252s encoding binary events over all warmup/measured inputs.
The independent subscriber spends 15.570s decoding/sequence checking, 11.593s on expected-source
validation, 4.832s in the map oracle and 7.276s hashing/export observation. These producer,
worker, sink and subscriber regions overlap and cannot be added as end-to-end latency.

Dense C at 1.5M has 42.83s in those four subscriber regions on 97.52 million total points,
an active-only upper bound around 2.28M points/s before socket checksum/read, latency bookkeeping
and scheduling costs. Its 2M attempt fails during warmup in output delivery, with neural input
queue peak one. The full-churn 2M attempts similarly fail delivery: 4,096 cells during measurement,
65,536 cells during warmup. These are complete-pipeline FAILs, not new neural-capacity limits or
sustainable partial rates. The consumer's decode/source verification and the sink's event
encoding dominate the remaining transport cost.

The existing persistent CUDA events and host issue timeline are retained. H2D/zeroing event
regions include issue gaps and are not pure transfer bandwidth. GPU synchronization currently
protects host upload-vector lifetimes. Pinned staging stays disabled after its historical
negative result. No GPU voxelization/map rewrite, weight change or preprocessing relaxation
is justified by this profile.

**Next optimization:** profile/reuse bounded source-ID and binary-event scratch storage, and
remove remaining success-path diagnostic allocations from shared classification validation
under exact native CPU/GPU comparisons. Investigate immutable payload forwarding before a
larger CUDA rewrite. Keep complete independent consumer verification in the qualification path.

## Runtime, coordinates and outputs

The [wire specification](STREAMING_R1_PROTOCOL.md), [architecture/commands](STREAMING_R1_RUNTIME.md)
and [independent target continuation](STREAMING_R1_PORTING.md) describe the native APIs and build.
Input is bounded complete-frame binary TCP on loopback. Partial reads, truncation, checksums,
unknown versions, duplicate/missing sequences, timestamp discontinuities, reconnects, bounded
queues, transforms and missing poses have CPU/native tests. Raw measurement UNIX time remains
separate from replay scheduling, receiver first-envelope-read time and output publication.
No geographic projection, SLAM integration or physical sensor timing is inferred.

Source-bound optional LAZ exports use a separate subscriber worker and the unchanged R4
preservation writer. First fresh observations cover every original source record, with per-record
frame/time provenance and source/model/taxonomy hashes. Partial/changed-coordinate products and
replacement attempts reject. Complete synchronized bundles publish atomically. Original source
LAS CRS VLRs remain unchanged; stream frame/CRS declarations are separately retained as provenance.
Generic arbitrary-stream LAS capture without an original bound source file is outside this slice.
The optional offline COPC consumer reuses unchanged R4 preservation/semantic/source-index/QGIS
checks and publishes its own complete bundle atomically. File encoding stays outside inference.

## Long run, fault and export closure

[The ten-minute full-churn trial](../../archive/evidence/streaming_r1_long_stability.json) passes at
999,981 freshly classified points and map observation updates/s over 600.026 measured seconds.
All 600,014,848 measured source points are independently validated, with zero classification or
delivery losses and zero reused labels. Source-to-validated-output p50/p95/p99 are
30.41/37.75/39.85 ms; maximum is 45.31 ms. The map remains at 65,536 cells with 108,060,250
measured insertions/evictions and 491,954,598 existing-cell updates. Receiver peak RSS is
456.97 MiB and measured RSS variation only 36 KiB. Sampled device memory peaks at 524 MiB;
whole-trial mean GPU utilization/power are 10.80% / 9.33 W. Queues retain their two-frame limits.

[Fault receipts](../../archive/evidence/streaming_r1_transport_faults.json) preserve native FAIL for
duplicate/missing sequences, sensor-clock discontinuity, truncated frame payload and subscriber
stall. Clean reconnect passes. During subscriber stall all 3,014,656 points still receive fresh
model classifications, while all failed point deliveries are counted and the overall trial
remains FAIL. [Requested cancellation](../../archive/evidence/streaming_r1_transport_cancellation.json)
fires during measurement after map fill, retains a coherent validated prefix, explicitly drops
16,384 queued points and returns zero owned CUDA/pinned bytes at exit. It is a fault test, not
a sustainable-rate result. CPU tests also cover checksum/version/budget rejection, one-byte
fragmentation, real loopback reconnect, optional attributes, timed rigid rotation/translation,
missing poses, separate sensor identities and over a million bounded sequence observations.

[Reference/decoded native consumer A/B](../../archive/evidence/streaming_r1_consumer_ab.json) both pass
60 measured seconds at 1M, with exact input/output and complete snapshot SHA256 equality. The
[final GPU regression](../../archive/evidence/streaming_r1_final_gpu_regressions.json) passes both frozen
models through the same binary without rebuilding: exact source classes/IDs and prepared tensor
bytes, zero traced logit error, native execution with Python absent from PATH, packet-loss and
out-of-order failure expectations. [Five physical R4 preservation tests](../../archive/evidence/streaming_r1_r4_gpu_preservation.json)
also pass; frozen predictions/models/releases are never replaced.

[Async export under load](../../archive/evidence/streaming_r1_async_export_trial.json) passes 60 seconds
on dense scene C at 999,786 fresh points/s, p99 36.70 ms and complete source coverage. A separate
worker publishes all 400,654 original source records, collecting first fresh observations after
warmup with exact record/frame/measurement-time provenance. Receiver/subscriber peak RSS are
382.02 / 312.43 MiB; sampled combined RSS peaks at 816.88 MiB. No inference-side file writer
is required. [Independent file qualification](../../archive/evidence/streaming_r1_file_export_qualification.json)
verifies all preserved records, semantic/model/source/taxonomy bindings and exact reproduction
of the classified LAZ digest. Standalone preservation encoding takes 0.362s; complete optional
offline COPC/validation/project creation takes 5.175s. These are file-job wall times, not
sustained export rates or fresh neural classification throughput. QGIS 3.44.14 saves/reopens
the real COPC layer, with EPSG:2154 retained from the source and no viewing CRS assignment.

The [initial export startup FAIL](../../archive/evidence/streaming_r1_export_startup_failure.json)
was a controller-relative taxonomy-path error, corrected with an explicit path. The
[Windows-mounted filesystem FAIL](../../archive/evidence/streaming_r1_copc_drvfs_failure.json) rejects
unsupported atomic no-replace publication; no complete bundle is published there. Native Linux
storage passes synchronized no-replace publication and an existing-bundle replacement rejection.
Partial products remain invalid. Native LAZ source/model/coordinate semantics were unchanged.

[Model-only TensorRT measurement](../../archive/evidence/streaming_r1_model_only.json) executes 185,536
real enqueue calls in 30.005 seconds: 6,184 prepared-block inferences/s, with 32 enqueues per
synchronization. Batch-average event latency p50/p95/p99 is 0.135/0.238/0.242 ms. It emits **zero
classified source points**. Its 18.23M point-equivalent upper bound uses only the prepared block's
2,948 represented points and excludes geometry, transfers, scatter, maps, validation and queues;
it cannot replace the full-stream rates. Cold model initialization in this probe is 1.052s;
ordinary receivers initialize around 0.6–0.9s once, then reuse their context for successive
millisecond-scale windows. Actual clocks remain dynamic and unmodified.

## Identities, boundaries and final gates

Native implementation commits are `e955b07`, `d4f7b87`, `7b32215` and **`f39fa29`**. Later
commits add independently checked evidence, offline export durability and qualification tooling;
all final native production files remain byte identical to `f39fa29`. The
[remote seal](../../archive/evidence/streaming_r1_remote_seal.json) verifies all archived files for three
qualified variants, exact executable hashes, both complete frozen model packages and all three
licensed source hashes. Final GPU state is idle: 0% utilization, 130 MiB and no compute jobs.
The trained engine remains `2f20d4fd17c1e1eb3399ac450f84a2f1c7737ab69e2098ae383dc1911a1455cd`;
its manifest remains `2d1206be62385669477a180d98c1b330c1caf40d516625f64682284012f7a325`.
The synthetic engine remains `4dd03d4c2c6291e98145c64ec2fb7e174ba69d7b7e1728e84cf7357e51772f59`.
Native CUDA, TensorRT execution, package contracts, original geometry and preservation-writer
arithmetic remain unchanged from accepted R0. RTX hardware is SM89, driver 610.88, CUDA 13.2.86 /
runtime 13020, TensorRT 10.16.1, AMD Ryzen 7 7700X under WSL2. Model/data terms remain separate
from independently licensable MIT source.

[Final gates](../../archive/evidence/streaming_r1_final_gates.json) retain exact PASS/FAIL/NOT_RUN statuses.
Protocol, live loopback, accepted fresh coverage, map output, source-bound async export, geometry,
GPU numerics, sustained RTX and ten-minute stability pass. Orin/SM87 and Thor/SM110 physical
readiness and public release readiness remain NOT_RUN. No public push or release is authorized.

Remaining limits are explicit: one ordered sensor/map identity and one subscriber per receiver;
clean input reconnect rather than late-subscriber resume; terminal network snapshots rather than
periodic snapshot requests; Unix-nanosecond measurement input and explicit local/metric coordinates;
no vendor UDP decoder, clock synchronization, SLAM interpolation or geographic reprojection.
Incoming-rate columns are replay pacing targets with exact received/emitted counts and bounded
scheduling lag; no physical sensor arrival-rate qualification is inferred. Source-bound LAZ
export requires an original file. Larger full-map snapshots need a separately bounded chunking
extension before 262,144-cell churn qualification. Desktop rendering, native ARM64 target builds
and larger models remain unqualified. All failures and frozen historical evidence are preserved.
