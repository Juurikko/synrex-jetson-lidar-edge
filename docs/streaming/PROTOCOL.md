# Native point stream R0 contract

<!-- Copyright (c) 2026 Tuomas Pietilä. SPDX-License-Identifier: CC-BY-4.0. -->

Track B is an isolated extension of accepted R4 source
`3f1f3fabd3d92d69c02394936188502678a6b9ed`, tag `byom-unet-fractal-r4-rc2`, final audit
`c3451b6db942560829a360de3562624fcdc4282d`. R4 file inference and model package v2 are
compatibility baselines. No model, weight, accepted output or tag is rewritten.

Input is UTF-8 JSON Lines, one `synrex.point_stream.v1` chunk per line. See the executable
[fixture](../../examples/streaming/deterministic.jsonl) and C++ [API](../../native/include/synrex/stream.hpp).
All keys are mandatory; nullable attributes use explicit JSON null. Duplicate/unknown keys,
unknown versions, excessive nesting, nonfinite coordinates, negative/fractional integers,
unsupported clocks/units/taxonomies and oversized records are rejected before inference.
There is no sensor transport in R0's first slice.

| Field | Contract |
| --- | --- |
| schema | Exactly `synrex.point_stream.v1` |
| sensor_id, stream_id | Printable ASCII identifiers, 1–128 bytes; one ordered pair per assembler |
| sequence | Unsigned 64-bit chunk sequence, strictly increasing; gaps counted, affected partial frame discarded |
| frame_id | Unsigned 64-bit, increasing when a new frame starts |
| frame_start_ns, frame_end_ns | Inclusive UNIX nanosecond interval, at most 1 second by default |
| chunk_index, chunk_count | Zero based index, declared positive chunk count; must arrive contiguously |
| frame_points | Exact total source point count across the frame; incomplete frames never reach inference |
| units | Exactly `metres` |
| coordinate_frame, crs | Explicit identifier and CRS declaration; `LOCAL:*` permitted for invented coordinates; no reprojection, motion compensation or inferred EPSG code |
| clock | Exactly `unix_ns`; source time is preserved independently from the monotonic host latency clock |
| taxonomy | `synrex.fractal7.v1` or `synrex.synthetic3.v1`; strict model projection match required |
| points | Nonempty array of source points in source order |
| points[].id | Exact uint64 source identity, unique within frame; identity key is (sensor, stream, frame, id) |
| points[].timestamp_ns | Exact uint64 UNIX timestamp within the declared frame interval |
| points[].xyz | Three finite double coordinates in metres, absolute value at most 1e9 |
| points[].intensity | uint16 or null; null rejected by a model requiring mean intensity |
| points[].return_number, return_count | Both null, or integers 1–15 with number ≤ count; retained unchanged |

Consumers must use exact uint64 parsing. JavaScript Number cannot preserve IDs/timestamps
above 2^53; the checked fixture intentionally exercises larger IDs. Coincident points are
separate source events. Sequence is not substituted for identity. No implicit timestamp
interpolation, unit conversion, LAS remapping or absent-intensity imputation is performed.
Only complete frames are accepted as inference windows. The window's source minimum defines
R4's spatial origin; grid shape, spatial overlap, voxel reduction, first/last-wins policy and
block traversal come from the unchanged model contract. Temporal halo is zero frames. Spatial
context is exactly R4's model block membership; there are no additional streaming halos.
Comparisons with a whole-scene file using different window boundaries are invalid.

The seven channels, in order, are: other/unclassified, ground, vegetation, building, water,
bridge, IGN permanent aboveground structure. Authoritative IDs are 0–6. The frozen safe R4
projection is `[1,2,200,6,9,17,201]`, with user-defined LAS 200/201 explicitly declared in
[the taxonomy](../../examples/unet/semantic_classes.json). No aggregation into LAS5/64 occurs.
Synthetic three-channel fixtures bind only to `[2,5,6]`; they do not claim FRACTAL semantics.
All seven IDs are supported even when a particular scene does not predict all seven.

Default budgets: 1 MiB per line, 4,096 points per chunk, 65,536 per frame, 1,024 chunks per
frame, one assembling frame and a two-second assembly deadline. Timeout is evaluated on the
next accept or explicit `expire(monotonic_ns)`; a future live transport must call expire on
idle deadlines. EOF, gap, incompatible frame metadata or new-frame arrival discards a partial
frame and counts its received points. Late/repeated packets are counted and rejected; no
unbounded reorder history is retained. Missing sequence counts refer to absent chunks; their
unknown point counts are never fabricated. A malformed line aborts replay with FAIL rather
than skipping unknown source content.

Service input and output queues default to two frames each, with a hard cap of eight each.
`block` propagates bounded backpressure; `reject-new` counts every rejected frame and makes
complete-coverage replay FAIL. An active GPU window finishes before cancellation; queued
frames/results are discarded with explicit accounting. CPU fixture sessions in unit tests
are labeled MODEL_FREE_TEST_ONLY and are never production inference substitutes.

`all_points_freshly_classified` events retain every point field, complete frame boundaries,
first/last sequence, exact model semantic channel and separate `model_las_code`. Confidence
is null: argmax logits do not establish calibrated confidence. The bounded semantic map uses
world-coordinate floor(XYZ/resolution), rejects coordinate/taxonomy changes, retains the last
classification by source timestamp, and evicts least recently touched cells. Deltas include
removed keys on eviction. Class or observation-time changes emit deltas. Snapshot age is
explicitly as-of-source-time minus last observation time, not wall-clock inference latency.
Map deltas and evictions are outputs, never freshly classified source point counts. Source
classification mode and map delta mode remain separate. Reused-class and geometry-only
counts are zero for this implementation.

The event sink runs on its own bounded output queue. A slow file writer applies the selected
input backpressure policy after queues fill. It cannot silently drop events and report PASS.
Optional LAZ/COPC export is NOT_RUN/NOT_IMPLEMENTED for this slice; R4's verified
[QGIS export](../byom/QGIS_R4.md) remains available for existing file artifacts. No exporter
is placed inside the inference worker.
