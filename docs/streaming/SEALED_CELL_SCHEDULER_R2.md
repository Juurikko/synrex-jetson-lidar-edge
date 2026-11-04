<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Tuomas Pietilä -->
# CPU sealed-cell scheduler R2 — bounded research prototype

This is an **opt-in, model-free CPU scheduler**, not a qualified LiDAR
classifier or a TensorRT/AGX implementation. It operates on source-ordered
points that MUST first pass source-v1 input provenance and time/coordinate
admission upstream. No new model features, count rescaling or water-mask
policy is implemented or approved.

## Scientific reference and exact limited scope

The independent Python reference is `sealed()` in
`Jetson-FAST-GC/tools/continuous_r1/continuous_policies.py`
at commit `0c916dbfcb4f080566be2e2c3fb23a0ba1432129`.
For each 32m XY cell, that policy waits until **G global stream points**
have elapsed since the last observation, then releases the cell's pending
labels using all *earlier* points of the same cell as context.
It never uses future measurements for predictions; the reference uses all
earlier cell points, without a storage cap.

The C++ `SealedCellScheduler` follows this causal release/index logic
**until the first resource eviction**:

- `idle_gap_points` defaults to 131,072 source-order points.
- `origin_xy_m` and `cell_width_m` define a stable 32m planar grid.
  Matching Python `floor(XY / 32)` requires choosing an origin aligned
  to the same global 32m lattice, not arbitrarily reanchoring each flight.
- Each cell retains its own previous observations, with a
  `fresh_begin/fresh_count` boundary for *new* labels only.
- Issued contexts are immutable `shared_ptr<const vector<SourcePoint>>`.
  A mismatched/counterfeit handle or metadata is rejected before commit.
- `release_point_index=last_cell_index+G+1` for inactivity-triggered
  releases. At actual source EOF, remaining cells are released explicitly;
  EOF timing can be earlier than the reference's hypothetical G-point
  timeout after EOF, but no future points enter their context.
- No classification is executed by the scheduler; caller-provided class
  arrays are accepted only for the exact privately issued context, and
  only newly admitted point identities are returned.

## Bounded admission and honest eviction

Unlike Python, native memory is explicitly bounded by
`max_active_cells`, `max_retained_points`,
`max_points_per_cell` and `max_outstanding_windows`.

A cell may be evicted **only after all its pending labels have been
issued and all its outputs acknowledged**. Until then, capacity pressure
returns `accepted=false` with no input-point consumption. A producer
MUST retain and retry the same point. If no eligible cell can be evicted,
the path stays blocked; increasing admitted capacity or aborting the
research run is required. There is no silent point drop, silent
unbounded growth or phantom classification output.

Eviction explicitly sets `history_may_be_truncated=true` and marks
every subsequent window `previous_history_may_be_lost=true`. This
conservatively rejects a full-history parity claim after an eviction,
even for a cell that is first observed afterwards. A full policy verdict
must independently test quality with a specified eviction policy.
Logical retained-point caps are not peak-RSS guarantees because C++
vector capacity and in-flight snapshot allocation add overhead.

A retryable call can issue idle windows even when its *new* input point
has not been accepted. The caller must deliver/commit those issued
windows, then retry the same point index. The caller may not equate
`accepted=false` with an empty list of work.

## Existing qualifications versus remaining work

Native tests cover a known two-generation XY-cell schedule, no future
inputs, sparse source IDs, EOF, delayed commit, counterfeit window
substitution, explicit evictions, hard capacity blocking, CRS/spatial
and index failures. Genuine AArch64 CPU compilation and x86
Release/ASan CI are separate gates.

**Not run/qualified**: 28-million-point production-like end-to-end
scheduler parity against the Python recorded release-index files;
native CNN tensors; GPU inference or point-count rescaling; water IoU;
physical sensor time; actual target power; bounded sustained service
memory under full backpressure. The original CPU benchmark indicates
a sealed-cell policy may be promising, but count gain 2 damages water
and is not a production setting.

The scientific decision remains: **RTX/Orin policy integration HOLD**.
Train a density-robust semantic model and compare qualified
quality/latency on independent continuous scans first.
