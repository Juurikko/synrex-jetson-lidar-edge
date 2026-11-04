<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Tuomas Pietilä -->
# Opt-in causal context R1 — native CPU contract

**Experimental, NOT connected to the released TensorRT executor, not an AGX
qualification.** The released 16,384-point Q1 policy, source-v1 codec and
public streaming R2 snapshot remain unchanged.

## Why this exists

The separate FAST-GC paired benchmark (PR #7) found that the frozen U-Net's
ground F1 drops from approximately 0.911 (whole 50 m FRACTAL tile) to 0.702
(16,384 fresh-point windows). On a **conditional subset**, retaining 112k
earlier points recovered near-whole-tile F1 at approximately 4.5x CPU
processing work. This is evidence worth porting, but not proof of full
streaming performance. Those small unordered archival tiles cannot establish
physical scan time or steady-state on a continuous sensor.

`CausalContextAssembler` is the **first native, deterministic CPU building
block** for testing those policies. It uses source-arrival order, not future
tile minima or future record data. It deliberately does no neural inference.

## Policy and output

- `fresh_points` — points committed at each step, usually **16,384**.
- `history_points` — up to **114,688** prior points (112 Ki-points).
- `max_context_points` — explicit cap, e.g. **131,072**.
- `fixed_world_origin_m` and `world_grid_size_m` — fixed, caller-admitted
  projected metre-frame spatial reference **not recomputed from each window**.
  These metadata values are NOT YET consumed by the dense TensorRT tensorizer.
- `max_inflight_windows` — at most 1–8 submitted-but-uncommitted windows.
- `require_exact_source_rows` — default true: each input frame must carry
  source-v1 header and original-index rows with exact contiguous coverage,
  which prevents a frame identity from masquerading as original LAS proof.
  Explicit false is an identity-only stream and makes NO raw-LAS custody claim.
- `semantic_class_count` — validation bound for classifier label outputs.

The assembly vector is `[most_recent_history, new_points]`.
`fresh_begin` and `fresh_count` mark exactly which source records can be
published. `commit_new_only` accepts a label for every member of the
context vector, rejects bad classes, and emits **only** the new source IDs and
labels. The history is retained for feature context but must never be
re-published as newly classified measurements.

One native assembler processes one descriptor/session/source. A changed
stream, CRS, frame identity, sequence, source-row index, header or unexpected
EOF fails closed. History and in-flight windows are bounded. A new source
requires a fresh object; no rollback or silent fallback.

A prefix probe runs the same immutable first window with future observations
deleted and changed, and verifies its already-issued output and fixed anchor
remain identical. The tests also exercise a six-point history ring, multi-frame
window assembly, EOF partial window, output-only-new IDs, duplicate/missing
indices and permanent poison on mismatched metadata.

## What this does NOT qualify

This component is not wired to `generic::InferenceSession`, does not
voxelize with the fixed origin, does not build model tensors, does not run
FP16/TensorRT/SM87, and does not measure a physical AGX or live sensor.
Even `cutoff_frame_end_ns` is the **sender's declared** frame boundary,
not independently established capture/availability time. A source-v1
header/row match is not a substitute for independent LAZ byte-parity checks.

The next GPU development step must compare whole-context tensor memberships,
input features, original point IDs, raw model outputs and reconstructed
**fresh-only** labels against a pinned CPU oracle. The existing model was
trained with whole-tile, source-minimum voxel anchors; changing origin to
fixed-world is a **different preprocessing policy** that requires a separately
qualified model or measured numerical-scientific equivalence. Do not equate
the native context assembly PASS with trained-model quality parity.

## CI gates and deployment

CPU CTest `causal_context_new_only_cpu` executes in the existing Release
and ASan/UBSan configurations. A separate native AArch64 source-only
contract job can execute without GPU allocation. No installed executable or
release protocol is changed. The physical AGX and RTX programmes remain
untouched until their controllers grant an idle allocation.
