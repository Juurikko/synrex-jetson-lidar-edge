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
- `fixed_world_origin_xy_m` and `world_grid_size_m` — **only X/Y** are
  anchored to an explicitly admitted planar grid. **No fixed Z origin** is
  supplied: Claude's exploratory Z-anchoring results do not yet qualify a
  new vertical normalization. The origin metadata are NOT YET consumed by
  the dense TensorRT tensorizer.
- `projected_crs` — explicitly required and checked against an intentionally
  narrow allowlist of known projected **metre** EPSG identifiers (including
  Finnish EPSG:3067, French EPSG:2154 and selected UTM families). A
  geographic frame such as EPSG:4326 or a foot-based frame such as
  EPSG:2263 is rejected before accumulating context. This is declared-CRS
  admission, not independent LAS WKT/VLR verification.
- `max_horizontal_offset_m` and `max_abs_elevation_m` — finite policy
  bounds checked for every incoming point before any window emission,
  preventing an accidental 100,000-km offset from reaching the voxelizer.
- `max_inflight_windows` — at most 1–8 submitted-but-uncommitted windows.
- `require_exact_source_rows` — default true: each input frame must carry
  source-v1 header and original-index rows with exact contiguous coverage,
  which prevents a frame identity from masquerading as original LAS proof.
  Explicit false is an identity-only stream and makes NO raw-LAS custody claim.
- `semantic_class_count` — validation bound for classifier label outputs.

The assembly vector is `[most_recent_history, new_points]`.
`fresh_begin` and `fresh_count` mark exactly which source records can be
published. Each issued window has a private, domain-separated SHA-256
binding across **all source IDs, timestamps, XYZ bits, intensity/return
attributes, history and fresh records, origin and other window metadata**.
`commit_new_only` recomputes that digest before publishing any labels,
rejects changed history *or* fresh data, and emits **only** the new source
IDs and labels. The history is retained for feature context but must never
be re-published as newly classified measurements. This validates a supplied
window against the CPU assembler's issuance, not whether a GPU truly used
that particular window for inference.

One native assembler processes one descriptor/session/source. A changed
stream, CRS, frame identity, sequence, source-row index, header or unexpected
EOF fails closed. History and in-flight windows are bounded. A new source
requires a fresh object; no rollback or silent fallback.

**Capacity is different from invalid data:** `try_ingest` returns
`accepted=false` when it cannot admit the *entire* next frame without
exceeding the outstanding-window budget, with no source state consumed.
The convenience `ingest` throws a retryable capacity error but does not
poison the session. After prior windows are committed, upstream must retry
the **same complete frame**. `try_source_eof` similarly postpones a
partial final window until capacity is available. The caller must use
backpressure/pausing; silently dropping a denied frame violates the contract.
This component does not yet install network flow control or solve end-to-end
GPU saturation. Oversized ingress frames may require upstream splitting.

A prefix probe runs the same immutable first window with future observations
deleted and changed, and verifies its already-issued output and fixed XY anchor
remain identical. Adversarial tests mutate history IDs, timestamps and XYZ,
fresh-point XYZ/intensity/returns, cutoff and history-state flags before commit;
all must be rejected. The tests also exercise a 65,536-point retried frame
while inference is outstanding, delayed EOF, geographic/foot CRS rejection,
extreme offsets, the bounded history ring, multi-frame assembly and source-ID
continuity.

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

## Scientific decision boundary after Claude's independent review

The same 112k-history policy does **not** match whole-tile over an entire stream:
validation all-points ground F1 was reported at 0.842 against 0.920 whole-tile;
validation after warm-up at 0.885 vs 0.897 over only 45 eligible tiles.
The test-set near-equality (0.895 vs 0.894) was conditional on 72 eligible
post-warm-up tiles, not the acquisition start. The original CPU reference
selection favored a 128k fresh window which exceeds most 50m tiles and is
not a deployment decision. Before native TensorRT integration, independently
compare 64k fresh-only with 16k+history on **long, time-ordered** acquisitions,
using the same pretrained model and no train/test leakage. Use validation
for policy selection; keep the consumed FRACTAL test set sealed.
