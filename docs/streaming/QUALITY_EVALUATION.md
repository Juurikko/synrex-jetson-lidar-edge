<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Tuomas Pietilä -->
# Q0/Q1 validation artifact evaluation

`tools/runtime_quality.py` evaluates a small, frozen validation subset. It runs no
inference, acquires no GPU lease, installs nothing, and does not qualify a trained
model. Its CPU fixtures are intentionally synthetic. Actual trained Q0/Q1 execution,
logit/discrete preprocessing parity, and final integrated qualification still need
source-bound runtime receipts.

The three questions remain separate:

| Comparison | Meaning | Reported decision |
|---|---|---|
| Q0 versus Q1 native | Whole-source context versus deployed streaming context | Quality deltas; requires acquisition-level uncertainty before a policy verdict |
| Q1 reference versus Q1 native | Identical frozen model, features, precision, and 16,384-point windows | Preregistered original-point label/quality gates only |
| Capacity and point age | Assembly, queue, processing, and coverage | Separate timing summaries; no capacity qualification |

Q0 uses each complete authoritative source with a per-source coordinate minimum.
Q1 uses independent, contiguous **16,384-point windows**, a separate minimum per
window, and an EOF partial window. Sources never share a window. A timeout flush,
overlap, stable grid, past context, or revised prediction is a new policy outside
this v1 schema. A final map cannot replace initial output in this scorer. Q0 is an
offline reference and is never labeled causal.

## Freeze inputs before inference

The API has two identities because prediction files do not exist before inference:

1. Prepare a spec with the exact fields below and relative artifact paths. Call
   `freeze_protocol(spec, base_dir)` before inference and independently retain that
   SHA256 in the campaign preregistration. It hashes actual model, preprocessing,
   per-stage precision, source-checkout identity artifact, source files, observations,
   truth, taxonomy, gates, source/acquisition grouping, and fixed policies. Prediction
   and window trace paths/content are excluded from this first identity.
2. Put that identity in `spec["protocol_sha256"]`. Run the separately admitted
   reference/native adapters against the frozen artifacts. Collect predictions and
   traces in manifest source order and original point-index order.
3. Call `seal_manifest(spec, base_dir)` after inference. This verifies the unchanged
   protocol inputs and binds every actual output file. Exclusively create a new
   manifest and retain its `manifest_sha256` independently.
4. Call `evaluate(manifest, base_dir, expected_manifest_sha256,
   expected_protocol_sha256)`. It rehashes all files and checks both independently
   retained identities. JSONL digests are checked again over the exact consumed
   record bytes. The CLI requires both identities explicitly.

Hashes detect changed artifacts; they do not prove preregistration chronology,
validation-split admission, actual executable behavior, or that a text file described
as a model really contains admitted weights. An independent execution receipt must
bind the real checkout/environment, model, engine, preprocessing, data extraction,
and execution to these same identities. Do not substitute an arbitrary caller policy
name or a self-issued trace for that evidence.

The fixed spec fields are:

```json
{
  "schema": "synrex.runtime_quality.v1",
  "purpose": "validation_development",
  "split": "validation",
  "clock": "shared_monotonic_ns",
  "policy": {
    "q0": "whole_source_minimum_v1",
    "q1": "independent_minimum_16384_eof_partial_v1"
  },
  "gates": {
    "minimum_label_agreement": 0.995,
    "maximum_mean_iou_loss": 0.01,
    "maximum_supported_class_iou_loss": 0.02,
    "supported_class_points": 1000
  },
  "taxonomy": {
    "classes": [{"id": 0, "name": "other"}, {"id": 7, "name": "ground"}],
    "ignored_reference_labels": [-100]
  },
  "sources": [
    {"artifact": "source_0", "points": 16385,
     "scene": "validation-scene-1", "acquisition": "validation-flight-1"}
  ],
  "artifacts": {},
  "protocol_sha256": "<independently retained input/protocol SHA256>"
}
```

Every artifact entry is `{"path": "relative/path", "sha256": "<actual SHA256>"}`
when sealed. Before sealing, `sha256` may be absent; an existing hash is never
silently refreshed. Absolute paths, `..`, or symlinks escaping the base directory
are rejected. Required artifact keys are `model`, `preprocessing`, `precision`,
`source_checkout`, `observations`, `truth`, each source's distinct `artifact` key,
and `{q0,q1_reference,q1_native}_{predictions,windows}`. There are no unused artifact
entries. `preprocessing` and `precision` must describe the admitted frozen model's
actual stages, including any admitted mixed precision, without changing them.

The source ID in every record is the SHA256 of the **actual bound source file**.
Source counts and the extraction from source bytes remain independently admitted
metadata: this tool does not decode LAS or inspect weights. The consumed published
test set and a new blind holdout are rejected as development `split` values. The
literal `validation` declaration also needs independent split-manifest verification;
this tool cannot discover a disguised test set from caller text.

## JSONL records and independent window checks

Observation fields are exact; extra fields, including labels, are rejected:

```json
{"source_sha256":"<source SHA256>","point_index":0,"available_ns":0,"captured_ns":null,"xyz":["240000.001","6700000.002","-0.01"]}
```

Indices must cover `0..points-1` exactly for each source, in manifest order.
Availability is nondecreasing within each source. All timestamps are unsigned
64-bit integers on one declared shared monotonic nanosecond clock. Capture is null
when unknown, otherwise no later than availability. Coordinates are finite decimal
strings, preserving source precision in the evaluator before any float conversion.
The adapter must independently establish how those values were extracted from LAS
integers/scales or adequate source doubles.

Truth records contain exactly `source_sha256`, `point_index`, and integer `label`.
Prediction records contain exactly those fields plus integer `window_index`.
Missing, duplicate, reordered, foreign-source, wrong-window, revised, extra, or
out-of-taxonomy predictions fail evaluation. Ignored truth labels remain mandatory
in output coverage and native/reference agreement.

Each window trace contains:

- `source_sha256`, zero-based `window_index`, and half-open original-index
  `begin`/`end` boundaries;
- `cutoff_ns`, equal to the last included observation's availability;
- `origin`, the canonical decimal-string minimum of the three input-coordinate
  axes, and `input_sha256`, the canonical hash of the exact observation chunk;
- `bindings`, with the actual `model_sha256`, `preprocessing_sha256`,
  `precision_sha256`, and `source_checkout_sha256`;
- `admitted_ns` and `output_ns`, satisfying cutoff ≤ admission ≤ output.

`build_reference_windows` reconstructs boundaries, coordinate minima, cutoffs, and
feature-input digests directly from observations; it does not call the native
window builder. Both Q1 implementations must match this same oracle. Coordinate
minima are a window-lineage diagnostic, not a replacement for the model's voxel
geometry, feature reduction, reconstruction, or floating numerical parity checks.

## Cutoffs and executable prefix probes

`cutoff_reference(observations, cutoff_ns)` constructs the independent offline
reference input using **all and only observations available by that cutoff**, with
equal timestamps included together. It provides the input digest and whole-available
reference windows. A future minimum cannot enter this reference. The evaluation
report records these per-source cutoff input digests for each Q1 output window.
Those are diagnostic **inputs**: inference and ground-truth scoring of the offline
cutoff reference are still required execution, not an implied completed result.

`prefix_causality_probe(execute, prefix, futures)` actually calls an adapter three
or more times: future removed, future appended, and future altered. `execute` takes
isolated feature-only observation records and returns committed prediction records
with `source_sha256`, `point_index`, `window_index`, `label`, `cutoff_ns`, and
`committed_ns`. All future inputs are validated before any callback. Every prefix
point must occur exactly once, in order, at its independently reconstructed Q1
window/cutoff. Only `committed_ns` is excluded from canonical comparison. A changed
label, window, cutoff, missing output, revision, or arbitrary added wall-clock field
fails the probe.

The probe accepts one source and a prefix ending at a complete Q1 window boundary.
It excludes EOF partial windows because appending after declared EOF is a different
session contract. Future availability must be strictly after the committed cutoff.
It records exact input/output hashes and calls the result
`SAMPLED_EXECUTION_PREFIX_PASS`. This establishes the sampled adapter's tested
prefix property; it is not universal proof, and a CPU fixture receipt is not GPU
execution evidence. Native process isolation, completion fences, and adapter/source
identity belong in the real job receipt.

## Metrics, timing, and precision scope

Quality uses `runtime_comparison.metric_summary`: rows truth, columns prediction;
zero denominator is null; absent-class union is null; all-class macro means count
absent classes as zero; no scored points yields null quality. The report includes
OA, all-class mIoU, present-class mIoU, macro-F1, each class's support/IoU/precision/
recall/F1, confusion matrices, and full original-point coverage. Raw Q1 reference/
native disagreement confusion includes ignored truth points. Rare classes are always
visible even when below the 1,000-point numerical guard.

Q1 native/reference agreement, ≤1 percentage-point all-class mIoU loss, and ≤2-point
supported-class IoU loss are reported as **label/quality gates**. With no scored
truth, an otherwise passing label gate is INCONCLUSIVE because mIoU is unavailable.
A PASS never implies
discrete executable preprocessing parity or logit qualification. A Q0/Q1 policy loss
never fails a numerical implementation gate. `per_acquisition` contains one metric
summary per run per acquisition for paired cluster uncertainty; point counts are
not independent statistical samples. The policy verdict remains unassessed pending
that uncertainty and the separately reviewed criteria.

Point-weighted nearest-rank p50/p99, mean, and max nanosecond summaries keep these
boundaries distinct:

| Quantity | Difference |
|---|---|
| Availability accumulation | window cutoff − point availability |
| Queue | admission − cutoff |
| Service | output − admission |
| Availability to output | output − point availability |
| Capture to output | output − capture, only for points with known capture |

Unknown capture time remains unavailable; report its separate support count. Clock
values are validated but a declaration does not prove clock synchronization. At
40,000 points/s, 16,384 observations occupy about 409.6 ms of stream time before
processing. In a synthetic schedule with the first sample at time zero and exact
25 μs sample spacing, the first-to-last sample difference is 409.575 ms. Both show
why low service latency cannot establish 250 ms total point age for Q1.

`local_lattice` is an independent exact rational LAS integer/scale and anchor oracle
for a **separately versioned** preserved-local-coordinate candidate. Tests translate
source offsets and anchor together by (+240,000 m, +6,700,000 m, 0), exercise negative
and exact boundaries, and retain precision beyond the default Decimal context.
This does not change or qualify a historical float32/rint preprocessing contract.
If the frozen baseline loses global-coordinate precision, characterize and retain
that limitation under its own admitted contract before comparing a new candidate.

## Bounded CLI and validation

The harness intentionally admits at most 50,000 source points and 64 sources, with
64 MiB of aggregate parsed JSONL bytes, 1 GiB of total artifact bytes, 256 MiB per
artifact, and at most one million aggregate confusion-matrix cells. This is a small
validation-subset evaluator, not a full 256-tile scoring engine. JSON records and
Hash work is admitted before reading: at most 76 artifact entries, regular files,
and the stated aggregate bytes per hashing pass. Metadata use the existing strict JSON parser (duplicate keys/nonfinite values
rejected). Prefix probes admit at most eight future variants and 50,000 points per
case. Full-source list storage and repeated cutoff construction are bounded by
these small-subset limits; larger-scale inference/scoring requires a separate
streaming implementation. These limits are implementation bounds, not GPU capacity
claims.

```bash
python tools/runtime_quality.py freeze --manifest spec.json --base artifacts --out protocol.json
# Retain protocol.json externally; set spec.protocol_sha256; run admitted inference.
python tools/runtime_quality.py seal --manifest spec.json --base artifacts --out manifest.json
python tools/runtime_quality.py evaluate --manifest manifest.json --base artifacts \
  --manifest-sha256 "$MANIFEST_ID" --protocol-sha256 "$PROTOCOL_ID" --out quality.json
```

All CLI outputs use exclusive creation. A failing matched Q1 label/quality gate
returns status 1 **after retaining its report**. Invalid inputs raise an error and
produce no report. See `tests/test_runtime_quality.py` for complete executable
fixture construction and adversarial lineage, leakage, window, clock, identity,
precision, and output-overwrite cases.
