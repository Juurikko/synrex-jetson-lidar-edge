<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Tuomas Pietilä -->
# Original-point runtime and policy comparison

`tools/runtime_comparison.py` is a CPU scorer for exported original-point labels.
It performs no inference, training, GPU scheduling or dataset copying. Its tests
use synthetic CPU fixtures; they establish no trained-model or runtime result.
Production adapters and measured Q0/Q1 outputs must be supplied separately.

```sh
python tools/runtime_comparison.py --spec frozen-comparison.json \
  --truth truth.jsonl --reference reference.jsonl --candidate candidate.jsonl \
  --out comparison.json
```

The Python API is `compare(spec, truth_path, reference_path, candidate_path)`;
`load_spec(path)` loads the bounded, strict JSON specification. The returned
report and specification use schema `synrex.runtime_comparison.v1`.
The CLI exclusively creates the output. A failed label-agreement threshold
returns exit 1 **and retains its report**. Invalid inputs fail before a report
is created; existing output files are never replaced.

## Source identity and adapter boundary

All three JSONL files contain exactly these fields per line:

```json
{"source_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","point_index":0,"label":7}
```

`source_sha256` identifies the original source bytes in the frozen authoritative
manifest. `point_index` is its zero-based original-point index, not the window,
voxel, packet or arrival index. Sources follow **manifest order**, and every
source contains indices `0..points-1` in order. Adapters must reconstruct and
canonicalize output first. This scorer deliberately rejects reordered streams
rather than sorting or holding a point-sized lookup table in memory. A source
may declare zero points; the complete comparison must contain at least one.

Missing, extra, duplicate or mismatched identities and unknown labels fail.
Ignored reference labels still require valid prediction records. Every original
point contributes to coverage and reference/candidate label agreement. Only
ground-truth quality scoring excludes ignored labels. History/context points
must never be added to the original-point population.

The source manifest is supplied authority, not discovered from predictions.
The scorer does not reread original LAS/LAZ files to verify manifest claims or
derive truth labels. Adapters must verify those hashes/counts and truth lineage
upstream. The report preserves every declared source hash/count and hashes the
exact bytes of all three complete record streams, canonical specification and
scorer file. Metadata is a declaration, not proof of execution or causality.

The CLI reads at most 4 MiB for the specification and 64 KiB for each JSONL
record (including newline). JSON duplicate keys, nonfinite numbers, malformed
records and unknown record fields fail. Scoring memory is O(K² + sources), with
K at most 256, independent of original-point count. Class IDs are distinct
integers `0..255`; class names are distinct nonempty strings. Ignored reference
labels are distinct integers outside the class IDs. Metadata/record booleans
are not accepted as integer labels or counts.

## Specification

Top-level fields:

| Field | Meaning |
| --- | --- |
| `schema` | `synrex.runtime_comparison.v1` |
| `track` | `runtime_parity` or `model_policy_quality` |
| `taxonomy` | `classes: [{id, name}, ...]`, `ignored_reference_labels: [...]`, optional `ground_class_id` (or null) |
| `sources` | ordered `[{sha256, points}, ...]`; unique source hashes |
| `evaluation` | shared `split`, `protocol` strings, `latency_budget_ms` number or null |
| `reference`, `candidate` | run objects described below |
| `minimum_label_agreement` | required preregistered number in [0,1] for runtime parity; forbidden for policy/model quality |

Each run includes:

| Field | Required contents |
| --- | --- |
| `name` | nonempty run identifier |
| `execution_mode` | `REFERENCE`, `HYBRID` or `NATIVE` |
| `execution_scope` | `whole_tile`, `causal_streaming`, `offline_lookahead`, `file_export`, `replay_looped`, `replay_distinct`, `loopback`, `network_transport` or `physical_sensor` |
| `limitations` | list of nonempty strings (may be empty) |
| `unsupported_modes` | object mapping unavailable execution modes to reasons (may be empty); cannot contain executed mode |
| `provenance` | `source_sha256`, `config_sha256`, `model_sha256`, `engine_sha256` (may be null), `source_manifest_sha256`, `scoring_taxonomy_sha256`, `package_taxonomy_sha256` (may be null), `package_taxonomy_unavailable_reason`, `hardware`, `software` |
| `semantics` | unique nonempty `input_fields` list; nonempty `preprocessing`, `windows`, `reconstruction` objects; nonempty `precision`, `policy`, `architecture`, `input_provenance` strings |
| `measurements` | nonempty `timing_boundary`; all numeric availability fields below |

SHA256 strings use lowercase hex. `source_manifest_sha256` is
`canonical_hash(spec["sources"])`; `scoring_taxonomy_sha256` is
`canonical_hash(spec["taxonomy"])`, using the exported helper (sorted keys,
compact JSON, UTF-8, no nonfinite numbers). Hardware/software must identify the
actual path; a Python benchmark driver does not make production inference hybrid.
These two canonical hashes identify the source manifest and **scoring contract**,
not the exact source manifest artifact bytes or native package taxonomy artifact.
`package_taxonomy_sha256` separately declares the exact package taxonomy artifact
bytes, verified upstream by the adapter. When unavailable, it must be null with
a nonempty `package_taxonomy_unavailable_reason`; when present, that reason must
be null. No package identity is inferred from the canonical scoring contract.
Record policy versions, causality bounds, anchor method and EOF/timeout behavior
inside the semantics objects where applicable. The scorer preserves additional
provenance/semantics fields; every semantics field must match in runtime parity.

Measurement availability fields are `offered_points_per_second`,
`achieved_points_per_second`, `p50_latency_ms`, `p95_latency_ms`, `p99_latency_ms`,
`max_latency_ms`, `peak_memory_bytes` and `energy_joules`. Each is required and
is a nonnegative finite number or **null when unavailable**; bytes are integers.
Declare what the boundary includes (assembly/wait, preprocessing, inference,
reconstruction, serialization, transport, consumer work) and exclusion limits.
The scorer does not manufacture measurements or qualify latency budgets.

## Separate questions and scoring rules

`runtime_parity` requires identical model identity and complete declared
semantics, including precision, windows and preprocessing. It emits only the
preregistered **label agreement** PASS/FAIL subverdict. Its denominator includes
ignored truth points. This is not a complete deployment qualification.
Both runs must declare the same package taxonomy artifact hash (or both null).
Both null produces `package_taxonomy_binding: UNAVAILABLE`, with reasons retained
in each run's provenance. Matching declared hashes produces
`MATCHED_DECLARED_ARTIFACT_HASH`; package qualification remains `NOT_ASSESSED`.
Discrete preprocessing, logits, quality margins, causality and resource checks
require external evidence. Preserve disagreements and the frozen criteria.

`model_policy_quality` permits differing models, fields and policies under the
shared taxonomy, frozen source population and declared evaluation/latency
budget. It reports observed differences with `numerical_parity: null`, without
turning an intentional policy change into a numerical failure. Differing input
provenance must remain visible; external RGB/NIR inputs do not establish a
controlled architecture ablation against LiDAR-only inputs.
Package taxonomy artifacts may differ, while the declared scoring mapping and
source manifest remain shared.

Both tracks report ground-truth confusion matrices, per-class support,
predicted counts, IoU, precision, recall and F1, OA, all-class mIoU, present-union
mIoU and all-class macro-F1. Matrix rows are truth and columns prediction, in
declared class order. Zero denominators produce null. All-class macro means
assign zero to absent classes; if **no points are scored**, aggregate quality
metrics and differences are null. A class absent from truth but predicted has
zero IoU/F1 and is penalized. Labels are already mapped to this taxonomy;
there is no implicit FRACTAL/ASPRS mapping.

The report distinguishes successful measurement (`measurement_status:
COMPLETE`) from scientific qualification (`qualification_status: NOT_ASSESSED`).
Candidate-minus-reference deltas are fractions, not percentage points.
No scene/acquisition bootstrap, independent holdout, rare-object support or
quality noninferiority claim is implied. Those analyses consume separately
frozen scene/acquisition provenance and criteria.

[Paired acquisition uncertainty](runtime_uncertainty.md) consumes identity-bound
paired confusion summaries with acquisition assignments frozen before evaluation.
[Q0/Q1 artifact evaluation](streaming/QUALITY_EVALUATION.md) validates the declared
whole-source and unchanged 16,384-point window artifacts before interpreting
their scores. Neither tool establishes executable/GPU lineage by itself.

If `ground_class_id` is declared, ground precision/recall and false-ground
contamination are a **semantic-label proxy**, excluding ignored truth. They do
not measure elevation error. Survey availability is false; DTM bias/RMSE,
quantiles and ditch/breakline retention remain null with `BLOCKED_INPUT`.
CRS, vertical datum/geoid, correspondence, survey reference, interpolation and
coverage masks are not invented. A later terrain evaluator must require those
inputs before making a terrain or survey-grade claim.
