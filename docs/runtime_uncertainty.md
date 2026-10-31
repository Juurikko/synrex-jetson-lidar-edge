<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Tuomas Pietilä -->
# Paired acquisition uncertainty

`tools/runtime_uncertainty.py` computes paired acquisition-cluster bootstrap
intervals for original-point comparison summaries. It performs no inference or
quality qualification. Its CPU fixtures are synthetic, and do not establish
Q0/Q1 model quality or satisfy the required GPU gates.

```sh
python tools/runtime_uncertainty.py --spec frozen-analysis.json --out intervals.json
```

The Python API is `analyze(spec)`. The CLI reads strict JSON through the comparison
scorer's 4 MiB limit and exclusively creates its output. Existing files are never
replaced. `INCONCLUSIVE` analyses retain their report and return CLI exit 0;
this means the analysis ran, not that the candidate passed a scientific criterion.

The specification uses `schema: synrex.runtime_uncertainty.v1`, a `classes` list
of distinct `{id, name}` entries, a `scenes` list and a frozen `plan`. Class order
must match confusion rows and columns. IDs can include 255; K is bounded at 256.
Each scene contains exactly:

| Field | Meaning |
| --- | --- |
| `scene_id` | Unique scene or region identity |
| `cluster_id` | Acquisition/statistical unit shared by dependent scenes |
| `source_sha256` | Nonempty list of authoritative source byte hashes; no source reuse between scenes |
| `evidence_sha256` | Exact upstream comparison/evaluation receipt hash |
| `reference_confusion`, `candidate_confusion` | Nonnegative integer matrices, truth rows and prediction columns |

Each reference/candidate pair must have identical truth support in every row.
Confusions contain scored points after the upstream evaluator applies ignored-label
rules. All original points, including ignored ones, must still pass upstream
coverage checks. This module verifies pairing, shapes and internal arithmetic;
it does **not** verify the hash-named receipt's contents, truth extraction, source
counts or executable/model lineage. Those remain the upstream evaluator's job.
Keep these upstream artifacts alongside the report. No object-level support or
physical temporal correspondence is inferred from point counts.

The plan contains exactly `replicates` (100–100,000), `seed` (unsigned 64-bit integer),
`confidence` (strictly between 0 and 1), `minimum_clusters` (at least two),
`population_sha256` and `taxonomy_sha256`. Freeze the plan and source/cluster
assignments before evaluating candidates. `population_sha256` is
`runtime_comparison.canonical_hash(runtime_uncertainty.population(scenes))` and
binds scene IDs, cluster IDs and source hashes in supplied order. It excludes
future output hashes so a plan can exist before execution. `taxonomy_sha256`
is `canonical_hash(classes)`. The report separately preserves upstream evidence
hashes, the complete input hash, plan hash and both scorer file hashes. No timestamp
or trusted preregistration authority is manufactured by this module.

Each bootstrap replicate samples as many acquisition clusters as observed, with
replacement and equal cluster selection probability. The **same draws** select
both runs; every scene within a selected cluster stays together. Confusions are
pooled before metric calculation. This estimates differences of pooled point
metrics; it is not an unweighted mean of scene mIoUs. Splitting a scene inside an
existing acquisition does not create additional independent observations.

Outputs contain OA, all-class mIoU, all-class macro-F1 and per-class IoU differences
(candidate minus reference, in fractions), two-sided empirical percentile intervals
using nearest-rank inverse empirical CDF, and rare-class point/scene/cluster support.
All-class macro metrics assign zero to absent classes, matching the comparison
scorer. Undefined per-class IoU or all-ignored bootstrap draws are counted; if **any**
draw is undefined, that metric's interval is null. Draws are never silently discarded.
Fewer than two clusters produces no interval. Fewer than the preregistered
`minimum_clusters` keeps metrics `INCONCLUSIVE`, even when an interval is available.
That threshold is a reporting rule, not evidence of statistical adequacy.

Intervals are marginal and assume exchangeable, appropriately chosen independent
clusters. They are not simultaneous per-class intervals and do not decide the
provisional 1 pp/2 pp noninferiority gates. No point count can replace missing
acquisition support. `qualification_status` remains `NOT_ASSESSED`.

CPU/resource limits reject more than 2,000,000 input matrix entries, 1,000,000
retained bootstrap metric values or a conservative 50,000,000 matrix-work estimate.
Counts and possible pooled bootstrap totals must fit the exact integer range
through 2^53. These are implementation admission bounds, separate from the frozen
quality, latency or GPU budgets. Increase neither a scientific margin nor its
population after inspecting results.
