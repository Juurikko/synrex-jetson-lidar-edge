<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Tuomas Pietilä -->

# Frozen validation representation diagnostics

`tools/runtime_representation.py` compares selected voxel resolutions on an
explicit validation subset. It consumes labelled original-point observations;
it performs no inference, model training, context tuning, or GPU scheduling.
It extends the existing voxel-majority analysis with per-class errors, mixed-label
collisions, sampled semantic boundaries, and optional annotated thin objects.

For a fixed source-specific partition, selecting its most frequent scored label
maximizes original-point **accuracy** among one-label-per-cell assignments. This
uses validation truth and is an oracle diagnostic, not a deployable classifier.
The resulting **mIoU is diagnostic and is not a mathematical upper bound**.
The tests include an exhaustive counterexample with strictly better mIoU and
lower accuracy under a different assignment.

## Freeze inputs before comparing resolutions

Create a manifest following the example below. Relative artifact paths are
resolved within `--base`; path traversal and symlink escapes are rejected.
Every raw source and parsed artifact is hashed. Existing hashes must match and
are never refreshed silently. Retain the sealed manifest identity outside the
mutable manifest file before examining results; sealing cannot attest when a
human actually preregistered the choice or verify split membership/rights.

```sh
python3 tools/runtime_representation.py seal \
  --spec representation.json --base admitted-subset --out frozen.json
python3 tools/runtime_representation.py analyze \
  --spec frozen.json --base admitted-subset --out representation-report.json
```

Both outputs use exclusive creation. Analysis verifies the sealed manifest and
all actual artifact hashes, then also hashes the parsed bytes as consumed. It
rejects changed metadata, resolutions, missing/extra/duplicate/reordered original
points, unknown labels, duplicate JSON keys, nonfinite metadata, and ambiguous
Boolean integer fields. Reports identify the scorer and reused metric helper by
actual file SHA256. Original raw-source hashes/counts are authoritative declarations;
this tool does **not** prove that an adapter extracted coordinates or truth correctly.

An unsealed example (`source.laz` must be the actual admitted source):

```json
{
  "schema": "synrex.runtime_representation.v1",
  "purpose": "representation_diagnostic",
  "split": "validation",
  "taxonomy": {
    "classes": [{"id": 2, "name": "ground"}, {"id": 5, "name": "vegetation"}],
    "ignored_reference_labels": [-100]
  },
  "sources": [{"artifact": "tile", "points": 2, "scene": "s1", "acquisition": "flight1"}],
  "artifacts": {
    "tile": {"path": "source.laz"},
    "observations": {"path": "observations.jsonl"}
  },
  "geometry": {
    "crs": "declared local metric frame derived from the source CRS",
    "units": "metres",
    "minimum_xyz": ["0", "0", "0"],
    "maximum_xyz": ["10", "10", "4"]
  },
  "resolutions": [
    {"name": "one-metre", "voxel_size_xyz": ["1", "1", "1"]},
    {"name": "half-metre", "voxel_size_xyz": ["0.5", "0.5", "0.5"]}
  ],
  "boundary_radius": "0.25",
  "thin_objects": null,
  "accounting": {"feature_channels": 6, "feature_bytes_per_scalar": 4}
}
```

Observations are JSONL records in manifest-source order, then contiguous original
point index starting at zero within each source. Replace `SOURCE_SHA256` with the
actual raw artifact hash. Include ignored points in coverage:

```json
{"source_sha256":"SOURCE_SHA256","point_index":0,"xyz":["0.1","0.1","0.1"],"label":2}
{"source_sha256":"SOURCE_SHA256","point_index":1,"xyz":["0.2","0.1","0.1"],"label":5}
```

Coordinates, lengths, and widths are plain decimal **strings**, at most 48
characters with magnitude at most 10^15; exponent notation is unsupported.
Exact rational arithmetic preserves decimal lattice boundaries, negative
coordinates, and jointly translated coordinates/anchors. Adapters should retain
original LAS integer/scale precision when producing these strings. Voxel indices
are `floor((xyz-minimum_xyz)/voxel_size_xyz)` in the declared half-open domain
`minimum_xyz <= xyz < maximum_xyz`; no clamp hides out-of-domain points.
Sources share the frozen physical extent but never share cells or neighborhoods.

## Collision and stratum definitions

Ignored labels affect occupied-cell and original-point coverage but never choose
a majority, create a mixed-label cell, or create a semantic boundary. Ties choose
the smallest numeric class ID, regardless of taxonomy order. Rows of each
confusion matrix are truth and columns are prediction in declared taxonomy order.
Absent unions and other zero denominators are null; absent classes contribute
zero to the all-class macro, while no scored points yields null.

`mixed_label_cell_rate` divides cells with at least two scored labels by occupied
cells with at least one scored label. `scored_point_collision_rate` divides scored
points in those cells by all scored points. Per-class rows give misclassified
points/rate, mixed-cell points/rate, and mixed cells containing that class.
Mixed-cell membership describes ambiguous representation even when a particular
point retains its correct majority label.

Boundary points have at least one differently labelled scored point within the
closed Euclidean `boundary_radius` in the same source. `interior` is its scored
complement. This is a sampled, truth-derived semantic adjacency diagnostic,
independent of the selected resolution; it is not a surveyed boundary, feature
thickness estimate, or object reconstruction. Ignored labels are not neighbors.
The same full-cell majority prediction is scored in every stratum, without
refitting a label inside the stratum.

Thin-object evaluation requires an additional `object_widths` JSONL artifact and
the following replacement for `thin_objects`:

```json
{
  "threshold": "0.2",
  "definition": "Minimum physical object width measured by the named annotation method",
  "provenance": "Exact annotation dataset/revision and method; synthetic or inferred status disclosed"
}
```

Width records have exactly `source_sha256`, `point_index`, `object_id`, and
`width`, in the same complete order as observations. `object_id` is a nonempty
string and `width` is a positive decimal string in metres. Repeated object IDs
within a source must have identical widths. Use null for **both** object ID and
width when annotation is unavailable. A known width at or below the threshold
selects `thin_object`; larger known widths select `other_object`; unavailable
widths select `unknown_object_width`. These three strata partition scored points
when annotations are supplied. Without the artifact, all three explicitly report
`NOT_SUPPORTED` with null metrics. No width is inferred from sparse voxel counts.
Annotation correctness and the chosen geometric meaning of width remain external.

## Resolution cost and execution limits

Each result records `ceil((maximum-minimum)/voxel_size)` dimensions, the full dense
cell count per source domain and across all source domains, occupied cell count,
and feature payload estimates under the frozen channel/scalar-width assumptions.
The ratio to the first selected resolution is reported. Halving all three sizes
costs eight times as many dense cells for a divisible fixed extent; edge rounding
can change that ratio. Occupancy may grow differently. No dense array is allocated
by this evaluator. Occupied feature payload includes ignored-only cells and
excludes indices, allocator overhead, activations, and model memory.

`context_points` stays equal to the frozen original-point subset size across
resolutions, with whole-source context and independent sources. These estimates
are not a native runtime capacity, RSS, memory, latency, or causal-context result.
Peak memory and unmeasured model/GPU/terrain claims remain unavailable explicitly.

Admission limits are 64 sources, 50,000 total original points, 16 distinct selected
resolutions, 256 scored classes, and one million aggregate confusion-matrix cells
over all resolutions/strata. Each artifact is at most 256 MiB, aggregate hashing
at most 1 GiB, parsed JSONL at most 64 MiB, and each record at most 64 KiB.
Boundary spatial hashing performs at most two million candidate-pair examinations;
dense neighborhoods can hit this limit even when labels agree. Exceeding a limit
rejects the analysis, never samples away points or publishes a partial report.
Use a scientifically declared smaller validation subset if admission fails.

The regression suite independently enumerates classifier assignments, uses set
intersection/union for mIoU, and checks the spatial-hash result against direct
all-pairs distances. It also covers numeric-label ties, exact negative/translated
boundaries, isolated sources, ignored-only data, width availability, identity
mutation, invalid records, and resource admission.
