# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
"""Frozen validation-only voxel representation diagnostics; no inference or GPU use."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import re
from fractions import Fraction
from pathlib import Path

_module = importlib.util.spec_from_file_location(
    "_representation_comparison", Path(__file__).with_name("runtime_comparison.py")
)
comparison = importlib.util.module_from_spec(_module)
_module.loader.exec_module(comparison)
require = comparison.require
integer = comparison.integer
canonical_hash = comparison.canonical_hash
SCHEMA = "synrex.runtime_representation.v1"
MAX_POINTS = 50_000
MAX_SOURCES = 64
MAX_RESOLUTIONS = 16
MAX_ARTIFACT_BYTES = 256 * 1024 * 1024
MAX_TOTAL_BYTES = 1024 * 1024 * 1024
MAX_PARSED_BYTES = 64 * 1024 * 1024
MAX_MATRIX_CELLS = 1_000_000
MAX_BOUNDARY_PAIRS = 2_000_000
STRATA = ("boundary", "interior", "thin_object", "other_object", "unknown_object_width")


def decimal(value):
    """Exact finite decimal strings; bound integer arithmetic before conversion."""
    require(
        isinstance(value, str)
        and len(value) <= 48
        and re.fullmatch(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?", value) is not None,
        "bounded plain decimal string required",
    )
    result = Fraction(value)
    require(abs(result) <= 10**15, "coordinate/length magnitude limit exceeded")
    return result


def vector(value, positive=False):
    require(isinstance(value, list) and len(value) == 3, "xyz triple required")
    result = tuple(decimal(item) for item in value)
    require(not positive or all(item > 0 for item in result), "positive xyz required")
    return result


def clone(value):
    return comparison.strict_json(json.dumps(value, allow_nan=False))


def artifact_path(base, entry):
    require(isinstance(entry, dict) and set(entry) <= {"path", "sha256"}, "invalid artifact")
    require(comparison.nonempty(entry.get("path")), "artifact path required")
    relative = Path(entry["path"])
    require(not relative.is_absolute() and ".." not in relative.parts, "relative artifact required")
    base = Path(base).resolve()
    path = base / relative
    require(path.resolve().is_relative_to(base), "artifact escapes base directory")
    require(path.is_file(), "regular artifact file required")
    return path


def _validate(spec, sealed=False):
    required = {
        "schema",
        "purpose",
        "split",
        "taxonomy",
        "sources",
        "artifacts",
        "geometry",
        "resolutions",
        "boundary_radius",
        "thin_objects",
        "accounting",
    }
    require(
        isinstance(spec, dict)
        and set(spec) == required | ({"manifest_sha256"} if sealed else set()),
        "invalid manifest fields",
    )
    require(spec["schema"] == SCHEMA, "unsupported schema")
    require(
        spec["purpose"] == "representation_diagnostic" and spec["split"] == "validation",
        "only validation representation diagnostics supported",
    )
    taxonomy = spec["taxonomy"]
    require(
        isinstance(taxonomy, dict) and set(taxonomy) == {"classes", "ignored_reference_labels"},
        "invalid taxonomy",
    )
    classes = taxonomy["classes"]
    require(isinstance(classes, list) and 1 <= len(classes) <= 256, "invalid classes")
    for row in classes:
        require(
            isinstance(row, dict)
            and set(row) == {"id", "name"}
            and integer(row["id"])
            and 0 <= row["id"] <= 255
            and comparison.nonempty(row["name"]),
            "invalid class",
        )
    ids = [row["id"] for row in classes]
    require(
        len(set(ids)) == len(ids) and len({row["name"] for row in classes}) == len(classes),
        "duplicate class",
    )
    ignored = taxonomy["ignored_reference_labels"]
    require(
        isinstance(ignored, list)
        and all(integer(v) for v in ignored)
        and len(set(ignored)) == len(ignored)
        and not set(ids) & set(ignored),
        "invalid ignored labels",
    )
    sources = spec["sources"]
    require(
        isinstance(sources, list) and 1 <= len(sources) <= MAX_SOURCES, "bounded sources required"
    )
    for row in sources:
        require(
            isinstance(row, dict)
            and set(row) == {"artifact", "points", "scene", "acquisition"}
            and comparison.nonempty(row["artifact"])
            and integer(row["points"])
            and row["points"] > 0
            and comparison.nonempty(row["scene"])
            and comparison.nonempty(row["acquisition"]),
            "invalid source",
        )
    source_names = {row["artifact"] for row in sources}
    require(len(source_names) == len(sources), "duplicate source")
    require(sum(row["points"] for row in sources) <= MAX_POINTS, "point limit exceeded")
    thin = spec["thin_objects"]
    if thin is not None:
        require(
            isinstance(thin, dict)
            and set(thin) == {"threshold", "definition", "provenance"}
            and comparison.nonempty(thin["definition"])
            and comparison.nonempty(thin["provenance"]),
            "width annotation provenance required",
        )
        require(decimal(thin["threshold"]) > 0, "positive thin-object threshold required")
    expected = source_names | {"observations"} | ({"object_widths"} if thin else set())
    require(not source_names & {"observations", "object_widths"}, "reserved artifact name")
    require(
        isinstance(spec["artifacts"], dict) and set(spec["artifacts"]) == expected,
        "missing or unused artifacts",
    )
    for entry in spec["artifacts"].values():
        require(isinstance(entry, dict) and set(entry) <= {"path", "sha256"}, "invalid artifact")
        if sealed or "sha256" in entry:
            comparison.sha256(entry.get("sha256"))
    geometry = spec["geometry"]
    require(
        isinstance(geometry, dict)
        and set(geometry) == {"crs", "units", "minimum_xyz", "maximum_xyz"}
        and comparison.nonempty(geometry["crs"])
        and geometry["units"] == "metres",
        "explicit metric CRS/domain required",
    )
    low, high = vector(geometry["minimum_xyz"]), vector(geometry["maximum_xyz"])
    require(all(a < b for a, b in zip(low, high)), "nonempty half-open domain required")
    require(decimal(spec["boundary_radius"]) > 0, "positive boundary radius required")
    resolutions = spec["resolutions"]
    require(
        isinstance(resolutions, list) and 1 <= len(resolutions) <= MAX_RESOLUTIONS,
        "bounded selected resolutions required",
    )
    names, sizes = set(), set()
    for row in resolutions:
        require(
            isinstance(row, dict)
            and set(row) == {"name", "voxel_size_xyz"}
            and comparison.nonempty(row["name"]),
            "invalid resolution",
        )
        size = vector(row["voxel_size_xyz"], positive=True)
        require(row["name"] not in names and size not in sizes, "duplicate resolution")
        names.add(row["name"])
        sizes.add(size)
    require(
        (len(STRATA) + 1) * len(resolutions) * len(classes) ** 2 <= MAX_MATRIX_CELLS,
        "aggregate matrix work limit exceeded",
    )
    accounting = spec["accounting"]
    require(
        isinstance(accounting, dict)
        and set(accounting) == {"feature_channels", "feature_bytes_per_scalar"}
        and all(integer(v) and 1 <= v <= 65536 for v in accounting.values()),
        "explicit positive feature allocation assumptions required",
    )
    return classes, low, high


def _hash(path, budget):
    digest, total = hashlib.sha256(), 0
    with path.open("rb") as stream:
        while raw := stream.read(1024 * 1024):
            total += len(raw)
            budget[0] -= len(raw)
            require(total <= MAX_ARTIFACT_BYTES and budget[0] >= 0, "artifact byte limit exceeded")
            digest.update(raw)
    return digest.hexdigest()


def _bind(spec, base):
    # Admit every file before reading any content; hash-time bounds also cover file growth.
    paths = {name: artifact_path(base, entry) for name, entry in spec["artifacts"].items()}
    sizes = [path.stat().st_size for path in paths.values()]
    require(
        all(size <= MAX_ARTIFACT_BYTES for size in sizes) and sum(sizes) <= MAX_TOTAL_BYTES,
        "artifact byte admission exceeded",
    )
    budget = [MAX_TOTAL_BYTES]
    for name, path in paths.items():
        digest = _hash(path, budget)
        entry = spec["artifacts"][name]
        require("sha256" not in entry or entry["sha256"] == digest, "artifact identity mismatch")
        entry["sha256"] = digest
    identities = [spec["artifacts"][row["artifact"]]["sha256"] for row in spec["sources"]]
    require(len(set(identities)) == len(identities), "duplicate source identity")
    return paths


def seal_manifest(spec, base_dir):
    """Freeze selected resolutions and actual inputs before examining diagnostic results."""
    spec = clone(spec)
    _validate(spec)
    _bind(spec, base_dir)
    spec["manifest_sha256"] = canonical_hash(spec)
    return spec


def _records(path, digest, budget, expected):
    rows, observed = [], hashlib.sha256()
    with path.open("rb") as stream:
        while raw := stream.readline(comparison.MAX_RECORD_BYTES + 1):
            budget[0] -= len(raw)
            require(
                len(raw) <= comparison.MAX_RECORD_BYTES and budget[0] >= 0,
                "parsed byte limit exceeded",
            )
            require(len(rows) < expected, "extra original point")
            observed.update(raw)
            row = comparison.strict_json(raw)
            require(isinstance(row, dict), "record object required")
            rows.append(row)
    require(len(rows) == expected, "missing original point")
    require(observed.hexdigest() == digest, "consumed artifact identity mismatch")
    return rows


def _boundary(points, radius):
    """Resolution-independent, exact closed-radius semantic neighborhoods, per source."""
    buckets, result, pairs = {}, set(), 0
    offsets = tuple(itertools.product((-1, 0, 1), repeat=3))
    for index, (source, xyz, label) in enumerate(points):
        if label is None:
            continue
        cell = tuple(v // radius for v in xyz)
        for offset in offsets:
            key = (source, *(v + d for v, d in zip(cell, offset)))
            for previous in buckets.get(key, ()):
                pairs += 1
                require(pairs <= MAX_BOUNDARY_PAIRS, "boundary pair work limit exceeded")
                other = points[previous]
                if (
                    other[2] != label
                    and sum((a - b) ** 2 for a, b in zip(xyz, other[1])) <= radius**2
                ):
                    result.update((index, previous))
        buckets.setdefault((source, *cell), []).append(index)
    return result, pairs


def analyze(spec, base_dir):
    spec = clone(spec)
    classes, low, high = _validate(spec, sealed=True)
    identity = spec.pop("manifest_sha256")
    comparison.sha256(identity)
    require(canonical_hash(spec) == identity, "frozen manifest identity mismatch")
    paths = _bind(spec, base_dir)
    spec["manifest_sha256"] = identity
    expected = sum(row["points"] for row in spec["sources"])
    budget = [MAX_PARSED_BYTES]
    observations = _records(
        paths["observations"], spec["artifacts"]["observations"]["sha256"], budget, expected
    )
    widths = None
    if spec["thin_objects"]:
        widths = _records(
            paths["object_widths"], spec["artifacts"]["object_widths"]["sha256"], budget, expected
        )
    class_index = {row["id"]: i for i, row in enumerate(classes)}
    allowed = set(class_index) | set(spec["taxonomy"]["ignored_reference_labels"])
    points, width_values, objects = [], [], {}
    for source in spec["sources"]:
        source_hash = spec["artifacts"][source["artifact"]]["sha256"]
        for index in range(source["points"]):
            position = len(points)
            row = observations[position]
            require(
                set(row) == {"source_sha256", "point_index", "xyz", "label"},
                "invalid observation fields",
            )
            require(
                row["source_sha256"] == source_hash
                and integer(row["point_index"])
                and row["point_index"] == index,
                "source/index coverage mismatch",
            )
            require(integer(row["label"]) and row["label"] in allowed, "label outside taxonomy")
            xyz = vector(row["xyz"])
            require(all(a <= v < b for a, v, b in zip(low, xyz, high)), "point outside domain")
            points.append((source_hash, xyz, class_index.get(row["label"])))
            width = None
            if widths is not None:
                item = widths[position]
                require(
                    set(item) == {"source_sha256", "point_index", "object_id", "width"},
                    "invalid width record",
                )
                require(
                    item["source_sha256"] == source_hash
                    and integer(item["point_index"])
                    and item["point_index"] == index,
                    "width source/index mismatch",
                )
                if item["width"] is None:
                    require(item["object_id"] is None, "unknown width must have null object id")
                else:
                    require(comparison.nonempty(item["object_id"]), "object id required")
                    width = decimal(item["width"])
                    require(width > 0, "positive object width required")
                    key = source_hash, item["object_id"]
                    require(
                        key not in objects or objects[key] == width, "inconsistent object width"
                    )
                    objects[key] = width
            width_values.append(width)
    boundary, pairs = _boundary(points, decimal(spec["boundary_radius"]))
    results = []
    for resolution in spec["resolutions"]:
        size = vector(resolution["voxel_size_xyz"], positive=True)
        dimensions = [-(-(b - a) // step) for a, b, step in zip(low, high, size)]
        dense_cells = dimensions[0] * dimensions[1] * dimensions[2]
        cells, point_cells = {}, []
        for source, xyz, label in points:
            key = source, *(int((v - a) // step) for v, a, step in zip(xyz, low, size))
            point_cells.append(key)
            counts = cells.setdefault(key, {})
            if label is not None:
                counts[label] = counts.get(label, 0) + 1
        majority = {
            key: min(counts, key=lambda c: (-counts[c], classes[c]["id"]))
            for key, counts in cells.items()
            if counts
        }
        mixed = {key for key, counts in cells.items() if len(counts) > 1}
        matrices = {name: [[0] * len(classes) for _ in classes] for name in ("all", *STRATA)}
        collision_points = [0] * len(classes)
        collision_cells = [0] * len(classes)
        for key in mixed:
            for label in cells[key]:
                collision_cells[label] += 1
        for position, (_, _, label) in enumerate(points):
            if label is None:
                continue
            key = point_cells[position]
            predicted = majority[key]
            names = ["all", "boundary" if position in boundary else "interior"]
            if widths is not None:
                width = width_values[position]
                names.append(
                    "unknown_object_width"
                    if width is None
                    else "thin_object"
                    if width <= decimal(spec["thin_objects"]["threshold"])
                    else "other_object"
                )
            for name in names:
                matrices[name][label][predicted] += 1
            collision_points[label] += key in mixed
        summary = comparison.metric_summary(matrices["all"], classes)
        for i, row in enumerate(summary["classes"]):
            row.update(
                majority_errors=row["support"] - matrices["all"][i][i],
                majority_error_rate=(row["support"] - matrices["all"][i][i]) / row["support"]
                if row["support"]
                else None,
                mixed_cell_points=collision_points[i],
                mixed_cells=collision_cells[i],
                mixed_cell_point_rate=collision_points[i] / row["support"]
                if row["support"]
                else None,
            )
        scored_cells = len(majority)
        feature_bytes = (
            spec["accounting"]["feature_channels"] * spec["accounting"]["feature_bytes_per_scalar"]
        )
        results.append(
            {
                "resolution": resolution,
                "majority_point_accuracy_oracle": summary,
                "collisions": {
                    "occupied_cells_all_points": len(cells),
                    "occupied_scored_cells": scored_cells,
                    "ignored_only_cells": len(cells) - scored_cells,
                    "mixed_label_cells": len(mixed),
                    "mixed_label_cell_rate": len(mixed) / scored_cells if scored_cells else None,
                    "scored_points_in_mixed_cells": sum(collision_points),
                    "scored_point_collision_rate": sum(collision_points) / summary["scored_points"]
                    if summary["scored_points"]
                    else None,
                },
                "strata": {
                    name: {
                        "status": "NOT_SUPPORTED"
                        if name in STRATA[2:] and widths is None
                        else "MEASURED",
                        "metrics": None
                        if name in STRATA[2:] and widths is None
                        else comparison.metric_summary(matrices[name], classes),
                    }
                    for name in STRATA
                },
                "allocation": {
                    "dense_dimensions_xyz": dimensions,
                    "dense_cells_per_source_domain": dense_cells,
                    "dense_feature_bytes_per_source_domain": dense_cells * feature_bytes,
                    "dense_cells_all_source_domains": dense_cells * len(spec["sources"]),
                    "occupied_feature_payload_bytes_all_sources": len(cells) * feature_bytes,
                    "peak_memory_bytes": None,
                    "context_points": expected,
                    "context_scope": (
                        "whole frozen source; sources never share cells or neighborhoods"
                    ),
                    "allocation_scope": (
                        "feature payload estimate only; excludes activations, indices and overhead"
                    ),
                },
            }
        )
    baseline = results[0]["allocation"]["dense_cells_per_source_domain"]
    for row in results:
        row["allocation"]["dense_cell_ratio_to_first_resolution"] = (
            row["allocation"]["dense_cells_per_source_domain"] / baseline
        )
    return {
        "schema": SCHEMA,
        "measurement_status": "COMPLETE",
        "qualification_status": "NOT_ASSESSED",
        "manifest": spec,
        "manifest_sha256": identity,
        "scorer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "metric_helper_sha256": hashlib.sha256(Path(comparison.__file__).read_bytes()).hexdigest(),
        "coverage": {
            "original_points": expected,
            "scored_points": sum(p[2] is not None for p in points),
            "ignored_points": sum(p[2] is None for p in points),
            "missing_points": 0,
        },
        "boundary_pairs_examined": pairs,
        "results": results,
        "rules": {
            "partition": (
                "source-specific floor((xyz-minimum_xyz)/voxel_size); half-open metric domain"
            ),
            "majority": "scored labels only; ties choose smallest numeric class id",
            "accuracy": (
                "maximum point accuracy among one-label-per-cell assignments on these labels"
            ),
            "miou": "diagnostic only; majority mIoU is not a mathematical upper bound",
            "boundary": (
                "point has differently labelled scored neighbor at distance <= radius "
                "in same source"
            ),
            "thin_object": (
                "externally annotated object width <= frozen threshold; null width is unknown"
            ),
            "strata": "global cell majority is scored in each stratum; no refitting within strata",
            "metrics": (
                "rows truth, columns prediction; absent union null; all-class macro uses zero"
            ),
        },
        "unsupported": {
            "trained_model_quality": "No model executed",
            "timing": "Not measured",
            "gpu": "Not used",
            "survey_terrain": "No independent terrain reference",
        },
        "limitations": [
            "Validation labels define an oracle diagnostic, not deployable predictions "
            "or trained-model quality.",
            "Source and annotation bytes are verified; observation derivation and annotation "
            "correctness are external.",
            "Boundary is sampled semantic adjacency, not a surveyed boundary "
            "or object reconstruction.",
            "Feature payload estimates do not establish native capacity, runtime memory "
            "or latency.",
            "Freeze identity does not attest wall-clock preregistration "
            "or rights/split provenance.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("seal", "analyze"))
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require(args.spec.resolve() != args.out.resolve(), "output aliases manifest")
    spec = comparison.load_spec(args.spec)
    result = seal_manifest(spec, args.base) if args.action == "seal" else analyze(spec, args.base)
    with args.out.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
