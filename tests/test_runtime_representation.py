# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
"""Independent combinatorial and geometric checks; fixtures execute no model."""

import copy
import hashlib
import importlib.util
import itertools
import json
import random
import subprocess
import sys
from decimal import ROUND_FLOOR, Decimal
from pathlib import Path

import pytest

MODULE = Path(__file__).resolve().parents[1] / "tools/runtime_representation.py"
LOADER = importlib.util.spec_from_file_location("runtime_representation", MODULE)
representation = importlib.util.module_from_spec(LOADER)
LOADER.loader.exec_module(representation)


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def make_case(tmp_path, labels=(9, 2, 9, 2, -1), xyz=None, widths=True):
    (tmp_path / "raw").write_bytes(b"raw source fixture; no trained inference")
    source = hashlib.sha256((tmp_path / "raw").read_bytes()).hexdigest()
    if xyz is None:
        xyz = [
            ("0.1", "0", "0"),
            ("0.2", "0", "0"),
            ("0.3", "0", "0"),
            ("1.1", "0", "0"),
            ("1.2", "0", "0"),
        ]
    rows = [
        {"source_sha256": source, "point_index": i, "xyz": list(point), "label": label}
        for i, (label, point) in enumerate(zip(labels, xyz))
    ]
    write_rows(tmp_path / "observations", rows)
    spec = {
        "schema": representation.SCHEMA,
        "purpose": "representation_diagnostic",
        "split": "validation",
        "taxonomy": {
            "classes": [
                {"id": 9, "name": "vegetation"},
                {"id": 2, "name": "ground"},
                {"id": 17, "name": "absent"},
            ],
            "ignored_reference_labels": [-1],
        },
        "sources": [
            {
                "artifact": "raw",
                "points": len(labels),
                "scene": "scene-a",
                "acquisition": "flight-a",
            }
        ],
        "artifacts": {name: {"path": name} for name in ("raw", "observations")},
        "geometry": {
            "crs": "local fixture metric frame",
            "units": "metres",
            "minimum_xyz": ["0", "0", "0"],
            "maximum_xyz": ["2", "2", "2"],
        },
        "resolutions": [
            {"name": "coarse", "voxel_size_xyz": ["1", "1", "1"]},
            {"name": "fine", "voxel_size_xyz": ["0.5", "0.5", "0.5"]},
        ],
        "boundary_radius": "0.1",
        "thin_objects": None,
        "accounting": {"feature_channels": 6, "feature_bytes_per_scalar": 4},
    }
    if widths:
        spec["thin_objects"] = {
            "threshold": "0.2",
            "definition": "surveyed minimum object width",
            "provenance": "synthetic annotated fixture only",
        }
        spec["artifacts"]["object_widths"] = {"path": "object_widths"}
        write_rows(
            tmp_path / "object_widths",
            [
                {
                    "source_sha256": source,
                    "point_index": i,
                    "object_id": "thin" if i == 0 else None if i == 2 else "wide",
                    "width": "0.2" if i == 0 else None if i == 2 else "1",
                }
                for i in range(len(labels))
            ],
        )
    return spec, rows


def run(spec, base):
    frozen = representation.seal_manifest(spec, base)
    return representation.analyze(frozen, base)


def independent_matrix(labels, predictions, ids):
    return [[sum(t == a and p == b for t, p in zip(labels, predictions)) for b in ids] for a in ids]


def independent_miou(labels, predictions, ids):
    # Set intersection/union, independent of the scorer's confusion arithmetic.
    values = []
    for c in ids:
        truth = {i for i, label in enumerate(labels) if label == c}
        predicted = {i for i, label in enumerate(predictions) if label == c}
        values.append(len(truth & predicted) / len(truth | predicted) if truth | predicted else 0)
    return sum(values) / len(ids)


def test_class_errors_collisions_strata_and_allocation(tmp_path):
    spec, _ = make_case(tmp_path)
    result = run(spec, tmp_path)
    coarse = result["results"][0]
    metrics = coarse["majority_point_accuracy_oracle"]
    assert metrics["confusion"] == [[2, 0, 0], [1, 1, 0], [0, 0, 0]]
    assert metrics["accuracy"] == 3 / 4
    assert metrics["classes"][1]["majority_errors"] == 1
    assert metrics["classes"][1]["mixed_cell_point_rate"] == 1 / 2
    assert metrics["classes"][2]["majority_error_rate"] is None
    assert coarse["collisions"]["mixed_label_cell_rate"] == 1 / 2
    assert coarse["collisions"]["scored_point_collision_rate"] == 3 / 4
    assert coarse["strata"]["boundary"]["metrics"]["scored_points"] == 3
    assert coarse["strata"]["thin_object"]["metrics"]["scored_points"] == 1
    assert coarse["strata"]["unknown_object_width"]["metrics"]["scored_points"] == 1
    assert coarse["allocation"]["dense_dimensions_xyz"] == [2, 2, 2]
    assert coarse["allocation"]["dense_feature_bytes_per_source_domain"] == 2**3 * 6 * 4
    assert result["results"][1]["allocation"]["dense_cell_ratio_to_first_resolution"] == 8
    assert result["results"][1]["allocation"]["peak_memory_bytes"] is None


def test_ties_numeric_label_order_and_no_refit_in_strata(tmp_path):
    spec, _ = make_case(tmp_path, labels=[9, 2], xyz=[("0", "0", "0"), ("0.1", "0", "0")])
    result = run(spec, tmp_path)["results"][0]
    assert result["majority_point_accuracy_oracle"]["confusion"] == [
        [0, 1, 0],
        [0, 1, 0],
        [0, 0, 0],
    ]
    # The thin point's own class is 9; it must retain the full-cell prediction 2.
    assert result["strata"]["thin_object"]["metrics"]["accuracy"] == 0
    spec["taxonomy"]["classes"].reverse()
    reordered = run(spec, tmp_path)["results"][0]["majority_point_accuracy_oracle"]
    assert next(row for row in reordered["classes"] if row["id"] == 2)["predicted"] == 2


def test_exhaustive_partition_accuracy_and_miou_counterexample(tmp_path):
    labels = [2] + [9] * 3 + [2] * 4
    xyz = [("0", "0", "0")] + [("1", "0", "0")] * 7
    spec, _ = make_case(tmp_path, labels, xyz, widths=False)
    result = run(spec, tmp_path)["results"][0]["majority_point_accuracy_oracle"]
    assignments = [[a] + [b] * 7 for a, b in itertools.product((9, 2), repeat=2)]
    accuracies = [sum(a == b for a, b in zip(labels, pred)) / len(labels) for pred in assignments]
    assert result["accuracy"] == max(accuracies) == 5 / 8
    ids = [9, 2, 17]
    assert result["mean_iou_all_classes"] == independent_miou(labels, [2] * 8, ids)
    assert (
        max(independent_miou(labels, pred, ids) for pred in assignments)
        > result["mean_iou_all_classes"]
    )


def test_random_cloud_against_pairwise_geometry_and_explicit_partitions(tmp_path):
    rng = random.Random(1856)
    labels = [rng.choice((2, 9, -1)) for _ in range(18)]
    xyz = [tuple(str(Decimal(rng.randrange(0, 8)) / 4) for _ in range(3)) for _ in labels]
    spec, rows = make_case(tmp_path, labels, xyz, widths=False)
    spec["boundary_radius"] = "0.5"
    result = run(spec, tmp_path)
    ids = [9, 2, 17]
    # Direct all-pairs squared-distance calculation does not use spatial hash buckets.
    boundary = {
        i
        for i in range(len(rows))
        for j in range(len(rows))
        if labels[i] != -1
        and labels[j] != -1
        and labels[i] != labels[j]
        and sum((Decimal(xyz[i][d]) - Decimal(xyz[j][d])) ** 2 for d in range(3)) <= Decimal("0.25")
    }
    for measured, resolution in zip(result["results"], spec["resolutions"]):
        groups = {}
        for i, point in enumerate(xyz):
            key = tuple(
                (Decimal(point[d]) / Decimal(resolution["voxel_size_xyz"][d])).to_integral_value(
                    rounding=ROUND_FLOOR
                )
                for d in range(3)
            )
            groups.setdefault(key, []).append(i)
        predicted = {}
        for group in groups.values():
            scored = [i for i in group if labels[i] != -1]
            if scored:
                options = [
                    (sum(labels[i] == label for i in scored), -label, label) for label in ids
                ]
                winner = max(options)[2]
                predicted.update({i: winner for i in scored})
        truth = [labels[i] for i in sorted(predicted)]
        guesses = [predicted[i] for i in sorted(predicted)]
        assert measured["majority_point_accuracy_oracle"]["confusion"] == independent_matrix(
            truth, guesses, ids
        )
        assert measured["strata"]["boundary"]["metrics"]["scored_points"] == len(boundary)
        assert measured["strata"]["boundary"]["metrics"]["confusion"] == independent_matrix(
            [labels[i] for i in sorted(boundary)], [predicted[i] for i in sorted(boundary)], ids
        )
        mixed = [g for g in groups.values() if len({labels[i] for i in g if labels[i] != -1}) > 1]
        assert measured["collisions"]["mixed_label_cells"] == len(mixed)
        for name in ("thin_object", "other_object", "unknown_object_width"):
            assert measured["strata"][name] == {"status": "NOT_SUPPORTED", "metrics": None}


def test_exact_large_translation_negative_boundaries_and_half_open_domain(tmp_path):
    spec, rows = make_case(
        tmp_path,
        labels=[9, 2, 9],
        xyz=[("0", "0", "0"), ("0.499999999999999999", "0", "0"), ("0.5", "0", "0")],
        widths=False,
    )
    before = run(spec, tmp_path)
    shift = [Decimal("-1000000000000.25"), Decimal("6700000.000000001"), Decimal("-2")]
    for row in rows:
        row["xyz"] = [str(Decimal(v) + shift[d]) for d, v in enumerate(row["xyz"])]
    # Decimal context must retain the intended full integer+fraction precision.
    from decimal import localcontext

    with localcontext() as context:
        context.prec = 60
        rows[1]["xyz"][0] = str(Decimal("0.499999999999999999") + shift[0])
    for field in ("minimum_xyz", "maximum_xyz"):
        spec["geometry"][field] = [
            str(Decimal(v) + shift[d]) for d, v in enumerate(spec["geometry"][field])
        ]
    write_rows(tmp_path / "observations", rows)
    after = run(spec, tmp_path)
    assert before["results"] == after["results"]
    rows[0]["xyz"] = spec["geometry"]["maximum_xyz"]
    write_rows(tmp_path / "observations", rows)
    with pytest.raises(ValueError, match="outside domain"):
        run(spec, tmp_path)


def test_ignored_labels_never_form_majority_or_boundary(tmp_path):
    spec, _ = make_case(
        tmp_path, labels=[-1, -1], xyz=[("0", "0", "0"), ("1", "0", "0")], widths=False
    )
    result = run(spec, tmp_path)
    row = result["results"][0]
    assert row["majority_point_accuracy_oracle"]["accuracy"] is None
    assert row["collisions"]["ignored_only_cells"] == 2
    assert row["collisions"]["mixed_label_cell_rate"] is None
    assert row["strata"]["boundary"]["metrics"]["scored_points"] == 0
    assert result["coverage"] == {
        "original_points": 2,
        "scored_points": 0,
        "ignored_points": 2,
        "missing_points": 0,
    }


def test_sources_do_not_share_voxels_or_boundary(tmp_path):
    spec, rows = make_case(tmp_path, labels=[9], xyz=[("0", "0", "0")], widths=False)
    (tmp_path / "raw2").write_bytes(b"second independent source")
    source2 = hashlib.sha256((tmp_path / "raw2").read_bytes()).hexdigest()
    spec["sources"].append({**spec["sources"][0], "artifact": "raw2"})
    spec["artifacts"]["raw2"] = {"path": "raw2"}
    rows.append({**rows[0], "source_sha256": source2, "label": 2})
    write_rows(tmp_path / "observations", rows)
    result = run(spec, tmp_path)["results"][0]
    assert result["majority_point_accuracy_oracle"]["accuracy"] == 1
    assert result["collisions"]["occupied_scored_cells"] == 2
    assert result["collisions"]["mixed_label_cells"] == 0
    assert result["strata"]["boundary"]["metrics"]["scored_points"] == 0
    assert result["allocation"]["dense_cells_all_source_domains"] == 16


@pytest.mark.parametrize("target", ["raw", "observations", "object_widths"])
def test_post_freeze_artifact_mutation_fails(tmp_path, target):
    spec, _ = make_case(tmp_path)
    frozen = representation.seal_manifest(spec, tmp_path)
    (tmp_path / target).write_bytes((tmp_path / target).read_bytes() + b" ")
    with pytest.raises(ValueError, match="identity mismatch"):
        representation.analyze(frozen, tmp_path)


def test_frozen_resolution_mutation_and_existing_hash_never_refresh(tmp_path):
    spec, _ = make_case(tmp_path)
    frozen = representation.seal_manifest(spec, tmp_path)
    frozen["resolutions"][0]["voxel_size_xyz"][0] = "0.75"
    with pytest.raises(ValueError, match="manifest identity"):
        representation.analyze(frozen, tmp_path)
    spec["artifacts"]["raw"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="artifact identity"):
        representation.seal_manifest(spec, tmp_path)


@pytest.mark.parametrize(
    "mutation",
    [
        "reorder",
        "duplicate",
        "missing",
        "extra",
        "bool_index",
        "unknown_label",
        "duplicate_json",
        "nan",
        "float_xyz",
        "inconsistent_width",
    ],
)
def test_adversarial_records(tmp_path, mutation):
    spec, rows = make_case(tmp_path)
    if mutation == "reorder":
        rows.reverse()
    elif mutation == "duplicate":
        rows[1] = rows[0]
    elif mutation == "missing":
        rows.pop()
    elif mutation == "extra":
        rows.append(rows[0])
    elif mutation == "bool_index":
        rows[0]["point_index"] = False
    elif mutation == "unknown_label":
        rows[0]["label"] = 256
    elif mutation == "float_xyz":
        rows[0]["xyz"][0] = 0.1
    write_rows(tmp_path / "observations", rows)
    if mutation == "duplicate_json":
        (tmp_path / "observations").write_text('{"label":9,"label":2}\n')
    elif mutation == "nan":
        (tmp_path / "observations").write_text('{"xyz":NaN}\n')
    elif mutation == "inconsistent_width":
        widths = [json.loads(row) for row in (tmp_path / "object_widths").read_text().splitlines()]
        widths[-1]["width"] = "0.8"
        write_rows(tmp_path / "object_widths", widths)
    with pytest.raises(ValueError):
        run(spec, tmp_path)


@pytest.mark.parametrize(
    "mutation",
    [
        "holdout",
        "zero_size",
        "duplicate_size",
        "bad_units",
        "width_provenance",
        "unused_artifact",
        "traversal",
        "bool_feature",
        "huge_decimal",
    ],
)
def test_adversarial_manifest(tmp_path, mutation):
    spec, _ = make_case(tmp_path)
    if mutation == "holdout":
        spec["split"] = "test"
    elif mutation == "zero_size":
        spec["resolutions"][0]["voxel_size_xyz"][0] = "0"
    elif mutation == "duplicate_size":
        spec["resolutions"][1]["voxel_size_xyz"] = ["1.0", "1", "1"]
    elif mutation == "bad_units":
        spec["geometry"]["units"] = "degrees"
    elif mutation == "width_provenance":
        spec["thin_objects"]["provenance"] = ""
    elif mutation == "unused_artifact":
        spec["artifacts"]["unused"] = {"path": "raw"}
    elif mutation == "traversal":
        spec["artifacts"]["raw"]["path"] = "../raw"
    elif mutation == "bool_feature":
        spec["accounting"]["feature_channels"] = True
    elif mutation == "huge_decimal":
        spec["boundary_radius"] = "9" * 1000
    with pytest.raises(ValueError):
        representation.seal_manifest(spec, tmp_path)


def test_resource_admission_before_hashing_and_pair_limit(tmp_path, monkeypatch):
    spec, _ = make_case(tmp_path)
    monkeypatch.setattr(representation, "MAX_TOTAL_BYTES", 1)
    monkeypatch.setattr(representation, "_hash", lambda *_: pytest.fail("hash before admission"))
    with pytest.raises(ValueError, match="admission"):
        representation.seal_manifest(spec, tmp_path)
    monkeypatch.undo()
    frozen = representation.seal_manifest(spec, tmp_path)
    monkeypatch.setattr(representation, "MAX_BOUNDARY_PAIRS", 0)
    with pytest.raises(ValueError, match="pair work"):
        representation.analyze(frozen, tmp_path)


def test_matrix_and_record_limits(tmp_path, monkeypatch):
    spec, _ = make_case(tmp_path)
    monkeypatch.setattr(representation, "MAX_MATRIX_CELLS", 1)
    with pytest.raises(ValueError, match="matrix work"):
        representation.seal_manifest(spec, tmp_path)
    monkeypatch.undo()
    monkeypatch.setattr(representation, "MAX_POINTS", 1)
    with pytest.raises(ValueError, match="point limit"):
        representation.seal_manifest(spec, tmp_path)


def test_cli_seal_analyze_exclusive_receipts_and_input_immutability(tmp_path):
    spec, _ = make_case(tmp_path)
    original = copy.deepcopy(spec)
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    frozen, report = tmp_path / "frozen.json", tmp_path / "report.json"
    for action, source, output in (("seal", path, frozen), ("analyze", frozen, report)):
        completed = subprocess.run(
            [
                sys.executable,
                str(MODULE),
                action,
                "--spec",
                str(source),
                "--base",
                str(tmp_path),
                "--out",
                str(output),
            ],
            capture_output=True,
            text=True,
            timeout=20,
        )
        assert completed.returncode == 0, completed.stderr
    payload = json.loads(report.read_text())
    assert payload["manifest_sha256"] == json.loads(frozen.read_text())["manifest_sha256"]
    assert payload["scorer_sha256"] == hashlib.sha256(MODULE.read_bytes()).hexdigest()
    before = report.read_bytes()
    completed = subprocess.run(
        [
            sys.executable,
            str(MODULE),
            "analyze",
            "--spec",
            str(frozen),
            "--base",
            str(tmp_path),
            "--out",
            str(report),
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert completed.returncode != 0
    assert report.read_bytes() == before
    assert spec == original
