# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
"""Cluster dependence, rare-class availability and invalid paired evidence checks."""

import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

MODULE = Path(__file__).resolve().parents[1] / "tools/runtime_uncertainty.py"
SPEC = importlib.util.spec_from_file_location("runtime_uncertainty", MODULE)
uncertainty = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(uncertainty)


def fixture_spec():
    scenes = []
    for i in range(4):
        reference, candidate = ([[100, 0], [0, 0]], [[0, 100], [0, 0]])
        if i >= 2:
            reference, candidate = candidate, reference
        scenes.append({
            "scene_id": f"scene-{i}", "cluster_id": f"acquisition-{i // 2}",
            "source_sha256": [hashlib.sha256(str(i).encode()).hexdigest()],
            "evidence_sha256": "f" * 64,
            "reference_confusion": reference, "candidate_confusion": candidate,
        })
    classes = [{"id": 0, "name": "ground"}, {"id": 255, "name": "other"}]
    return {
        "schema": uncertainty.SCHEMA, "classes": classes, "scenes": scenes,
        "plan": {
            "replicates": 1000, "seed": 49, "confidence": 0.95, "minimum_clusters": 2,
            "population_sha256": uncertainty.comparison.canonical_hash(
                uncertainty.population(scenes)),
            "taxonomy_sha256": uncertainty.comparison.canonical_hash(classes),
        },
    }


def test_paired_cluster_draws_preserve_dependence_and_hand_computed_metrics():
    spec = fixture_spec()
    result = uncertainty.analyze(spec)
    assert result["scenes"] == 4 and result["clusters"] == 2
    assert result["metrics"]["accuracy"] == {
        "candidate_minus_reference": 0, "percentile_interval": [-1, 1],
        "undefined_replicates": 0, "status": "REPORTED",
    }
    assert result["pooled_quality"]["candidate"]["accuracy"] == 0.5
    assert result["pooled_quality"]["candidate"]["mean_iou_all_classes"] == 0.25
    assert result["class_support"] == [
        {"id": 0, "name": "ground", "points": 400, "scenes": 4, "clusters": 2,
         "objects": None},
        {"id": 255, "name": "other", "points": 0, "scenes": 0, "clusters": 0,
         "objects": None},
    ]
    assert result["qualification_status"] == "NOT_ASSESSED"
    assert result == uncertainty.analyze(spec)  # deterministic paired sample stream
    assert result["plan"] == spec["plan"]


def test_splitting_scenes_within_same_acquisition_cannot_increase_sample_size():
    spec = fixture_spec()
    original = uncertainty.analyze(spec)
    spec["scenes"] = [spec["scenes"][0], spec["scenes"][2]]
    for scene in spec["scenes"]:
        for name in ("reference_confusion", "candidate_confusion"):
            scene[name] = [[2 * n for n in row] for row in scene[name]]
    spec["plan"]["population_sha256"] = uncertainty.comparison.canonical_hash(
        uncertainty.population(spec["scenes"]))
    merged = uncertainty.analyze(spec)
    assert merged["metrics"] == original["metrics"]
    assert merged["pooled_quality"] == original["pooled_quality"]


def test_inadequate_clusters_and_all_ignored_samples_remain_inconclusive():
    spec = fixture_spec()
    spec["plan"]["minimum_clusters"] = 10
    result = uncertainty.analyze(spec)
    assert all(v["status"] == "INCONCLUSIVE" for v in result["metrics"].values())
    for scene in spec["scenes"]:
        scene["cluster_id"] = "one-flight"
        scene["reference_confusion"] = scene["candidate_confusion"] = [[0, 0], [0, 0]]
    spec["plan"]["population_sha256"] = uncertainty.comparison.canonical_hash(
        uncertainty.population(spec["scenes"]))
    result = uncertainty.analyze(spec)
    assert result["clusters"] == 1 and result["method"]["evaluated_replicates"] == 0
    assert all(v["percentile_interval"] is None
               and v["candidate_minus_reference"] is None for v in result["metrics"].values())


def test_undefined_draws_are_not_discarded_to_narrow_intervals():
    spec = fixture_spec()
    for scene in spec["scenes"][:2]:
        scene["reference_confusion"] = scene["candidate_confusion"] = [[0, 0], [0, 0]]
    result = uncertainty.analyze(spec)
    accuracy = result["metrics"]["accuracy"]
    assert 150 < accuracy["undefined_replicates"] < 350
    assert accuracy["percentile_interval"] is None and accuracy["status"] == "INCONCLUSIVE"


@pytest.mark.parametrize("mutation", [
    "support", "duplicate_source", "duplicate_scene", "bool_count", "negative",
    "shape", "population", "taxonomy", "nan", "replicates", "seed", "confidence",
    "min_clusters", "evidence", "overflow", "work_limit", "memory_limit",
])
def test_invalid_pairing_lineage_and_unbounded_work_fail(mutation):
    spec = fixture_spec()
    scene = spec["scenes"][0]
    if mutation == "support":
        scene["candidate_confusion"][0][0] += 1
    elif mutation == "duplicate_source":
        scene["source_sha256"] = spec["scenes"][1]["source_sha256"]
    elif mutation == "duplicate_scene":
        scene["scene_id"] = spec["scenes"][1]["scene_id"]
    elif mutation in {"bool_count", "negative", "nan", "overflow"}:
        scene["candidate_confusion"][0][0] = {
            "bool_count": True, "negative": -1, "nan": float("nan"), "overflow": 2**53 + 1
        }[mutation]
    elif mutation == "shape":
        scene["reference_confusion"].pop()
    elif mutation in {"population", "taxonomy"}:
        spec["plan"][mutation + "_sha256"] = "a" * 64
    elif mutation == "evidence":
        scene["evidence_sha256"] = "unchecked"
    elif mutation == "work_limit":
        spec["plan"]["replicates"] = 100_000
        spec["classes"] = [{"id": i, "name": str(i)} for i in range(7)]
        for row in spec["scenes"]:
            row["reference_confusion"] = row["candidate_confusion"] = [[0] * 7 for _ in range(7)]
        spec["plan"]["taxonomy_sha256"] = uncertainty.comparison.canonical_hash(spec["classes"])
        # 100000 * ((2+1)*49+7) is within the work cap; lower it for this bounded fixture.
        from unittest.mock import patch
        with patch.object(uncertainty, "MAX_WORK", 100):
            with pytest.raises(ValueError, match="work limit"):
                uncertainty.analyze(spec)
        return
    elif mutation == "memory_limit":
        from unittest.mock import patch
        with patch.object(uncertainty, "MAX_MATRIX_ENTRIES", 1):
            with pytest.raises(ValueError, match="memory limit"):
                uncertainty.analyze(spec)
        return
    else:
        spec["plan"][mutation] = {"replicates": 99, "seed": True, "confidence": 1,
                                  "min_clusters": 1}[mutation]
        if mutation == "min_clusters":
            spec["plan"]["minimum_clusters"] = spec["plan"].pop("min_clusters")
    with pytest.raises(ValueError):
        uncertainty.analyze(spec)


def test_cli_retains_inconclusive_report_and_never_overwrites(tmp_path):
    spec = fixture_spec()
    spec["plan"]["minimum_clusters"] = 10
    path, out = tmp_path / "analysis.json", tmp_path / "result.json"
    path.write_text(json.dumps(spec))
    args = [sys.executable, str(MODULE), "--spec", str(path), "--out", str(out)]
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    report = json.loads(out.read_text())
    assert report["metrics"]["accuracy"]["status"] == "INCONCLUSIVE"
    before = out.read_bytes()
    assert subprocess.run(args, capture_output=True, check=False).returncode != 0
    assert out.read_bytes() == before
    assert uncertainty.analyze(copy.deepcopy(spec))["spec_sha256"] == report["spec_sha256"]
