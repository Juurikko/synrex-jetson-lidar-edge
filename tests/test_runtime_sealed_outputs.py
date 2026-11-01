# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
"""Real LAS fixtures test sealed-sidecar rescoring without engine execution."""

import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import laspy
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
SPEC = importlib.util.spec_from_file_location(
    "sealed_outputs", ROOT / "tools/runtime_sealed_outputs.py")
sealed = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sealed)


def artifact(path, value):
    path.write_text(json.dumps(value))
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


@pytest.fixture
def fixture(tmp_path):
    taxonomy = {"classes": [{"id": 0, "name": "other"}, {"id": 7, "name": "ground"},
                            {"id": 255, "name": "rare"}],
                "ignored_reference_labels": [-100], "ground_class_id": 7}
    sources = []
    artifacts = []
    runs = {}
    for name, projection in (("reference", [1, 2, 64]), ("candidate", [1, 2, 201])):
        runs[name] = {
            "package_manifest": artifact(tmp_path / (name + "-manifest.json"), {
                "schema": "synrex.srxmodel.v2", "contract": {"las_classes": projection},
                "artifacts": {"engine": {"path": "model.engine", "sha256": "e" * 64}},
            }), "channel_to_label": [0, 7, 255],
        }
    for number, (truth, reference, candidate) in enumerate((
        ([1, 2, 2, 65], [0, 1, 0, 1], [1, 1, 0, 0]),
        ([1, 2], [0, 1], [0, 1]),
    )):
        cloud = laspy.LasData(laspy.LasHeader(point_format=6, version="1.4"))
        cloud.x = np.arange(len(truth)) + number * 100
        cloud.y = np.zeros(len(truth))
        cloud.z = np.ones(len(truth))
        cloud.classification = truth
        path = tmp_path / f"source-{number}.las"
        cloud.write(path)
        source = {"sha256": sealed.digest(path), "points": len(truth)}
        sources.append(source)
        row = {"path": str(path)}
        for name, values, projection in (("reference", reference, [1, 2, 64]),
                                          ("candidate", candidate, [1, 2, 201])):
            row[name] = artifact(tmp_path / f"source-{number}-{name}.json", {
                "schema": "synrex.point_semantics.v1",
                "index_order": "original source point record order",
                "semantic_id_definition": "zero-based model output channel; no LAS class collapse",
                "source_sha256": source["sha256"], "point_count": len(truth),
                "model_manifest_sha256": runs[name]["package_manifest"]["sha256"],
                "las_classes": projection, "semantic_ids": values,
                "mapped_labels_sha256": hashlib.sha256(
                    bytes(projection[v] for v in values)).hexdigest(),
            })
        artifacts.append(row)
    run = {
        "name": "Retained CPU fixture; no inference", "execution_mode": "NATIVE",
        "execution_scope": "file_export", "limitations": ["Synthetic LAS fixture"],
        "unsupported_modes": {"HYBRID": "Not executed"},
        "provenance": {
            "source_sha256": "a" * 64, "config_sha256": "b" * 64,
            "model_sha256": "c" * 64, "engine_sha256": "e" * 64,
            "source_manifest_sha256": sealed.comparison.canonical_hash(sources),
            "scoring_taxonomy_sha256": sealed.comparison.canonical_hash(taxonomy),
            "package_taxonomy_sha256": None,
            "package_taxonomy_unavailable_reason": "Fixture package has no taxonomy artifact",
            "hardware": "CPU fixture", "software": "No inference executed",
        },
        "semantics": {
            "input_fields": ["x", "y", "z"], "preprocessing": {"version": "fixture"},
            "windows": {"version": "whole-source"}, "reconstruction": {"version": "original-order"},
            "precision": "declared FP16; not executed", "policy": "fixture",
            "architecture": "fixture", "input_provenance": "fixture LAS",
        },
        "measurements": {"timing_boundary": "Retained outputs; inference timing unavailable",
                         **dict.fromkeys(("offered_points_per_second", "achieved_points_per_second",
                                          "p50_latency_ms", "p95_latency_ms", "p99_latency_ms",
                                          "max_latency_ms", "peak_memory_bytes", "energy_joules"))},
    }
    return {
        "schema": sealed.SCHEMA, "comparison": {
            "schema": sealed.comparison.SCHEMA, "track": "model_policy_quality",
            "taxonomy": taxonomy, "sources": sources,
            "evaluation": {"split": "synthetic fixture", "protocol": "No inference",
                           "latency_budget_ms": None},
            "reference": copy.deepcopy(run), "candidate": copy.deepcopy(run),
        },
        "source_las_to_label": [{"las_code": k, "label": v}
                                for k, v in ((1, 0), (2, 7), (6, 255), (65, -100))],
        "runs": runs, "sources": artifacts,
    }


def test_actual_raw_las_and_hand_computed_metrics(fixture):
    result = sealed.evaluate(fixture)
    assert result["coverage"] == {"original_points": 6, "scored_points": 5,
                                   "ignored_points": 1, "missing_points": 0}
    assert result["quality"]["reference"]["confusion"] == [[2, 0, 0], [1, 2, 0], [0, 0, 0]]
    assert result["quality"]["candidate"]["confusion"] == [[1, 1, 0], [1, 2, 0], [0, 0, 0]]
    assert result["quality"]["reference"]["accuracy"] == 4 / 5
    assert result["quality"]["candidate"]["mean_iou_all_classes"] == pytest.approx(5 / 18)
    assert result["candidate_minus_reference"]["accuracy"] == pytest.approx(-1 / 5)
    assert result["label_disagreement"]["points"] == 2  # Includes the ignored raw truth.
    assert result["label_disagreement"]["agreement"] == 2 / 3
    assert result["ground_proxy"]["candidate"]["precision"] == 2 / 3
    assert result["ground_proxy"]["candidate"]["dtm_rmse"] is None
    assert result["qualification_status"] == "NOT_ASSESSED"
    assert result["execution"] == "CPU_REEVALUATION_RETAINED_SIDECARS_NO_INFERENCE"
    assert sum(row["points"] for row in result["sources"]) == 6


@pytest.mark.parametrize("field,value", [
    ("schema", "future"), ("index_order", "arrival"),
    ("semantic_id_definition", "projected LAS classes"), ("source_sha256", "0" * 64),
    ("model_manifest_sha256", "0" * 64), ("point_count", True),
    ("point_count", 3), ("las_classes", [1, 2, 200]), ("las_classes", [True, 2, 201]),
    ("semantic_ids", [0, 1, 0]), ("semantic_ids", [0, 1, 0, 256]),
    ("semantic_ids", [0, 1, 0, True]), ("mapped_labels_sha256", "0" * 64),
])
def test_reject_coherently_resealed_wrong_sidecar(fixture, field, value):
    record = fixture["sources"][0]["candidate"]
    path = Path(record["path"])
    info = json.loads(path.read_text())
    info[field] = value
    fixture["sources"][0]["candidate"] = artifact(path, info)
    with pytest.raises(ValueError):
        sealed.evaluate(fixture)


@pytest.mark.parametrize("mutation", ["source-hash", "sidecar-hash", "package-hash", "engine",
                                      "point-limit", "unmapped", "mapping-duplicate",
                                      "mapping-bool", "channel-duplicate", "channel-bool",
                                      "channel-unknown", "parity",
                                      "source-roster", "source-extra", "extra-spec"])
def test_reject_invalid_identity_mapping_and_bounds(fixture, mutation):
    if mutation == "source-hash":
        fixture["comparison"]["sources"][0]["sha256"] = "0" * 64
        manifest_hash = sealed.comparison.canonical_hash(fixture["comparison"]["sources"])
        for name in ("reference", "candidate"):
            fixture["comparison"][name]["provenance"]["source_manifest_sha256"] = manifest_hash
    elif mutation == "sidecar-hash":
        fixture["sources"][0]["candidate"]["sha256"] = "0" * 64
    elif mutation == "package-hash":
        fixture["runs"]["candidate"]["package_manifest"]["sha256"] = "0" * 64
    elif mutation == "engine":
        fixture["comparison"]["candidate"]["provenance"]["engine_sha256"] = "0" * 64
    elif mutation == "point-limit":
        fixture["comparison"]["sources"][0]["points"] = sealed.MAX_SOURCE_POINTS + 1
        for name in ("reference", "candidate"):
            fixture["comparison"][name]["provenance"]["source_manifest_sha256"] = (
                sealed.comparison.canonical_hash(fixture["comparison"]["sources"]))
    elif mutation == "unmapped":
        fixture["source_las_to_label"].pop()
    elif mutation == "mapping-duplicate":
        fixture["source_las_to_label"].append(fixture["source_las_to_label"][0])
    elif mutation == "mapping-bool":
        fixture["source_las_to_label"][0]["las_code"] = True
    elif mutation.startswith("channel-"):
        fixture["runs"]["candidate"]["channel_to_label"][-1] = {
            "channel-duplicate": 0, "channel-bool": True, "channel-unknown": 254,
        }[mutation]
    elif mutation == "parity":
        fixture["comparison"]["track"] = "runtime_parity"
        fixture["comparison"]["minimum_label_agreement"] = 0.995
    elif mutation == "source-roster":
        fixture["sources"].pop()
    elif mutation == "source-extra":
        fixture["sources"][0]["unknown"] = "field"
    else:
        fixture["unknown"] = "field"
    with pytest.raises(ValueError):
        sealed.evaluate(fixture)


def test_chunking_ignored_only_and_empty_classes_are_not_fabricated(fixture, monkeypatch):
    monkeypatch.setattr(sealed, "CHUNK_POINTS", 1)
    fixture["source_las_to_label"] = [{"las_code": v, "label": -100} for v in [1, 2, 65]]
    result = sealed.evaluate(fixture)
    assert result["coverage"]["scored_points"] == 0
    assert result["quality"]["candidate"]["accuracy"] is None
    assert result["candidate_minus_reference"]["accuracy"] is None
    assert result["label_disagreement"]["points"] == 2


def test_raw_source_count_header_is_independently_checked(fixture):
    row = fixture["sources"][0]
    path = Path(row["path"])
    cloud = laspy.read(path)
    cloud.points = cloud.points[:3]
    cloud.write(path)
    fixture["comparison"]["sources"][0]["sha256"] = sealed.digest(path)
    for name in ("reference", "candidate"):
        fixture["comparison"][name]["provenance"]["source_manifest_sha256"] = (
            sealed.comparison.canonical_hash(fixture["comparison"]["sources"]))
        info = json.loads(Path(row[name]["path"]).read_text())
        info["source_sha256"] = sealed.digest(path)
        row[name] = artifact(Path(row[name]["path"]), info)
    with pytest.raises(ValueError, match="header point count"):
        sealed.evaluate(fixture)


def test_cli_exclusive_report_and_explicit_no_inference(fixture, tmp_path):
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(fixture))
    out = tmp_path / "report.json"
    command = [sys.executable, str(ROOT / "tools/runtime_sealed_outputs.py"),
               "--spec", str(spec_path), "--out", str(out)]
    first = subprocess.run(command, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    original = out.read_bytes()
    assert json.loads(original)["execution"] == "CPU_REEVALUATION_RETAINED_SIDECARS_NO_INFERENCE"
    again = subprocess.run(command, capture_output=True, text=True)
    assert again.returncode != 0 and out.read_bytes() == original


def test_duplicate_json_and_resource_cap(fixture, monkeypatch):
    row = fixture["sources"][0]["candidate"]
    path = Path(row["path"])
    raw = path.read_bytes()
    monkeypatch.setattr(sealed, "MAX_SIDECAR_BYTES", len(raw) - 1)
    with pytest.raises(ValueError, match="byte bound"):
        sealed.evaluate(fixture)
    monkeypatch.setattr(sealed, "MAX_SIDECAR_BYTES", len(raw) * 2)
    path.write_bytes(b'{"schema":"duplicate",' + raw[1:])
    row["sha256"] = sealed.digest(path)
    with pytest.raises(ValueError, match="duplicate JSON key"):
        sealed.evaluate(fixture)
