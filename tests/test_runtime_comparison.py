# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
"""Adversarial CPU fixtures for source identity, metrics and comparison contracts."""

import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "tools/runtime_comparison.py"
SPEC = importlib.util.spec_from_file_location("runtime_comparison", MODULE)
comparison = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(comparison)


def fixture_spec():
    taxonomy = {
        "classes": [{"id": 0, "name": "other"}, {"id": 7, "name": "ground"},
                    {"id": 255, "name": "absent"}],
        "ignored_reference_labels": [-100], "ground_class_id": 7,
    }
    sources = [{"sha256": "a" * 64, "points": 4}, {"sha256": "b" * 64, "points": 2}]
    run = {
        "name": "CPU fixture, no inference", "execution_mode": "REFERENCE",
        "execution_scope": "whole_tile", "limitations": ["Synthetic labels only"],
        "unsupported_modes": {"HYBRID": "not exercised"},
        "provenance": {
            "source_sha256": "c" * 64, "config_sha256": "d" * 64,
            "model_sha256": "e" * 64, "engine_sha256": None,
            "source_manifest_sha256": comparison.canonical_hash(sources),
            "scoring_taxonomy_sha256": comparison.canonical_hash(taxonomy),
            "package_taxonomy_sha256": None,
            "package_taxonomy_unavailable_reason": "CPU fixture has no admitted package taxonomy",
            "hardware": "CPU fixture", "software": "Python stdlib fixture",
        },
        "semantics": {
            "input_fields": ["x", "y", "z"], "preprocessing": {"version": "fixture-v1"},
            "windows": {"points": 4, "eof_flush": True},
            "reconstruction": {"version": "source-index-v1"}, "precision": "FP32",
            "policy": "fixture Q0", "architecture": "fixture-only",
            "input_provenance": "fixture geometry only",
        },
        "measurements": {
            "timing_boundary": "unmeasured fixture; no production timing",
            **dict.fromkeys(("offered_points_per_second", "achieved_points_per_second",
                             "p50_latency_ms", "p95_latency_ms", "p99_latency_ms",
                             "max_latency_ms", "peak_memory_bytes", "energy_joules")),
        },
    }
    result = {
        "schema": comparison.SCHEMA, "track": "runtime_parity", "taxonomy": taxonomy,
        "sources": sources, "evaluation": {
            "split": "synthetic-fixture", "protocol": "CPU contract tests only",
            "latency_budget_ms": None,
        }, "minimum_label_agreement": 0.995,
        "reference": copy.deepcopy(run), "candidate": copy.deepcopy(run),
    }
    result["candidate"]["execution_mode"] = "NATIVE"
    return result


def records(spec, labels):
    identities = [(source["sha256"], i) for source in spec["sources"]
                  for i in range(source["points"])]
    return [{"source_sha256": sha, "point_index": i, "label": label}
            for (sha, i), label in zip(identities, labels, strict=True)]


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


@pytest.fixture
def inputs(tmp_path):
    spec = fixture_spec()
    paths = [tmp_path / f"{name}.jsonl" for name in ("truth", "reference", "candidate")]
    for path, labels in zip(paths, ([0, 7, 7, -100, 0, 7], [0, 7, 0, 7, 0, 7],
                                   [7, 7, 0, 0, 0, 7]), strict=True):
        write_rows(path, records(spec, labels))
    return spec, paths


def test_hand_computed_quality_parity_and_semantic_proxy(inputs):
    spec, paths = inputs
    report = comparison.compare(spec, *paths)
    assert report["coverage"] == {
        "original_points": 6, "reference_points": 6, "candidate_points": 6,
        "scored_points": 5, "ignored_points": 1, "missing_points": 0,
    }
    reference, candidate = (report["quality"][name] for name in ("reference", "candidate"))
    assert reference["confusion"] == [[2, 0, 0], [1, 2, 0], [0, 0, 0]]
    assert candidate["confusion"] == [[1, 1, 0], [1, 2, 0], [0, 0, 0]]
    assert reference["accuracy"] == 0.8
    assert reference["mean_iou_all_classes"] == pytest.approx(4 / 9)
    assert reference["mean_iou_present"] == pytest.approx(2 / 3)
    assert reference["macro_f1_all_classes"] == pytest.approx(1.6 / 3)
    assert reference["classes"][2] == {
        "id": 255, "name": "absent", "support": 0, "predicted": 0,
        "iou": None, "precision": None, "recall": None, "f1": None,
    }
    assert report["candidate_minus_reference"]["accuracy"] == pytest.approx(-0.2)
    assert report["label_disagreement"]["points"] == 2  # ignored truth still covered
    assert report["numerical_parity"]["label_agreement_verdict"] == "FAIL"
    assert report["numerical_parity"]["package_taxonomy_binding"] == "UNAVAILABLE"
    assert report["numerical_parity"]["package_taxonomy_qualification"] == "NOT_ASSESSED"
    assert report["qualification_status"] == "NOT_ASSESSED"
    proxy = report["ground_proxy"]["candidate"]
    assert proxy["precision"] == proxy["recall"] == pytest.approx(2 / 3)
    assert proxy["false_ground_points"] == 1
    assert proxy["false_ground_contamination"] == pytest.approx(1 / 3)
    assert proxy["survey_reference_available"] is False
    assert proxy["dtm_rmse"] is None and proxy["dtm_status"] == "BLOCKED_INPUT"
    assert report["input_record_sha256"] == {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in zip(("truth", "reference", "candidate"), paths, strict=True)
    }
    assert report["spec"]["sources"] == spec["sources"]
    assert report["spec"]["candidate"]["measurements"]["energy_joules"] is None
    json.dumps(report, allow_nan=False)


def test_nonidentical_policies_have_quality_results_without_parity_verdict(inputs):
    spec, paths = inputs
    spec["track"] = "model_policy_quality"
    del spec["minimum_label_agreement"]
    spec["candidate"]["semantics"]["windows"]["points"] = 2
    spec["candidate"]["provenance"]["model_sha256"] = "f" * 64
    result = comparison.compare(spec, *paths)
    assert result["numerical_parity"] is None
    assert result["quality"]["candidate"]["accuracy"] == 0.6


@pytest.mark.parametrize("stream", [0, 1, 2])
@pytest.mark.parametrize("mutation", ["missing", "duplicate", "extra", "reorder", "source"])
def test_identity_mismatches_fail_in_every_stream(inputs, stream, mutation):
    spec, paths = inputs
    rows = [json.loads(line) for line in paths[stream].read_text().splitlines()]
    if mutation == "missing":
        del rows[3]  # ignored ground truth points are not optional
    elif mutation == "duplicate":
        rows[2] = rows[1]
    elif mutation == "extra":
        rows.append(rows[-1])
    elif mutation == "reorder":
        rows[0], rows[1] = rows[1], rows[0]
    else:
        rows[0]["source_sha256"] = "f" * 64
    write_rows(paths[stream], rows)
    with pytest.raises(ValueError, match="missing|duplicate|source/index"):
        comparison.compare(spec, *paths)


@pytest.mark.parametrize("stream,label", [(0, 123), (1, -100), (2, 123), (2, True), (1, 0.0)])
def test_unknown_ignored_prediction_or_noninteger_label_fails(inputs, stream, label):
    spec, paths = inputs
    rows = [json.loads(line) for line in paths[stream].read_text().splitlines()]
    rows[0]["label"] = label
    write_rows(paths[stream], rows)
    with pytest.raises(ValueError, match="taxonomy|invalid label"):
        comparison.compare(spec, *paths)


@pytest.mark.parametrize("field", ["preprocessing", "windows", "reconstruction", "precision",
                                   "input_fields", "policy"])
def test_parity_rejects_different_declared_semantics(inputs, field):
    spec, paths = inputs
    value = spec["candidate"]["semantics"][field]
    spec["candidate"]["semantics"][field] = (
        {"changed": True} if isinstance(value, dict) else ["x"]
        if isinstance(value, list) else "changed"
    )
    with pytest.raises(ValueError, match="parity semantics"):
        comparison.compare(spec, *paths)


def test_parity_declarations_use_json_identity_including_number_types(inputs):
    spec, paths = inputs
    spec["candidate"]["semantics"]["windows"]["points"] = 4.0
    with pytest.raises(ValueError, match="parity semantics"):
        comparison.compare(spec, *paths)


@pytest.mark.parametrize("field", ["model_sha256", "scoring_taxonomy_sha256",
                                   "source_manifest_sha256"])
def test_identity_metadata_mismatch_fails(inputs, field):
    spec, paths = inputs
    spec["candidate"]["provenance"][field] = "f" * 64
    with pytest.raises(ValueError, match="mismatch"):
        comparison.compare(spec, *paths)


def test_matched_declared_package_hashes_have_separate_binding(inputs):
    spec, paths = inputs
    for name in ("reference", "candidate"):
        spec[name]["provenance"].update(package_taxonomy_sha256="f" * 64,
                                       package_taxonomy_unavailable_reason=None)
    result = comparison.compare(spec, *paths)
    parity = result["numerical_parity"]
    assert parity["package_taxonomy_binding"] == "MATCHED_DECLARED_ARTIFACT_HASH"
    assert parity["package_taxonomy_qualification"] == "NOT_ASSESSED"


@pytest.mark.parametrize("mutation", ["missing_hash", "missing_reason", "empty_reason",
                                     "invalid_hash", "present_reason", "different_hash"])
def test_package_taxonomy_identity_availability_and_parity(inputs, mutation):
    spec, paths = inputs
    provenance = spec["candidate"]["provenance"]
    if mutation == "missing_hash":
        del provenance["package_taxonomy_sha256"]
    elif mutation == "missing_reason":
        del provenance["package_taxonomy_unavailable_reason"]
    elif mutation == "empty_reason":
        provenance["package_taxonomy_unavailable_reason"] = " "
    elif mutation == "invalid_hash":
        provenance["package_taxonomy_sha256"] = "not-a-hash"
    elif mutation == "present_reason":
        provenance["package_taxonomy_sha256"] = "f" * 64
    else:
        provenance.update(package_taxonomy_sha256="f" * 64,
                          package_taxonomy_unavailable_reason=None)
    with pytest.raises(ValueError):
        comparison.compare(spec, *paths)


def test_policy_quality_accepts_differing_declared_package_artifacts(inputs):
    spec, paths = inputs
    spec["track"] = "model_policy_quality"
    del spec["minimum_label_agreement"]
    spec["candidate"]["provenance"].update(package_taxonomy_sha256="f" * 64,
                                          package_taxonomy_unavailable_reason=None)
    assert comparison.compare(spec, *paths)["numerical_parity"] is None


def test_explicit_preregistered_threshold_controls_label_verdict(inputs):
    spec, paths = inputs
    spec["minimum_label_agreement"] = 2 / 3
    assert comparison.compare(spec, *paths)["numerical_parity"]["label_agreement_verdict"] == "PASS"
    del spec["minimum_label_agreement"]
    with pytest.raises(ValueError, match="preregister"):
        comparison.compare(spec, *paths)


def test_no_scored_points_returns_null_metrics_with_full_coverage(inputs):
    spec, paths = inputs
    write_rows(paths[0], records(spec, [-100] * 6))
    report = comparison.compare(spec, *paths)
    assert report["coverage"]["ignored_points"] == 6
    assert report["quality"]["candidate"]["accuracy"] is None
    assert report["quality"]["candidate"]["mean_iou_all_classes"] is None
    assert report["candidate_minus_reference"]["accuracy"] is None
    assert report["ground_proxy"]["reference"]["recall"] is None


def test_absent_truth_but_false_prediction_penalizes_iou(inputs):
    spec, paths = inputs
    rows = records(spec, [255, 7, 0, 0, 0, 7])
    write_rows(paths[2], rows)
    result = comparison.compare(spec, *paths)["quality"]["candidate"]["classes"][2]
    assert result["support"] == 0 and result["predicted"] == 1
    assert result["iou"] == result["precision"] == result["f1"] == 0
    assert result["recall"] is None


@pytest.mark.parametrize("raw", [b'{"label":0,"label":1}', b'{"label":NaN}',
                                 b'{"label":1e999}', b'\xff', b'[]', b'\n',
                                 '{"label":0}'.encode("utf-16")])
def test_malformed_records_fail(inputs, raw):
    spec, paths = inputs
    paths[1].write_bytes(raw + b"\n")
    with pytest.raises(ValueError):
        comparison.compare(spec, *paths)


def test_record_and_spec_read_limits(inputs, tmp_path):
    spec, paths = inputs
    paths[1].write_bytes(b" " * (comparison.MAX_RECORD_BYTES + 1))
    with pytest.raises(ValueError, match="record exceeds"):
        comparison.compare(spec, *paths)
    large = tmp_path / "large.json"
    large.write_bytes(b" " * (comparison.MAX_JSON_BYTES + 1))
    with pytest.raises(ValueError, match="spec exceeds"):
        comparison.load_spec(large)


@pytest.mark.parametrize("raw", ['{"schema":0,"schema":1}', '{"threshold":NaN}',
                                 '{"threshold":1e999}'])
def test_strict_spec_loader_rejects_ambiguous_json(tmp_path, raw):
    path = tmp_path / "spec.json"
    path.write_text(raw)
    with pytest.raises(ValueError):
        comparison.load_spec(path)


@pytest.mark.parametrize("mutation", ["oversize", "class256", "duplicate_class", "ignored_overlap",
                                     "duplicate_source", "negative_count", "bool_count",
                                     "missing_availability", "nonfinite", "unsupported"])
def test_invalid_contracts_rejected(inputs, mutation):
    spec, paths = inputs
    if mutation == "oversize":
        spec["taxonomy"]["classes"] *= 100
    elif mutation == "class256":
        spec["taxonomy"]["classes"][0]["id"] = 256
    elif mutation == "duplicate_class":
        spec["taxonomy"]["classes"][0]["id"] = 7
    elif mutation == "ignored_overlap":
        spec["taxonomy"]["ignored_reference_labels"] = [7]
    elif mutation == "duplicate_source":
        spec["sources"].append(spec["sources"][0])
    elif mutation in {"negative_count", "bool_count"}:
        spec["sources"][0]["points"] = -1 if mutation == "negative_count" else True
    elif mutation == "missing_availability":
        del spec["reference"]["measurements"]["energy_joules"]
    elif mutation == "nonfinite":
        spec["reference"]["measurements"]["energy_joules"] = float("nan")
    else:
        spec["reference"]["unsupported_modes"]["REFERENCE"] = "unavailable"
    with pytest.raises(ValueError):
        comparison.compare(spec, *paths)


def test_streaming_does_not_read_full_point_files(inputs, monkeypatch):
    spec, paths = inputs
    original = Path.open

    class Bounded:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.stream.close()

        def readline(self, limit):
            assert limit == comparison.MAX_RECORD_BYTES + 1
            return self.stream.readline(limit)

    def bounded(path, *args, **kwargs):
        stream = original(path, *args, **kwargs)
        return Bounded(stream) if path in paths else stream

    monkeypatch.setattr(Path, "open", bounded)
    assert comparison.compare(spec, *paths)["coverage"]["original_points"] == 6


def test_cli_retains_failed_parity_receipt_and_prevents_overwrite(inputs, tmp_path):
    spec, paths = inputs
    spec_path, output = tmp_path / "spec.json", tmp_path / "receipt.json"
    spec_path.write_text(json.dumps(spec))
    command = [sys.executable, str(MODULE), "--spec", str(spec_path), "--truth", str(paths[0]),
               "--reference", str(paths[1]), "--candidate", str(paths[2]), "--out", str(output)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=15)
    assert result.returncode == 1, result.stderr
    assert json.loads(output.read_text())["numerical_parity"]["label_agreement_verdict"] == "FAIL"
    saved = output.read_bytes()
    result = subprocess.run(command, capture_output=True, text=True, timeout=15)
    assert result.returncode != 0 and "FileExistsError" in result.stderr
    assert output.read_bytes() == saved


def test_cli_rejects_input_alias(inputs, tmp_path):
    spec, paths = inputs
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec))
    saved = paths[0].read_bytes()
    result = subprocess.run([
        sys.executable, str(MODULE), "--spec", str(spec_path), "--truth", str(paths[0]),
        "--reference", str(paths[1]), "--candidate", str(paths[2]), "--out", str(paths[0]),
    ], capture_output=True, text=True, timeout=15)
    assert result.returncode != 0 and "output aliases" in result.stderr
    assert paths[0].read_bytes() == saved
