# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
"""Independent, adversarial CPU evidence for the frozen quality evaluator."""

import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

MODULE = Path(__file__).resolve().parents[1] / "tools/runtime_quality.py"
SPEC = importlib.util.spec_from_file_location("runtime_quality", MODULE)
quality = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(quality)


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def observations(source, count):
    return [
        {
            "source_sha256": source,
            "point_index": i,
            "available_ns": i * 25000,
            "captured_ns": i * 25000 - 100 if i else 0,
            "xyz": [str(-i), "6700000.000000001", "0"],
        }
        for i in range(count)
    ]


@pytest.fixture
def fixture(tmp_path):
    artifacts = {}
    for name in (*quality.BINDINGS, "raw_source"):
        path = tmp_path / name
        path.write_bytes((name + " fixture; no trained inference").encode())
        artifacts[name] = {"path": name}
    source = hashlib.sha256((tmp_path / "raw_source").read_bytes()).hexdigest()
    rows = observations(source, 4)
    for name, data in (
        ("observations", rows),
        (
            "truth",
            [
                {"source_sha256": source, "point_index": i, "label": label}
                for i, label in enumerate([0, 1, 1, -100])
            ],
        ),
    ):
        path = tmp_path / name
        write_rows(path, data)
        artifacts[name] = {"path": name}
    bindings = {name + "_sha256": quality.digest_file(tmp_path / name) for name in quality.BINDINGS}
    for run, labels in (
        ("q0", [0, 1, 1, 0]),
        ("q1_reference", [0, 1, 0, 0]),
        ("q1_native", [0, 1, 0, 1]),
    ):
        windows = quality.build_reference_windows(rows, "q0" if run == "q0" else "q1")
        for window in windows:
            window.update(
                bindings=bindings,
                admitted_ns=window["cutoff_ns"] + 1000,
                output_ns=window["cutoff_ns"] + 3000,
            )
        preds = [
            {"source_sha256": source, "point_index": i, "window_index": 0, "label": label}
            for i, label in enumerate(labels)
        ]
        for kind, data in (("windows", windows), ("predictions", preds)):
            name = f"{run}_{kind}"
            write_rows(tmp_path / name, data)
            artifacts[name] = {"path": name}
    spec = {
        "schema": quality.SCHEMA,
        "purpose": "validation_development",
        "split": "validation",
        "clock": "shared_monotonic_ns",
        "artifacts": artifacts,
        "sources": [
            {"artifact": "raw_source", "points": 4, "scene": "scene-a", "acquisition": "flight-a"}
        ],
        "taxonomy": {
            "classes": [
                {"id": 0, "name": "other"},
                {"id": 1, "name": "ground"},
                {"id": 2, "name": "absent"},
            ],
            "ignored_reference_labels": [-100],
        },
        "policy": {
            "q0": "whole_source_minimum_v1",
            "q1": "independent_minimum_16384_eof_partial_v1",
        },
        "gates": {
            "minimum_label_agreement": 0.995,
            "maximum_mean_iou_loss": 0.01,
            "maximum_supported_class_iou_loss": 0.02,
            "supported_class_points": 1000,
        },
    }
    spec["protocol_sha256"] = quality.freeze_protocol(spec, tmp_path)
    return spec, tmp_path, rows


def evaluate(spec, base):
    manifest = quality.seal_manifest(spec, base)
    return quality.evaluate(manifest, base, manifest["manifest_sha256"], spec["protocol_sha256"])


def mutate_records(base, name, mutate):
    path = base / name
    records = quality._records(path)
    mutate(records)
    write_rows(path, records)


def test_metrics_age_and_gate_scopes(fixture):
    spec, base, rows = fixture
    report = evaluate(spec, base)
    assert report["coverage"] == {
        "original_points": 4,
        "scored_points": 3,
        "ignored_points": 1,
        "missing_points": 0,
    }
    assert report["quality"]["q0"]["accuracy"] == 1
    assert report["quality"]["q1_native"]["accuracy"] == pytest.approx(2 / 3)
    assert report["disagreement_points"] == {"q0_vs_q1_native": 2, "q1_reference_vs_native": 1}
    assert report["q1_native_minus_reference"]["mean_iou_all_classes"] == 0
    assert (
        report["matched_q1_label_quality_gates"]["verdict"] == "FAIL"
    )  # ignored truth still joins/parity
    assert report["q0_q1_policy_quality_verdict"] == "NOT_ASSESSED_REQUIRES_CLUSTER_UNCERTAINTY"
    assert report["qualification_status"] == "NOT_ASSESSED"
    assert report["per_acquisition"]["flight-a"]["q1_native"] == report["quality"]["q1_native"]
    age = report["point_age_q1_native"]
    assert age["availability_accumulation"]["max_ns"] == 75000
    assert age["service"]["max_ns"] == 2000
    assert age["queue"]["max_ns"] == 1000
    assert age["availability_to_output"]["max_ns"] == 78000
    assert age["capture_to_output"]["count"] == 4
    assert report["cutoff_reference_inputs"][0]["input_sha256"] == quality.canonical_hash(rows)


def test_protocol_can_freeze_before_any_prediction_exists(fixture):
    spec, base, _ = fixture
    expected = spec["protocol_sha256"]
    for run in quality.RUNS:
        for kind in ("predictions", "windows"):
            (base / f"{run}_{kind}").unlink()
    assert quality.freeze_protocol(spec, base) == expected
    with pytest.raises(ValueError, match="regular artifact file"):
        quality.seal_manifest(spec, base)


@pytest.mark.parametrize(
    "change",
    [
        lambda s: s.update(split="consumed-test"),
        lambda s: s["gates"].update(minimum_label_agreement=0.1),
        lambda s: s["policy"].update(q1="16384_timeout_flush"),
        lambda s: s["taxonomy"].update(ignored_reference_labels=[-100, 1]),
    ],
)
def test_protocol_change_cannot_reuse_identity(fixture, change):
    spec, base, _ = fixture
    change(spec)
    with pytest.raises(ValueError, match="protocol identity"):
        quality.seal_manifest(spec, base)


@pytest.mark.parametrize(
    "name,mutate,match",
    [
        ("q1_native_predictions", lambda r: r[1].update(point_index=0), "source/index/window"),
        (
            "q1_native_predictions",
            lambda r: r[0].update(source_sha256="f" * 64),
            "source/index/window",
        ),
        ("q1_native_predictions", lambda r: r[0].update(window_index=1), "source/index/window"),
        ("q1_native_predictions", lambda r: r.pop(), "coverage"),
        ("q1_native_predictions", lambda r: r.append(r[0]), "coverage"),
        ("q1_native_predictions", lambda r: r[0].update(revision="final-map"), "revised/map"),
        ("q1_native_predictions", lambda r: r[0].update(label=True), "invalid prediction"),
        ("q1_native_windows", lambda r: r[0].update(cutoff_ns=99999999), "membership/cutoff"),
        (
            "q1_native_windows",
            lambda r: r[0].update(origin=["-999", "0", "0"]),
            "membership/cutoff",
        ),
        ("q1_native_windows", lambda r: r[0].update(input_sha256="f" * 64), "membership/cutoff"),
        (
            "q1_native_windows",
            lambda r: r[0]["bindings"].update(model_sha256="f" * 64),
            "model/preprocessing",
        ),
        ("q1_native_windows", lambda r: r[0].update(output_ns=0), "service clock"),
        ("q1_native_windows", lambda r: r[0].update(begin=False), "discrete window"),
        ("truth", lambda r: r[0].update(point_index=False), "truth source/index"),
    ],
)
def test_resealed_bad_outputs_fail_contract(fixture, name, mutate, match):
    spec, base, _ = fixture
    mutate_records(base, name, mutate)
    if name == "truth":
        spec["protocol_sha256"] = quality.freeze_protocol(spec, base)
    with pytest.raises(ValueError, match=match):
        evaluate(spec, base)


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda r: r[0].update(label=1), "labels forbidden"),
        (lambda r: r[1].update(available_ns=-1), "availability"),
        (lambda r: r[0].update(captured_ns=1), "capture clock"),
        (lambda r: r[0].update(xyz=[1.0, "0", "0"]), "source precision"),
        (lambda r: r[0].update(xyz=["NaN", "0", "0"]), "nonfinite"),
        (lambda r: r.reverse(), "source/index"),
    ],
)
def test_malformed_or_label_leaking_observations(fixture, mutate, match):
    spec, base, _ = fixture
    mutate_records(base, "observations", mutate)
    # Re-freezing an intentionally malformed fixture does not make it valid.
    spec["protocol_sha256"] = quality.freeze_protocol(spec, base)
    with pytest.raises(ValueError, match=match):
        evaluate(spec, base)


def test_artifact_and_external_identity_changes_rejected(fixture):
    spec, base, _ = fixture
    manifest = quality.seal_manifest(spec, base)
    with pytest.raises(ValueError, match="manifest identity"):
        quality.evaluate(manifest, base, "f" * 64, spec["protocol_sha256"])
    with pytest.raises(ValueError, match="protocol identity"):
        quality.evaluate(manifest, base, manifest["manifest_sha256"], "f" * 64)
    mutate_records(base, "q1_native_predictions", lambda r: r[0].update(label=1))
    with pytest.raises(ValueError, match="artifact identity"):
        quality.evaluate(manifest, base, manifest["manifest_sha256"], spec["protocol_sha256"])
    with pytest.raises(ValueError, match="existing artifact identity"):
        quality.seal_manifest({k: v for k, v in manifest.items() if k != "manifest_sha256"}, base)


def test_q1_full_window_eof_minimum_and_low_rate_age():
    rows = observations("a" * 64, 16385)
    quality.validate_observations(rows, [{"sha256": "a" * 64, "points": len(rows)}])
    windows = quality.build_reference_windows(rows, "q1")
    assert [(w["begin"], w["end"]) for w in windows] == [(0, 16384), (16384, 16385)]
    assert windows[0]["origin"] == ["-16383", "6700000.000000001", "0"]
    assert windows[1]["origin"][0] == "-16384"
    assert quality.build_reference_windows(rows, "q0")[0]["origin"][0] == "-16384"
    for window in windows:
        window.update(admitted_ns=window["cutoff_ns"], output_ns=window["cutoff_ns"] + 1000)
    assert quality.point_age(rows, windows)["availability_accumulation"]["max_ns"] == 409575000
    assert quality.point_age(rows, windows)["service"]["max_ns"] == 1000


def test_cutoff_reference_uses_only_available_observations_and_includes_ties():
    rows = observations("a" * 64, 5)
    rows[2]["available_ns"] = rows[1]["available_ns"]
    reference = quality.cutoff_reference(rows, 25000)
    assert reference["observations"] == rows[:3]
    assert reference["whole_available_reference_windows"][0]["origin"][0] == "-2"
    rows[-1]["xyz"] = ["-999999", "0", "0"]
    assert quality.cutoff_reference(rows, 25000) == reference
    assert quality.cutoff_reference(rows, 0)["observations"] == rows[:1]


def test_missing_capture_time_stays_unavailable(fixture):
    _, _, rows = fixture
    for row in rows:
        row["captured_ns"] = None
    windows = quality.build_reference_windows(rows, "q1")
    for window in windows:
        window.update(admitted_ns=window["cutoff_ns"], output_ns=window["cutoff_ns"])
    assert quality.point_age(rows, windows)["capture_to_output"] == {
        "count": 0,
        "mean_ns": None,
        "p50_ns": None,
        "p99_ns": None,
        "max_ns": None,
    }


@pytest.fixture(scope="module")
def prefix():
    return observations("a" * 64, 16384)


def execute_prefix(rows):
    result = []
    for begin in range(0, len(rows), 16384):
        chunk = rows[begin : begin + 16384]
        if len(chunk) < 16384:
            continue
        for row in chunk:
            result.append(
                {
                    "source_sha256": row["source_sha256"],
                    "point_index": row["point_index"],
                    "window_index": begin // 16384,
                    "label": 0,
                    "cutoff_ns": chunk[-1]["available_ns"],
                    "committed_ns": chunk[-1]["available_ns"] + len(rows),
                }
            )
    return result


def future_cases(prefix):
    future = copy.deepcopy(prefix[-1])
    future.update(point_index=16384, available_ns=prefix[-1]["available_ns"] + 25000)
    changed = copy.deepcopy(future)
    changed["xyz"][0] = "-9999999"
    return [[future], [changed]]


def test_executed_prefix_probe_ignores_only_commit_clock(prefix):
    receipt = quality.prefix_causality_probe(execute_prefix, prefix, future_cases(prefix))
    assert receipt["status"] == "SAMPLED_EXECUTION_PREFIX_PASS"
    assert receipt["cases"] == 3 and receipt["prefix_points"] == 16384


@pytest.mark.parametrize(
    "fault,match",
    [
        ("label", "future changed"),
        ("omit", "coverage"),
        ("revise", "duplicate"),
        ("cutoff", "sees future"),
        ("extra", "fields"),
    ],
)
def test_executable_prefix_probe_catches_actual_future_leakage(prefix, fault, match):
    def execute(rows):
        outputs = execute_prefix(rows)
        if len(rows) > len(prefix):
            if fault == "label":
                outputs[0]["label"] = int(rows[-1]["xyz"][0] == "-9999999")
            elif fault == "omit":
                outputs.pop()
            elif fault == "revise":
                outputs.append(outputs[0])
            elif fault == "cutoff":
                outputs[0]["cutoff_ns"] = rows[-1]["available_ns"]
            else:
                outputs[0]["untrusted_wall_field"] = len(rows)
        return outputs

    with pytest.raises(ValueError, match=match):
        quality.prefix_causality_probe(execute, prefix, future_cases(prefix))


def test_prefix_requires_real_changed_future_and_full_window(prefix):
    futures = future_cases(prefix)
    with pytest.raises(ValueError, match="identical"):
        quality.prefix_causality_probe(execute_prefix, prefix, [futures[0], futures[0]])
    with pytest.raises(ValueError, match="complete Q1"):
        quality.prefix_causality_probe(execute_prefix, prefix[:-1], futures)
    futures[0][0]["available_ns"] = 0
    with pytest.raises(ValueError, match="overlaps"):
        quality.prefix_causality_probe(execute_prefix, prefix, futures)


def test_exact_las_lattice_large_translation_negative_and_boundaries():
    integer_xyz = [-1, 20, 40]
    scales = ["0.01"] * 3
    origin = ["0", "0", "0"]
    lattice = quality.local_lattice(integer_xyz, scales, origin, origin, "0.2")
    assert lattice == [-1, 1, 2]
    translated = ["240000", "6700000", "0"]
    assert quality.local_lattice(integer_xyz, scales, translated, translated, "0.2") == lattice
    # Exact arithmetic remains correct beyond Decimal's default 28 significant digits.
    huge = ["10000000000000000000000000000000000000000000000"] * 3
    assert quality.local_lattice(integer_xyz, scales, huge, huge, "0.2") == lattice


def test_cli_exclusive_receipt_and_reproducible_bindings(fixture):
    spec, base, _ = fixture
    manifest = quality.seal_manifest(spec, base)
    path, output = base / "manifest.json", base / "report.json"
    path.write_text(json.dumps(manifest))
    command = [
        sys.executable,
        str(MODULE),
        "evaluate",
        "--manifest",
        str(path),
        "--base",
        str(base),
        "--out",
        str(output),
        "--manifest-sha256",
        manifest["manifest_sha256"],
        "--protocol-sha256",
        spec["protocol_sha256"],
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 1  # actual label gate fail, report retained
    report = json.loads(output.read_text())
    assert report["manifest_sha256"] == manifest["manifest_sha256"]
    previous = output.read_bytes()
    repeated = subprocess.run(command, capture_output=True, text=True)
    assert repeated.returncode != 0 and output.read_bytes() == previous
    assert "FileExistsError" in repeated.stderr


@pytest.mark.parametrize("field", ["available_ns", "captured_ns"])
def test_observation_clock_uint64_bound(field):
    rows = observations("a" * 64, 1)
    rows[0][field] = 1 << 64
    with pytest.raises(ValueError, match="availability|capture"):
        quality.validate_observations(rows, [{"sha256": "a" * 64, "points": 1}])


def test_trace_clock_uint64_bound(fixture):
    spec, base, _ = fixture
    mutate_records(base, "q1_native_windows", lambda r: r[0].update(output_ns=1 << 64))
    with pytest.raises(ValueError, match="service clock"):
        evaluate(spec, base)


def test_probe_rejects_labels_in_future_before_callback_sees_them(prefix):
    futures = future_cases(prefix)
    futures[0][0]["label"] = 1
    calls = []

    def execute(rows):
        calls.append(rows)
        return execute_prefix(rows)

    with pytest.raises(ValueError, match="labels forbidden"):
        quality.prefix_causality_probe(execute, prefix, futures)
    assert calls == []


def test_taxonomy_names_must_be_unique(fixture):
    spec, base, _ = fixture
    spec["taxonomy"]["classes"][1]["name"] = "other"
    spec["protocol_sha256"] = quality.freeze_protocol(spec, base)
    with pytest.raises(ValueError, match="duplicate class"):
        quality.seal_manifest(spec, base)


def test_global_memory_and_matrix_work_limits(fixture, monkeypatch):
    spec, base, _ = fixture
    monkeypatch.setattr(quality, "MAX_PARSED_BYTES", 10)
    with pytest.raises(ValueError, match="parsed byte limit"):
        evaluate(spec, base)
    monkeypatch.setattr(quality, "MAX_MATRIX_CELLS", 1)
    with pytest.raises(ValueError, match="matrix work limit"):
        quality.seal_manifest(spec, base)


def test_escaped_artifact_path_rejected(fixture):
    spec, base, _ = fixture
    spec["artifacts"]["model"]["path"] = "../model"
    with pytest.raises(ValueError, match="relative without"):
        quality.freeze_protocol(spec, base)


def test_all_ignored_matched_quality_is_inconclusive(fixture):
    spec, base, _ = fixture
    mutate_records(base, "truth", lambda rows: [row.update(label=-100) for row in rows])
    (base / "q1_native_predictions").write_bytes((base / "q1_reference_predictions").read_bytes())
    spec["protocol_sha256"] = quality.freeze_protocol(spec, base)
    report = evaluate(spec, base)
    gate = report["matched_q1_label_quality_gates"]
    assert gate["label_agreement_pass"] is True
    assert gate["mean_iou_pass"] is None
    assert gate["verdict"] == "INCONCLUSIVE"
    assert report["coverage"]["scored_points"] == 0


@pytest.mark.parametrize("action", ["freeze", "seal"])
@pytest.mark.parametrize("limit", ["artifact_count", "aggregate_bytes"])
def test_hash_work_is_admitted_before_any_hash_read(fixture, monkeypatch, action, limit):
    spec, base, _ = fixture

    def forbidden_read(*args, **kwargs):
        raise AssertionError("hash read occurred before work admission")

    monkeypatch.setattr(quality, "digest_file", forbidden_read)
    if limit == "artifact_count":
        spec["artifacts"] = {str(i): {"path": "model"} for i in range(1000)}
        message = "bounded artifact map"
    else:
        monkeypatch.setattr(quality, "MAX_TOTAL_ARTIFACT_BYTES", 1)
        message = "aggregate artifact byte limit"
    with pytest.raises(ValueError, match=message):
        if action == "freeze":
            quality.freeze_protocol(spec, base)
        else:
            quality.seal_manifest(spec, base)
