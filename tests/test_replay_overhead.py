# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
"""Paired actual CLI paths with opaque CPU classes; no GPU/timing-target claim."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
from test_manifest_live import FIXTURE, pipeline

spec = importlib.util.spec_from_file_location(
    "replay_overhead", Path(__file__).parents[1] / "tools/replay_overhead.py"
)
overhead = importlib.util.module_from_spec(spec)
spec.loader.exec_module(overhead)


@pytest.mark.parametrize("classes,wire", [(4, "binary-v3"), (7, "binary-source-v1")])
def test_paired_actual_pipeline_optional_instrumentation(tmp_path, classes, wire):
    plan = {
        "schema": "synrex.replay.overhead_pairs.v1",
        "scope": "opaque_cpu_fixture",
        "executable": {
            "path": str(FIXTURE),
            "sha256": hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
        },
        "pairs": [],
    }
    for index in range(3):
        order = ["disabled", "enabled"] if index % 2 == 0 else ["enabled", "disabled"]
        pair = {"order": order}
        for setting in order:
            out = tmp_path / f"pair-{index}-{setting}"
            out.mkdir()
            result, _ = pipeline(
                tmp_path,
                count=classes,
                wire=wire,
                timing=setting,
                reuse_inputs=bool(index or setting == "enabled"),
                receipt_dir=out,
            )
            assert all(row["code"] == 0 for row in result.values()), result
            pair[setting] = {mode: str(out / f"{mode}.json") for mode in overhead.MODES}
            for mode, row in result.items():
                receipt = row["receipt"]
                assert receipt["process_user_cpu_seconds"] >= 0
                assert receipt["process_system_cpu_seconds"] >= 0
                assert "whole process lifetime" in receipt["process_cpu_scope"]
                if setting == "disabled":
                    assert receipt["trial_timings"] == {}
                    assert all(source["timings"] is None for source in receipt["sources"])
                    if mode == "send":
                        assert all(
                            source["maximum_sender_scheduling_lag_seconds"] is None
                            for source in receipt["sources"]
                        )
        plan["pairs"].append(pair)
    (tmp_path / "pairs.json").write_text(json.dumps(plan, indent=2) + "\n")
    result = overhead.compare(plan, tmp_path)
    assert result["status"] == "PAIRED_OUTPUTS_IDENTICAL"
    assert result["actual_gpu_instrumentation_qualification"] == "NOT_ESTABLISHED_BY_THIS_TOOL"
    assert all(
        sample["modes"]["receive"]["disabled"]["tail_histograms"] is None
        for sample in result["pairs"]
    )
    (tmp_path / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    # Adversarial edits never turn an incomplete/mismatched trial into a result.
    path = Path(plan["pairs"][0]["enabled"]["receive"])
    original = path.read_bytes()
    for mutation in (
        "status",
        "output",
        "identity",
        "config",
        "backend_profiling",
        "tail",
        "resource",
        "fractional_rss",
        "peer",
    ):
        value = json.loads(original)
        if mutation == "status":
            value["status"] = "FAIL"
        elif mutation == "output":
            value["sources"][0]["output_completion"]["snapshot_sha256"] = "0" * 64
        elif mutation == "identity":
            value["descriptor"]["context_policy_sha256"] = "0" * 64
        elif mutation == "config":
            value["replay_configuration"]["queue_frames"] += 1
        elif mutation == "backend_profiling":
            value["backend_profiling"] = "enabled"
        elif mutation == "tail":
            value["trial_timings"] = {}
        elif mutation == "fractional_rss":
            value["process_peak_rss_bytes"] = 0.5
        elif mutation == "resource":
            value["process_user_cpu_seconds"] = None
        else:
            value["sources"][0]["input_completion"]["input_sha256"] = "0" * 64
        path.write_text(json.dumps(value))
        with pytest.raises(ValueError):
            overhead.compare(plan, tmp_path)
    # Validate raw evidence itself, even when all paired copies might agree on a bad value.
    for field in ("output_completion", "input_completion"):
        value = json.loads(original)
        value["sources"][0][field]["status"] = "FAILED"
        with pytest.raises(ValueError, match="completion fence"):
            overhead.validate_counts(value)
    value = json.loads(original)
    value["freshly_classified_records"] = 0
    with pytest.raises(ValueError, match="trial counters"):
        overhead.validate_counts(value)
    value = json.loads(original)
    value["trial_timings"]["receiver_service"]["count"] = -1
    with pytest.raises(ValueError, match="count/sum"):
        overhead.validate_timings(value)
    path.write_bytes(original)
    implicit = json.loads(original)
    explicit = copy.deepcopy(implicit)
    explicit["backend_profiling"] = "disabled"
    assert overhead.semantic_identity(implicit) == overhead.semantic_identity(explicit)
    explicit["backend_profiling"] = "enabled"
    assert overhead.semantic_identity(implicit) != overhead.semantic_identity(explicit)
    explicit["backend_profiling"] = None
    with pytest.raises(ValueError, match="backend profiling setting"):
        overhead.semantic_identity(explicit)
    subscriber_paths = [
        Path(pair[setting]["subscribe"])
        for pair in plan["pairs"]
        for setting in ("enabled", "disabled")
    ]
    originals = {path: path.read_bytes() for path in subscriber_paths}
    try:
        for path, raw in originals.items():
            value = json.loads(raw)
            value["sources"][0]["source_sha256"] = "0" * 64
            value["sources"][0]["coverage"]["source_sha256"] = "0" * 64
            path.write_text(json.dumps(value))
        with pytest.raises(ValueError, match="peer source identity"):
            overhead.compare(plan, tmp_path)
    finally:
        for path, raw in originals.items():
            path.write_bytes(raw)
    duplicate = copy.deepcopy(plan)
    duplicate["pairs"][1] = duplicate["pairs"][0]
    with pytest.raises(ValueError):
        overhead.compare(duplicate, tmp_path)
    changed_binary = copy.deepcopy(plan)
    changed_binary["executable"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="executable identity"):
        overhead.compare(changed_binary, tmp_path)


@pytest.mark.parametrize("failure", ["inference", "late-decode"])
def test_disabled_instrumentation_failure_retains_truthful_prefix(tmp_path, failure):
    results, _ = pipeline(tmp_path, failure=failure, timing="disabled")
    for row in results.values():
        receipt = row["receipt"]
        assert row["code"] == 1
        assert receipt["status"] == "FAIL"
        assert receipt["timing_availability"] == "UNAVAILABLE_INSTRUMENTATION_DISABLED"
        assert receipt["trial_timings"] == {}
        assert receipt["execution_seconds"] > 0
        assert all(source.get("timings") is None for source in receipt["sources"])
        if failure == "late-decode":
            assert [source["status"] for source in receipt["sources"]] == ["PASS", "FAIL"]


def test_enabled_instrumentation_without_samples_is_unavailable(tmp_path):
    results, _ = pipeline(tmp_path, failure="missing-subscriber")
    for row in results.values():
        assert row["code"] == 1
        assert row["receipt"]["timing_instrumentation"] == "enabled"
        assert row["receipt"]["timing_availability"] == "UNAVAILABLE_NO_SAMPLES"
        assert row["receipt"]["trial_timings"] == {}


def test_overhead_json_reads_are_strict_and_bounded(tmp_path):
    path = tmp_path / "raw.json"
    for text in (
        '{"status":"PASS","status":"FAIL"}',
        '{"value":NaN}',
        '{"value":Infinity}',
        '{"value":-Infinity}',
    ):
        path.write_text(text)
        with pytest.raises(ValueError):
            overhead.read_json(path)
    path.write_text('{"value":"too many bytes"}')
    with pytest.raises(ValueError, match="byte budget"):
        overhead.read_json(path, maximum=8)


def test_histogram_quantile_requires_possible_rank_sum():
    value = {
        "schema": "synrex.replay.latency_histogram.v1",
        "count": 100,
        "sum_ns": 1,
        "maximum_ns": 1,
        "storage_bytes": 30232,
        **{f"p{p}_ns": {"lower_ns": 1, "upper_ns": 1} for p in (50, 95, 99)},
    }
    with pytest.raises(ValueError, match="rank inconsistent"):
        overhead.validate_histogram(value, 100)
