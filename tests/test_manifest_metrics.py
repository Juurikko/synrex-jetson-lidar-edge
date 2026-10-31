# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
"""Measurement populations and finite accumulation through actual CPU TCP peers."""

import pytest
from test_manifest_live import pipeline


def check_histogram(value, count):
    assert value["schema"] == "synrex.replay.latency_histogram.v1"
    assert value["count"] == count
    assert value["storage_bytes"] <= 32 * 1024
    if not count:
        assert value["maximum_ns"] is None
        assert value["sum_ns"] == 0
        assert all(value[f"p{percent}_ns"] is None for percent in (50, 95, 99))
        return
    assert 0 <= value["maximum_ns"] <= value["sum_ns"]
    previous = 0
    for percent in (50, 95, 99):
        bound = value[f"p{percent}_ns"]
        assert previous <= bound["lower_ns"] <= bound["upper_ns"]
        assert bound["lower_ns"] <= value["maximum_ns"]
        assert bound["upper_ns"] - bound["lower_ns"] <= bound["lower_ns"] // 64
        previous = bound["lower_ns"]


@pytest.mark.parametrize("wire", ["binary-v3", "binary-source-v1"])
def test_manifest_timing_counts_sources_and_clock_boundaries(tmp_path, wire):
    results, _ = pipeline(tmp_path, wire=wire)
    for mode, result in results.items():
        receipt = result["receipt"]
        assert result["code"] == 0, results
        assert receipt["physical_capture_age_seconds"] is None
        assert "per source only" in receipt["timing_population"]
        assert receipt["instrumentation_overhead"] == "NOT_MEASURED_FOR_ACTUAL_PIPELINE"
        assert receipt["process_peak_rss_bytes"] > 0
        assert "whole process lifetime" in receipt["process_peak_rss_scope"]
        assert receipt["descriptor"]["binary_wire_schema"] == (
            "synrex.binary.source.v1" if wire == "binary-source-v1" else "synrex.binary.v3"
        )
        for row in receipt["sources"]:
            points, frames = row["expected_records"], row["frames"]
            if mode == "send":
                for name in ("decode_window", "assembly_and_validation", "frame_send",
                             "sender_scheduling_lag"):
                    check_histogram(row["timings"][name], frames)
                check_histogram(row["timings"]["synthetic_accumulation"], points)
                assert row["maximum_sender_scheduling_lag_seconds"] == (
                    row["timings"]["sender_scheduling_lag"]["maximum_ns"] / 1e9
                )
            elif mode == "receive":
                check_histogram(row["timings"]["frame_ingress"], frames)
                service = row["timings"]["receiver_service"]
                check_histogram(service, points)
                assert "excludes accumulation" in service["boundary"]
            else:
                for name in ("classification_receive_and_map", "source_attribute_validation"):
                    check_histogram(row["timings"][name], frames)


def test_manifest_accumulation_includes_partial_window_and_each_point(tmp_path):
    results, _ = pipeline(tmp_path, rate=80, sources=1, wire="binary-source-v1")
    assert all(result["code"] == 0 for result in results.values()), results
    sender = results["send"]["receipt"]
    accumulation = sender["sources"][0]["timings"]["synthetic_accumulation"]
    check_histogram(accumulation, 23)
    # 8+8+7 source observations at 80/s; last point also waits one period.
    expected_periods = 2 * (8 * 9 // 2) + 7 * 8 // 2
    assert accumulation["sum_ns"] == expected_periods * 12500000
    assert accumulation["maximum_ns"] == 100000000
    assert sender["offered_finite_duration_seconds"] == 23 / 80
    assert sender["producer_elapsed_seconds"] >= 23 / 80


def test_manifest_failure_retains_empty_service_population_and_failed_prefix(tmp_path):
    results, _ = pipeline(tmp_path, failure="inference", sources=1)
    receiver = results["receive"]
    assert receiver["code"] == 1
    row = receiver["receipt"]["sources"][0]
    assert row["accepted_records"] == 8
    assert row["freshly_classified_records"] == row["emitted_records"] == 0
    check_histogram(row["timings"]["frame_ingress"], 1)
    check_histogram(row["timings"]["receiver_service"], 0)
    assert row["status"] == "FAIL"
