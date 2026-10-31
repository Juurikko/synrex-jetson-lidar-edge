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
        assert "trial_timings merges" in receipt["timing_population"]
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
        check_trial_histograms(receipt)


def check_trial_histograms(receipt):
    expected_names = set().union(*(row.get("timings", {}) for row in receipt["sources"]))
    assert set(receipt["trial_timings"]) == expected_names
    for name, trial in receipt["trial_timings"].items():
        populations = [row["timings"][name] for row in receipt["sources"]
                       if name in row.get("timings", {})]
        check_histogram(trial, sum(value["count"] for value in populations))
        assert trial["sum_ns"] == sum(value["sum_ns"] for value in populations)
        maxima = [value["maximum_ns"] for value in populations if value["count"]]
        assert trial["maximum_ns"] == (max(maxima) if maxima else None)
        assert all(value["population"] == trial["population"] and
                   value["boundary"] == trial["boundary"] for value in populations)


@pytest.mark.parametrize("wire", ["binary-v3", "binary-source-v1"])
def test_manifest_saturated_bounded_queue_drains_without_claiming_capacity(tmp_path, wire):
    results, _ = pipeline(tmp_path, rate=80, sources=1, wire=wire, queue_frames=1)
    assert all(result["code"] == 0 for result in results.values()), results
    sender = results["send"]["receipt"]
    queue = sender["sources"][0]["decode_queue"]
    assert queue["capacity"] == queue["high_water_items"] == 1
    assert queue["producer_waits"] > 0 and queue["producer_wait_ns"] > 0
    assert queue["high_water_allocation_bytes"] <= queue["byte_limit"]
    assert queue["queued_items"] == queue["queued_allocation_bytes"] == 0
    assert queue["discarded_items"] == queue["discarded_points"] == 0
    assert queue["closed"] and not queue["failed"]
    assert sender["producer_elapsed_seconds"] >= 23 / 80
    for result in results.values():
        assert result["receipt"]["capacity_qualification"] == "NOT_RUN"
        check_trial_histograms(result["receipt"])


@pytest.mark.parametrize("wire", ["binary-v3", "binary-source-v1"])
@pytest.mark.parametrize("failure", ["inference", "late-decode", "output-map",
                                    "missing-subscriber"])
def test_manifest_trial_failure_preserves_elapsed_and_measured_populations(tmp_path, wire, failure):
    results, _ = pipeline(tmp_path, wire=wire, failure=failure,
                          sources=1 if failure == "output-map" else 2)
    for mode, result in results.items():
        receipt = result["receipt"]
        import json
        assert json.loads((tmp_path / f"{mode}.json").read_text()) == receipt
        assert result["code"] == (0 if mode == "send" and failure == "output-map" else 1)
        assert 0 < receipt["execution_seconds"] <= receipt["elapsed_seconds"]
        assert receipt["capacity_qualification"] == "NOT_RUN"
        check_trial_histograms(receipt)
        if mode == "send" and "producer_elapsed_seconds" in receipt:
            assert receipt["actual_sender_rate_pps"] == pytest.approx(
                receipt["emitted_records"] / receipt["producer_elapsed_seconds"])
            assert "including failed prefixes" in receipt["actual_sender_rate_scope"]
            assert receipt["trial_timings"]["frame_send"]["count"] == sum(
                row["frames"] for row in receipt["sources"])
        if failure == "late-decode":
            assert [row["status"] for row in receipt["sources"]] == ["PASS", "FAIL"]
            assert receipt["trial_timings"]


@pytest.mark.parametrize("queue_frames", [1, 2])
def test_manifest_cancellation_keeps_failed_prefix_and_wakes_full_queue(tmp_path, queue_frames):
    results, _ = pipeline(tmp_path, failure="cancel", sources=1, rate=80,
                          wire="binary-source-v1", queue_frames=queue_frames)
    for result in results.values():
        receipt = result["receipt"]
        assert result["code"] == 1 and receipt["status"] == "FAIL"
        assert receipt["capacity_qualification"] == "NOT_RUN"
        assert result["backend"]["live_after_return"] == 0
        assert 0 < receipt["execution_seconds"] <= receipt["elapsed_seconds"]
        check_trial_histograms(receipt)
    sender = results["send"]["receipt"]
    row = sender["sources"][0]
    queue = row["decode_queue"]
    assert queue["high_water_items"] == queue_frames
    assert queue["high_water_allocation_bytes"] <= queue["byte_limit"]
    assert queue["failed"] and queue["closed"] and not queue["producer_waiting"]
    assert queue["queued_items"] == queue["queued_allocation_bytes"] == 0
    assert queue["producer_waits"] > 0 and queue["discarded_points"] > 0
    assert 0 < row["emitted_records"] < row["expected_records"]
    assert sender["declared_records_not_emitted"] == 1001 - row["emitted_records"]
    assert row["frames"] == row["timings"]["frame_send"]["count"]
    assert row["decode_and_dispatch_seconds"] > 0
    assert row["maximum_sender_scheduling_lag_seconds"] is not None
    assert sender["actual_sender_rate_pps"] == pytest.approx(
        row["emitted_records"] / sender["producer_elapsed_seconds"])


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
