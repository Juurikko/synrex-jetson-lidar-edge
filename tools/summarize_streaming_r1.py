# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
"""Summarize external transport receipts without promoting failed or short trials."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def summarize(paths, destination):
    destination.mkdir(exist_ok=False)
    rows = []
    groups = defaultdict(list)
    group_fields = (
        "binary_sha256",
        "source_sha256",
        "requested_points_s",
        "map_capacity",
        "churn_mm",
        "requested_seconds",
        "validation_mode",
        "async_export",
        "model_manifest_sha256",
    )

    def identity(row):
        return tuple(row[k] for k in group_fields)

    for path in paths:
        report = json.loads(path.read_text())
        for index, trial in enumerate(report["trials"]):
            native = trial.get("native", {})
            receiver, consumer, sender = (
                native.get(k, {}) for k in ("receive", "subscribe", "send")
            )
            service = receiver.get("service", {})
            delivery = receiver.get("output_delivery", {})
            latency = consumer.get(
                "source_to_subscriber",
                consumer.get("validated_prefix", {}).get("source_to_subscriber", {}),
            )
            samples = trial.get("device_samples", [])
            stages = receiver.get("measured_stage_seconds", {})
            ops = receiver.get("measured_map_operations", {})
            seconds = receiver.get("measured_seconds", 0)
            row = dict(
                campaign=path.name,
                trial=index,
                binary_sha256=trial.get("binary_sha256"),
                source_sha256=trial["source_sha256"],
                model_manifest_sha256=service.get("session", {}).get("manifest_sha256"),
                status=trial["status"],
                requested_points_s=trial["rate"],
                requested_seconds=trial["seconds"],
                validation_mode=consumer.get("validation_mode", "reference"),
                async_export="async_laz_export" in consumer,
                measured_seconds=seconds,
                freshly_classified_points_s=receiver.get("freshly_classified_points_per_second", 0),
                measured_fresh_points=receiver.get("freshly_classified_measured_points", 0),
                map_updates_s=receiver.get("map_updates_per_second", 0),
                insertions_s=ops.get("insertions", 0) / seconds if seconds else 0,
                evictions_s=ops.get("removals", 0) / seconds if seconds else 0,
                map_capacity=trial["cells"],
                churn_mm=trial["churn_mm"],
                complete_coverage=consumer.get("coverage", 0),
                classification_drops=service.get("dropped_points", 0),
                failed_point_deliveries=delivery.get("failed_point_deliveries", 0),
                p50_ms=latency.get("p50_seconds_upper_bound", 0) * 1000,
                p95_ms=latency.get("p95_seconds_upper_bound", 0) * 1000,
                p99_ms=latency.get("p99_seconds_upper_bound", 0) * 1000,
                maximum_age_ms=latency.get("maximum_seconds", 0) * 1000,
                scheduling_lag_ms=sender.get("schedule_lag_max_seconds", 0) * 1000,
                input_queue_peak=service.get("input_queue", {}).get("high_water_frames", 0),
                output_queue_peak=service.get("output_queue", {}).get("high_water_frames", 0),
                delivery_queue_peak=delivery.get("queue", {}).get("high_water_frames", 0),
                receiver_peak_rss_bytes=receiver.get("peak_rss_bytes", 0),
                receiver_rss_range_bytes=receiver.get("measured_rss_range_bytes", 0),
                combined_sampled_peak_rss_bytes=max(
                    (s.get("combined_rss_bytes", 0) for s in samples), default=0
                ),
                gpu_peak_mib=max((s["used_mib"] for s in samples), default=0),
                gpu_mean_percent=sum(s["utilization_percent"] for s in samples) / len(samples)
                if samples
                else 0,
                gpu_mean_watts=sum(s["power_watts"] for s in samples) / len(samples)
                if samples
                else 0,
                initialization_seconds=receiver.get("initialization_seconds", 0),
                inference_active_seconds=stages.get("tensorrt_seconds", 0),
                geometry_active_seconds=stages.get("cpu_geometry_seconds", 0),
                input_sha256=receiver.get("input_sha256"),
                output_sha256=receiver.get("output_sha256"),
                snapshot_sha256=receiver.get("map_snapshot_sha256"),
                failure_phase=receiver.get(
                    "failure_phase",
                    "measured"
                    if receiver.get("freshly_classified_measured_points", 0)
                    else "warmup",
                )
                if trial["status"] != "PASS"
                else "none",
                stop_reason=receiver.get("stop_reason", receiver.get("error", "")),
            )
            if trial["status"] == "PASS":
                assert all(
                    native[k]["status"] == "PASS" and native[k]["exit_code"] == 0 for k in native
                )
                assert not trial["hardware_stop"] and trial["model_before"] == trial["model_after"]
                assert (
                    consumer["coverage"] == 1
                    and service["dropped_points"] == 0
                    and delivery["failed_point_deliveries"] == 0
                )
                assert receiver["fresh_execution_count_match"]
                assert row["p99_ms"] <= 250 and row["scheduling_lag_ms"] <= 50
                assert (
                    row["input_queue_peak"] <= 2
                    and row["output_queue_peak"] <= 2
                    and row["delivery_queue_peak"] <= 2
                )
                for queue in (service["input_queue"], service["output_queue"], delivery["queue"]):
                    assert queue["capacity_frames"] == 2 and queue["depth_frames"] == 0
                assert (
                    row["receiver_peak_rss_bytes"] <= 1024**3
                    and row["receiver_rss_range_bytes"] <= 64 * 1024**2
                )
                assert row["gpu_peak_mib"] <= 4096
                count = sender["incoming_points"]
                assert (
                    count
                    == receiver["received_points"]
                    == receiver["output_points"]
                    == consumer["points"]
                    == service["freshly_classified_points"]
                    == service["emitted_points"]
                )
                assert (
                    consumer["input_sha256"] == receiver["input_sha256"]
                    and consumer["output_sha256"] == receiver["output_sha256"]
                )
            rows.append(row)
            if trial["seconds"] >= 60 and trial.get("sustained_claim"):
                key = identity(row)
                groups[key].append(row)
    with (destination / "trials.csv").open("x") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    qualified = []
    for key, trials in groups.items():
        # Any matching failed attempt prevents that image/configuration being promoted.
        matching = [r for r in rows if identity(r) == key and r["requested_seconds"] >= 60]
        if len(trials) >= 2 and all(r["status"] == "PASS" for r in matching):
            assert all(
                r["input_sha256"] == trials[0]["input_sha256"]
                and r["output_sha256"] == trials[0]["output_sha256"]
                for r in trials
            )
            snapshots = [r["snapshot_sha256"] for r in trials if r["snapshot_sha256"]]
            if snapshots:
                assert all(x == snapshots[0] for x in snapshots)
            qualified.append(
                dict(
                    binary_sha256=key[0],
                    source_sha256=key[1],
                    rate=key[2],
                    cells=key[3],
                    churn_mm=key[4],
                    requested_seconds=key[5],
                    validation_mode=key[6],
                    async_export=key[7],
                    model_manifest_sha256=key[8],
                    repetitions=len(trials),
                    status="PASS",
                )
            )
    result = dict(
        schema="synrex.streaming_r1.transport_summary.v1",
        status="PASS",
        trial_count=len(rows),
        passing_trials=sum(r["status"] == "PASS" for r in rows),
        failed_trials=sum(r["status"] == "FAIL" for r in rows),
        repeated_sustained_configurations=qualified,
        scope=(
            "Summary verification only. Individual FAILs retained. At least two sustained PASS "
            "repetitions and no matching failure required for a qualified configuration."
        ),
    )
    (destination / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--receipts", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(summarize(a.receipts, a.out), indent=2))


if __name__ == "__main__":
    main()
