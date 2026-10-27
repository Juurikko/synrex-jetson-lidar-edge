# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
"""Summarize immutable native campaign receipts without hiding individual failed trials."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def summarize(t, campaign, index):
    if t.get("schema") == "synrex.model_only_sustained.v1":
        keep = [
            "schema",
            "status",
            "measurement",
            "warmup",
            "source_points_represented_per_inference",
            "source_point_equivalents_per_second",
            "batch_average_inference_latency",
            "batch_enqueues_per_sync",
            "freshly_classified_emitted_source_points",
            "scope",
            "initialization_wall_seconds",
        ]
    else:
        keep = [
            "schema",
            "status",
            "correctness_status",
            "latency_status",
            "sustained_duration_status",
            "fresh_execution_count_status",
            "stop_reason",
            "failure_phase",
            "mode",
            "source",
            "frame_points",
            "requested_rate_points_per_second",
            "warmup_seconds",
            "measured_seconds_planned",
            "measured_seconds_completed",
            "measurement_including_drain_seconds",
            "incoming_points_per_second",
            "freshly_neural_classified_points_per_second",
            "fully_classified_emitted_points_per_second",
            "semantic_map_updates_per_second",
            "map_updates",
            "map_cells",
            "map_capacity",
            "measured_incoming_points",
            "measured_source_points",
            "expected_measured_source_points",
            "classification_coverage",
            "measurement_completion_fraction",
            "duplicated_or_corrupt_output_points",
            "delayed_points_over_latency_budget",
            "source_to_output_latency",
            "inference_window_latency",
            "input_queue_wait",
            "output_consumer_latency",
            "maximum_pending_frames",
            "maximum_oldest_pending_source_age_seconds",
            "maximum_schedule_lag_seconds",
            "host_initial_rss_bytes",
            "host_peak_rss_bytes",
            "measured_rss_range_bytes",
            "memory_stability_status",
            "point_output_digest",
            "map_delta_digest",
            "stage_seconds_totals",
            "cold_first_window",
            "initialization_wall_seconds",
            "source_decode_seconds_before_replay",
            "source_generation_seconds_including_warmup",
            "json_frontend_seconds",
            "optional_file_io",
            "optional_encoded_bytes",
            "classes_reused_points",
            "geometry_only_points",
            "spatial_source_overlap_percent",
            "arrival_jitter_maximum_microseconds",
            "motion_x_amplitude_metres",
            "occlusion_alternating_rank_x_percent",
            "observed_neural_blocks_including_warmup",
            "measured_neural_blocks",
            "semantic_histogram",
            "occupied_voxels_total",
            "occupied_voxels_per_window",
        ]
    d = {k: t[k] for k in keep if k in t}
    d["expected_status"] = t.get("expected_status")
    if d.get("status") == "FAIL" and d.get("failure_phase") == "none":
        d["failure_phase"] = "final_gate"
        d["failure_phase_note"] = (
            "Derived from final FAIL; original receipt predates phase-report correction"
        )
    for key in (
        "geometry_mode",
        "map_index_profile",
        "map_oracle_status",
        "map_snapshot_sha256",
        "map_operations_measured",
        "map_operations_including_warmup",
        "first_eviction_replay_seconds",
        "full_capacity_churn_seconds_including_drain",
        "churn_x_translation_metres_per_frame",
    ):
        if key in t:
            d[key] = t[key]
    d["trial_id"] = f"{campaign}:{t.get('experiment_label', index)}"
    d["command"] = t.get("command")
    d["binary_sha256"] = t.get("binary_sha256")
    d["exit_code"] = t.get("exit_code")
    d["hardware_stop"] = t.get("hardware_stop")
    if "failure_phase" not in d and t.get("schema") != "synrex.model_only_sustained.v1":
        d["failure_phase"] = (
            "none"
            if t["status"] == "PASS"
            else "warmup"
            if not t.get("measured_seconds_completed")
            else "measured"
        )
    if "service" in t:
        d["service"] = {k: v for k, v in t["service"].items() if k != "session"}
    model = t.get("model", t.get("session", {}))
    d["session"] = {k: v for k, v in model.items() if k not in ("contract", "target", "artifacts")}
    samples = t.get("device_samples", [])
    if samples:
        d["gpu"] = {
            "scope": (
                "whole native process, including initialization and warmup; "
                "250ms polling sleep plus query overhead; raw sample timestamps retained"
            ),
            "sample_count": len(samples),
            "peak_used_mib": max(s["used_mib"] for s in samples),
            "mean_utilization_percent": statistics.mean(s["utilization_percent"] for s in samples),
            "max_utilization_percent": max(s["utilization_percent"] for s in samples),
            "peak_power_watts": max(s["power_watts"] for s in samples),
            "mean_power_watts": statistics.mean(s["power_watts"] for s in samples),
        }
        for key in ["sm_clock_mhz", "memory_clock_mhz"]:
            vals = [s[key] for s in samples if key in s]
            if vals:
                d["gpu"][key + "_range"] = [min(vals), max(vals)]
    d["raw_trial_canonical_sha256"] = hashlib.sha256(
        json.dumps(t, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return d


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", action="append", required=True, metavar="LABEL=RECEIPT")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output must be new")
    report = {
        "schema": "synrex.sustained_rtx_trials.v1",
        "scope": (
            "Bounded offline-emulated native source-point replay, "
            "not live sensor/general-purpose hardware qualification"
        ),
        "artifacts": [],
        "trials": [],
        "not_run": [],
    }
    for item in args.campaign:
        name, file = item.split("=", 1)
        p = Path(file)
        r = json.loads(p.read_text())
        report["artifacts"].append(
            {
                "campaign": name,
                "source_commit": r["source_commit"],
                "raw_receipt_sha256": sha(p),
                "model_package_unchanged": r.get(
                    "model_package_unchanged", r.get("model_unchanged")
                ),
            }
        )
        report["trials"].extend(summarize(t, name, i) for i, t in enumerate(r["trials"]))
        report["not_run"] += r.get("skipped_rates", r.get("not_run", []))
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print("trials", len(report["trials"]))


if __name__ == "__main__":
    main()
