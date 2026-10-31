# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
"""Compare paired manifest receipts; never launches jobs or qualifies GPU capacity."""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

MODES = ("send", "receive", "subscribe")
COUNTERS = (
    "accepted_records",
    "freshly_classified_records",
    "emitted_records",
    "independently_validated_records",
    "subscriber_snapshot_acknowledged_records",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


MAX_RECEIPT_BYTES = 16 * 1024 * 1024


def strict_json(raw):
    def unique(items):
        value = {}
        for key, item in items:
            require(key not in value, "duplicate JSON key")
            value[key] = item
        return value

    def invalid_constant(_):
        raise ValueError("non-finite JSON constant")

    return json.loads(raw, object_pairs_hook=unique, parse_constant=invalid_constant)


def read_json(path, maximum=MAX_RECEIPT_BYTES):
    with path.open("rb") as source:
        raw = source.read(maximum + 1)
    require(len(raw) <= maximum, "JSON evidence exceeds byte budget")
    return raw, strict_json(raw)


def uint(value, positive=False):
    return type(value) is int and (value > 0 if positive else value >= 0) and value <= 2**64 - 1


def validate_histogram(value, count):
    require(
        isinstance(value, dict) and value["schema"] == "synrex.replay.latency_histogram.v1",
        "invalid latency histogram",
    )
    require(
        uint(value["count"]) and value["count"] == count and uint(value["sum_ns"]),
        "invalid histogram count/sum",
    )
    require(
        uint(value["storage_bytes"], True) and value["storage_bytes"] <= 32768,
        "invalid histogram storage bound",
    )
    if not count:
        require(
            value["sum_ns"] == 0
            and value["maximum_ns"] is None
            and all(value[f"p{p}_ns"] is None for p in (50, 95, 99)),
            "empty histogram has tails",
        )
        return
    maximum = value["maximum_ns"]
    require(
        uint(maximum) and maximum <= value["sum_ns"] <= maximum * count,
        "invalid histogram maximum/sum",
    )
    previous_lower = previous_upper = 0
    for percentile in (50, 95, 99):
        bounds = value[f"p{percentile}_ns"]
        require(
            isinstance(bounds, dict) and set(bounds) == {"lower_ns", "upper_ns"},
            "invalid histogram quantile",
        )
        lower, upper = bounds["lower_ns"], bounds["upper_ns"]
        require(
            uint(lower)
            and uint(upper)
            and previous_lower <= lower <= upper
            and previous_upper <= upper
            and lower <= maximum
            and upper - lower <= lower // 64,
            "invalid histogram quantile bounds",
        )
        rank = (percentile * count + 99) // 100
        require(
            lower * (count - rank + 1)
            <= value["sum_ns"]
            <= upper * rank + maximum * (count - rank),
            "quantile rank inconsistent with sum",
        )
        previous_lower, previous_upper = lower, upper


def validate_timings(receipt):
    names = {
        "send": (
            "decode_window",
            "assembly_and_validation",
            "frame_send",
            "sender_scheduling_lag",
            "synthetic_accumulation",
        ),
        "receive": ("frame_ingress", "receiver_service"),
        "subscribe": ("classification_receive_and_map", "source_attribute_validation"),
    }
    expected = names[receipt["mode"]]
    require(
        isinstance(receipt["trial_timings"], dict)
        and set(receipt["trial_timings"]) == set(expected),
        "invalid trial timing metrics",
    )
    for name in expected:
        values = []
        for row in receipt["sources"]:
            require(
                isinstance(row["timings"], dict) and set(row["timings"]) == set(expected),
                "invalid source timing metrics",
            )
            value = row["timings"][name]
            count = (
                row["expected_records"]
                if name in ("synthetic_accumulation", "receiver_service")
                else row["frames"]
            )
            validate_histogram(value, count)
            values.append(value)
        trial = receipt["trial_timings"][name]
        validate_histogram(trial, sum(value["count"] for value in values))
        require(
            trial["sum_ns"] == sum(value["sum_ns"] for value in values)
            and trial["maximum_ns"] == max(value["maximum_ns"] for value in values)
            and all(
                value["population"] == trial["population"]
                and value["boundary"] == trial["boundary"]
                for value in values
            ),
            "trial timing population mismatch",
        )


def validate_counts(receipt):
    require(
        uint(receipt["source_count"], True)
        and 1 <= receipt["source_count"] <= 4096
        and len(receipt["sources"]) == receipt["source_count"]
        and uint(receipt["declared_records"], True),
        "invalid declared population",
    )
    config = receipt["replay_configuration"]
    require(
        set(config)
        == {"window_points", "points_per_second", "map_cells", "queue_frames", "io_timeout_ms"}
        and all(uint(value, True) for value in config.values())
        and config["window_points"] <= 65536
        and config["queue_frames"] <= 2
        and config["window_points"] <= config["points_per_second"] <= 20000000
        and config["map_cells"] <= 65536
        and config["io_timeout_ms"] <= 120000,
        "invalid replay configuration",
    )
    records = 0
    role = receipt["mode"]
    for index, row in enumerate(receipt["sources"]):
        count, frames = row["expected_records"], row["frames"]
        require(
            uint(count, True) and uint(frames, True) and row["source_index"] == index,
            "invalid source population",
        )
        window = receipt["replay_configuration"]["window_points"]
        require(
            uint(window, True) and frames == (count + window - 1) // window,
            "invalid frame coverage",
        )
        coverage = row["coverage"]
        require(
            coverage["accepted_points"] == count
            and coverage["windows"] == frames
            and coverage["source_sha256"] == row["source_sha256"]
            and coverage["source_index"] == index
            and coverage["wire_begin"] == records,
            "inconsistent source coverage",
        )
        if role == "send":
            queue = row["decode_queue"]
            require(
                queue["capacity"] == config["queue_frames"]
                and queue["window_points"] == window
                and queue["closed"] is True
                and queue["failed"] is False
                and all(
                    queue[key] == 0
                    for key in (
                        "queued_items",
                        "queued_bytes",
                        "queued_allocation_bytes",
                        "discarded_items",
                        "discarded_points",
                    )
                )
                and 0 <= queue["high_water_items"] <= queue["capacity"]
                and 0 <= queue["high_water_allocation_bytes"] <= queue["byte_limit"],
                "inconsistent decode queue resource",
            )
        expected = {
            "send": (0, 0, count, 0, 0),
            "receive": (count, count, count, 0, count),
            "subscribe": (count, 0, 0, count, 0),
        }[role]
        require(
            all(
                uint(row[key]) and row[key] == value
                for key, value in zip(COUNTERS, expected, strict=True)
            ),
            "inconsistent source counters",
        )
        for field in (
            ("input_completion", "output_completion")
            if role == "receive"
            else ("input_completion",)
            if role == "send"
            else ("output_completion",)
        ):
            completion = row[field]
            is_input = field == "input_completion"
            point_key, frame_key = (
                ("input_points", "input_frames")
                if is_input
                else ("classified_points", "classification_frames")
            )
            status = (
                "PEER_VALIDATED"
                if (role == "send" or role == "receive" and not is_input)
                else "VALIDATED_ACK_SENT"
            )
            require(
                completion["schema"]
                == ("synrex.input.delivery.v1" if is_input else "synrex.output.delivery.v1")
                and completion["status"] == status
                and completion[point_key] == count
                and completion[frame_key] == frames
                and all(
                    completion[key] == row[key]
                    for key in ("descriptor_sha256", "taxonomy_sha256", "map_epoch")
                ),
                "inconsistent completion fence",
            )
        records += count
    require(
        records == receipt["declared_records"]
        and all(
            uint(receipt[key]) and receipt[key] == sum(row[key] for row in receipt["sources"])
            for key in COUNTERS
        ),
        "inconsistent trial counters",
    )


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def semantic_identity(receipt):
    """Exact transcript hashes bind points/classes/map; random ACK nonces are excluded."""
    identity = {
        name: receipt[name]
        for name in (
            "mode",
            "binary_wire",
            "manifest_sha256",
            "descriptor",
            "declared_records",
            "source_count",
            "offered_rate_pps",
            "boundary_policy",
            "replay_configuration",
            *COUNTERS,
        )
    }
    # Profiling is independent of the replay histogram switch. Do not attribute
    # a profiling configuration change to timing-instrumentation overhead.
    profile_setting = receipt.get("backend_profiling", "disabled")
    require(profile_setting in ("disabled", "enabled"), "invalid backend profiling setting")
    identity["backend_profiling"] = profile_setting
    identity["sources"] = []
    for row in receipt["sources"]:
        source = {
            name: row[name]
            for name in (
                "source_index",
                "source_sha256",
                "expected_records",
                "descriptor_sha256",
                "taxonomy_sha256",
                "map_epoch",
                "frames",
                "coverage",
                *COUNTERS,
            )
        }
        require(
            row["status"] == "PASS"
            and row["coverage"]["state"] == "SUCCEEDED"
            and row["coverage"]["declared_frame_coverage_complete"],
            "incomplete source coverage",
        )
        for name in ("input_completion", "output_completion"):
            if name in row:
                source[name] = {k: v for k, v in row[name].items() if not k.endswith("challenge")}
        mode = receipt["mode"]
        completion = source["input_completion" if mode == "send" else "output_completion"]
        hashes = (
            ("input_sha256",) if mode == "send" else ("classification_sha256", "snapshot_sha256")
        )
        for name in hashes:
            require(
                isinstance(completion[name], str)
                and len(completion[name]) == 64
                and all(c in "0123456789abcdef" for c in completion[name]),
                "missing completion transcript hash",
            )
        if mode == "receive":
            source["session_identity"] = row["session_identity"]
        identity["sources"].append(source)
    require(len(identity["sources"]) == receipt["source_count"], "unattempted source")
    return identity


def finite_number(value, positive=False):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and (value > 0 if positive else value >= 0)
    )


def compare(plan, root):
    require(plan["schema"] == "synrex.replay.overhead_pairs.v1", "unknown pair schema")
    require(
        plan["scope"] in ("opaque_cpu_fixture", "unqualified_native_pipeline"),
        "scope must preserve CPU/native qualification boundary",
    )
    require(3 <= len(plan["pairs"]) <= 100, "require 3..100 paired repetitions")
    executable = (root / plan["executable"]["path"]).resolve()
    executable_sha = hashlib.sha256(executable.read_bytes()).hexdigest()
    require(executable_sha == plan["executable"]["sha256"], "executable identity changed")
    baseline = None
    evidence = []
    used_paths = set()
    used_receipt_hashes = set()
    for index, pair in enumerate(plan["pairs"]):
        order = ["disabled", "enabled"] if index % 2 == 0 else ["enabled", "disabled"]
        require(pair["order"] == order, "paired order must alternate")
        sample = {"pair": index, "order": order, "modes": {}}
        loaded = {}
        for setting in order:
            require(set(pair[setting]) == set(MODES), "all three process receipts required")
            loaded[setting] = {}
            for mode in MODES:
                path = (root / pair[setting][mode]).resolve()
                require(path not in used_paths, "receipt reused across observations")
                used_paths.add(path)
                raw, receipt = read_json(path)
                raw_sha = hashlib.sha256(raw).hexdigest()
                require(
                    raw_sha not in used_receipt_hashes, "receipt bytes reused across observations"
                )
                used_receipt_hashes.add(raw_sha)
                require(
                    receipt["schema"] == "synrex.generic_manifest_cli.v1"
                    and receipt["status"] == "PASS"
                    and receipt["mode"] == mode,
                    "only complete passing process receipts may be compared",
                )
                require(receipt["timing_instrumentation"] == setting, "wrong instrumentation mode")
                require(
                    receipt["capacity_qualification"] == "NOT_RUN"
                    and receipt["physical_capture_age_seconds"] is None,
                    "unexpected qualification claim",
                )
                validate_counts(receipt)
                require(
                    finite_number(receipt["elapsed_seconds"], True)
                    and finite_number(receipt["execution_seconds"], True)
                    and receipt["execution_seconds"] <= receipt["elapsed_seconds"],
                    "inconsistent process execution duration",
                )
                if mode == "receive" and plan["scope"] == "opaque_cpu_fixture":
                    require(
                        all(
                            "opaque CPU" in row["session_identity"].get("fixture", "")
                            for row in receipt["sources"]
                        ),
                        "CPU fixture scope lacks fixture identity",
                    )
                if setting == "disabled":
                    require(
                        receipt["timing_availability"] == "UNAVAILABLE_INSTRUMENTATION_DISABLED"
                        and not receipt["trial_timings"]
                        and all(row["timings"] is None for row in receipt["sources"]),
                        "disabled timing must remain unavailable",
                    )
                else:
                    require(
                        receipt["timing_availability"] == "MEASURED" and receipt["trial_timings"],
                        "enabled timing unavailable",
                    )
                    validate_timings(receipt)
                identity = semantic_identity(receipt)
                loaded[setting][mode] = identity
                metrics = {
                    name: receipt[name]
                    for name in (
                        "execution_seconds",
                        "process_user_cpu_seconds",
                        "process_system_cpu_seconds",
                        "process_peak_rss_bytes",
                    )
                }
                require(
                    all(
                        finite_number(
                            v, positive=k in ("execution_seconds", "process_peak_rss_bytes")
                        )
                        for k, v in metrics.items()
                    ),
                    "unavailable or invalid process resource",
                )
                require(uint(metrics["process_peak_rss_bytes"], True), "invalid integer peak RSS")
                metrics.update(
                    {
                        "receipt_path": str(path),
                        "receipt_sha256": raw_sha,
                        "semantic_sha256": digest(identity),
                        "tail_histograms": receipt["trial_timings"]
                        if setting == "enabled"
                        else None,
                    }
                )
                sample["modes"].setdefault(mode, {})[setting] = metrics
        for setting in order:
            peers = loaded[setting]
            for key in (
                "binary_wire",
                "manifest_sha256",
                "descriptor",
                "declared_records",
                "source_count",
                "offered_rate_pps",
                "boundary_policy",
                "replay_configuration",
            ):
                require(
                    all(peers[mode][key] == peers["send"][key] for mode in MODES),
                    "peer workload identity disagreement",
                )
            for mode in ("receive", "subscribe"):
                for source, peer in zip(
                    peers["send"]["sources"], peers[mode]["sources"], strict=True
                ):
                    require(
                        all(
                            source[key] == peer[key]
                            for key in (
                                "source_index",
                                "source_sha256",
                                "expected_records",
                                "frames",
                                "descriptor_sha256",
                                "taxonomy_sha256",
                                "map_epoch",
                            )
                        ),
                        "peer source identity disagreement",
                    )
                    common_coverage = set(source["coverage"]) - {
                        "exact_attribute_points",
                        "identity_only_points",
                        "raw_source_attribute_points",
                    }
                    require(
                        all(
                            source["coverage"][key] == peer["coverage"][key]
                            for key in common_coverage
                        ),
                        "peer source coverage disagreement",
                    )
            for sender, receiver, field, hashes in (
                (
                    "send",
                    "receive",
                    "input_completion",
                    ("input_sha256", "input_points", "input_frames"),
                ),
                (
                    "receive",
                    "subscribe",
                    "output_completion",
                    (
                        "classification_sha256",
                        "snapshot_sha256",
                        "classified_points",
                        "classification_frames",
                    ),
                ),
            ):
                for a, b in zip(peers[sender]["sources"], peers[receiver]["sources"], strict=True):
                    require(
                        all(a[field][key] == b[field][key] for key in hashes),
                        "peer completion transcript disagreement",
                    )
        require(loaded["enabled"] == loaded["disabled"], "paired output or workload mismatch")
        if baseline is None:
            baseline = loaded["enabled"]
        require(loaded["enabled"] == baseline, "workload/output changed across repetitions")
        for values in sample["modes"].values():
            values["enabled_minus_disabled"] = {
                name: values["enabled"][name] - values["disabled"][name]
                for name in (
                    "execution_seconds",
                    "process_user_cpu_seconds",
                    "process_system_cpu_seconds",
                    "process_peak_rss_bytes",
                )
            }
        evidence.append(sample)
    return {
        "schema": "synrex.replay.overhead_comparison.v1",
        "status": "PAIRED_OUTPUTS_IDENTICAL",
        "scope": plan["scope"],
        "pairs": evidence,
        "executable": {"path": str(executable), "sha256": executable_sha},
        "source_identity": plan.get("source_identity"),
        "semantic_sha256": digest(baseline),
        "median_execution_delta_seconds": {
            mode: statistics.median(
                pair["modes"][mode]["enabled_minus_disabled"]["execution_seconds"]
                for pair in evidence
            )
            for mode in MODES
        },
        "actual_gpu_instrumentation_qualification": "NOT_ESTABLISHED_BY_THIS_TOOL",
        "capacity_qualification": "NOT_RUN",
        "physical_capture_age_seconds": None,
        "limitations": "Finite paired replay only. Signed deltas include noise, startup, pacing; "
        "no significance, capacity or tail comparison against disabled measurements. "
        "Queue/backend/correctness instrumentation and per-source histogram arrays remain. "
        "Pair order is declared by the recording runner, not independently timed here.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pairs", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    raw, plan = read_json(args.pairs, 1024 * 1024)
    result = compare(plan, args.pairs.resolve().parent)
    result["pair_plan_sha256"] = hashlib.sha256(raw).hexdigest()
    with args.output.open("x") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
