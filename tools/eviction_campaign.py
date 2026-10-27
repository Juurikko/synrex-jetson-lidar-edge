# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
"""Guarded frozen/indexed semantic-map parity, sustained churn and stable-scene trials."""

import argparse
import json
from pathlib import Path

from sustained_campaign import run_trial, sha

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--native", type=Path, required=True)
p.add_argument("--baseline", type=Path, required=True)
p.add_argument("--model", type=Path, required=True)
p.add_argument("--sources", type=Path, nargs=3, required=True)
p.add_argument("--source-commit", required=True)
p.add_argument(
    "--phase", choices=["matched", "sweep", "stable", "faults", "full-window"], required=True
)
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=False)
expected_baseline = "a52d815dc923931845a85d1e080ce8d61c8ad6c9e95de95b226d263742a45a97"
if sha(a.baseline) != expected_baseline:
    raise RuntimeError("frozen accepted baseline binary differs")
before = {f.name: sha(f) for f in a.model.iterdir()}
report = {
    "schema": "synrex.eviction_campaign.v1",
    "status": "FAIL",
    "source_commit": a.source_commit,
    "phase": a.phase,
    "frozen_binary_source_commit": "3f97a490860529505b1ebfad03e2726e2081dbe2",
    "frozen_checkpoint": "958c418d6799196605155f0223418d6f3b4f5c4d",
    "baseline_binary_sha256": sha(a.baseline),
    "optimized_binary_sha256": sha(a.native),
    "model_hashes_before": before,
    "trials": [],
    "parity": [],
    "not_run": [],
    "scope": "Controller completion does not override individual trial FAILs",
}


def trial(
    label, rate, points=256, cells=65536, seconds=60, churn=64000, baseline=False, source=0, **kw
):
    r = run_trial(
        a.baseline if baseline else a.native,
        a.model,
        a.sources[source],
        a.out / (label + ".json"),
        rate=rate,
        points=points,
        seconds=seconds,
        mode="points-map",
        geometry="direct",
        map_cells=cells,
        churn=churn,
        verify_map=True,
        **({} if baseline else {"map_policy": "indexed"}),
        **kw,
    )
    r["experiment_label"] = label
    r["implementation"] = "frozen-reference" if baseline else "intrusive-index"
    report["trials"].append(r)
    save("progress.json")
    return r


def save(name):
    (a.out / name).write_text(json.dumps(report, indent=2) + "\n")


def parity(label, runs):
    if any(r["status"] != "PASS" for r in runs):
        report["parity"].append(
            {"label": label, "status": "NOT_RUN", "reason": "trial gate failed"}
        )
        return
    for key in (
        "point_output_digest",
        "map_delta_digest",
        "map_snapshot_sha256",
        "semantic_histogram",
    ):
        if any(r[key] != runs[0][key] for r in runs[1:]):
            raise RuntimeError("exact output parity failed: " + key)
    for key in (
        "insertions",
        "existing_updates",
        "removals",
        "late_rejections",
        "delta_events",
        "observations",
        "peak_cells",
    ):
        if any(
            r["map_operations_including_warmup"][key]
            != runs[0]["map_operations_including_warmup"][key]
            for r in runs[1:]
        ):
            raise RuntimeError("map operation parity failed: " + key)
    report["parity"].append(
        {"label": label, "status": "PASS", "trials": [r["experiment_label"] for r in runs]}
    )


def pair(label, rate, **kw):
    one = trial(label + "-t1", rate, **kw)
    if one["status"] != "PASS":
        return False
    two = trial(label + "-t2", rate, **kw)
    parity(label, [one, two])
    return two["status"] == "PASS"


try:
    if a.phase == "matched":
        for label, rate, points, cells, seconds in [
            ("original4096", 50000, 256, 4096, 60),
            ("original65536", 2500, 32, 65536, 120),
        ]:
            base = trial(label + "-frozen", rate, points, cells, seconds, baseline=True)
            first = trial(label + "-indexed-t1", rate, points, cells, seconds)
            second = trial(label + "-indexed-t2", rate, points, cells, seconds)
            parity(label, [base, first, second])
    elif a.phase == "sweep":
        for points, rates in [
            (256, (50000, 100000)),
            (4096, (100000, 250000, 500000, 750000, 1000000)),
        ]:
            for rate in rates:
                if not pair(
                    f"churn65536-p{points}-{rate}",
                    rate,
                    points=points,
                    seconds=120 if rate == 100000 else 60,
                ):
                    report["not_run"].append(
                        {
                            "status": "NOT_RUN",
                            "window_points": points,
                            "higher_than_failed_rate": rate,
                            "reason": "Gate failed; higher rates skipped for this packet size",
                        }
                    )
                    break
        # Cancellation is expected to fail completion visibly; partial maps still pass the oracle.
        trial("cancellation-visible", 100000, points=4096, seconds=60, cancel=64)
    elif a.phase == "full-window":
        # Revisit the exact packet size/rate of the preserved pre-index warmup FAIL.
        # Stop after this bounded target; do not extrapolate to higher rates.
        pair("churn65536-p16384-2000000", 2000000, points=16384, seconds=60)
    elif a.phase == "faults":
        cancelled = trial("post-fill-cancellation", 100000, points=4096, seconds=60, cancel=6500)
        if (
            cancelled["status"] != "FAIL"
            or cancelled.get("stop_reason") != "requested_cancellation"
            or cancelled.get("map_oracle_status") != "PASS"
            or cancelled["map_operations_including_warmup"]["removals"] == 0
        ):
            raise RuntimeError("post-fill cancellation did not preserve a checked map prefix")
    else:
        for scene, rate in [(0, 2000000), (0, 3000000), (1, 3000000), (2, 2500000)]:
            label = f"stable-scene{scene}-{rate}"
            base = trial(
                label + "-frozen", rate, points=16384, churn=0, source=scene, baseline=True
            )
            one = trial(label + "-indexed-t1", rate, points=16384, churn=0, source=scene)
            two = trial(label + "-indexed-t2", rate, points=16384, churn=0, source=scene)
            parity(label, [base, one, two])
    report["model_hashes_after"] = {f.name: sha(f) for f in a.model.iterdir()}
    report["model_unchanged"] = before == report["model_hashes_after"]
    if not report["model_unchanged"]:
        raise RuntimeError("frozen model package changed")
    report["status"] = "PASS"
finally:
    save("receipt.json")
