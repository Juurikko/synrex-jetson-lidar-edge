# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
"""Matched reference/direct/flat and changing-coordinate neural map qualification."""

import argparse
import json
from pathlib import Path

from sustained_campaign import run_trial, sha


def campaign(a):
    a.out.mkdir(parents=True, exist_ok=False)
    before = {p.name: sha(p) for p in a.model.iterdir()}
    report = {
        "schema": "synrex.geometry_campaign.v1",
        "source_commit": a.source_commit,
        "status": "FAIL",
        "trials": [],
        "not_run": [],
        "model_hashes_before": before,
    }

    def trial(label, rate, strategy, source=None, points=16384, seconds=60, **kwargs):
        r = run_trial(
            a.native,
            a.model,
            source or a.sources[0],
            a.out / (label + ".json"),
            rate=rate,
            mode="points-map",
            points=points,
            seconds=seconds,
            geometry=strategy,
            verify_map=True,
            **kwargs,
        )
        r["experiment_label"] = label
        report["trials"].append(r)
        (a.out / "progress.json").write_text(json.dumps(report, indent=2) + "\n")
        return r

    def twice(label, rate, strategy, **kwargs):
        first = trial(label + "-t1", rate, strategy, **kwargs)
        if first["status"] != "PASS":
            return False
        second = trial(label + "-t2", rate, strategy, **kwargs)
        if second["status"] != "PASS":
            return False
        for key in ("point_output_digest", "map_delta_digest", "map_snapshot_sha256"):
            if first[key] != second[key]:
                raise RuntimeError("repeat output parity mismatch")
        return True

    try:
        # Same binary, unchanged reference function, identical validators and queues.
        for strategy in ("reference", "direct", "flat"):
            twice(f"ab-{strategy}-2000000", 2000000, strategy)
        complete_ab = [
            r
            for r in report["trials"]
            if r["experiment_label"].startswith("ab-") and r["status"] == "PASS"
        ]
        for key in ("point_output_digest", "map_delta_digest", "map_snapshot_sha256"):
            if len({r[key] for r in complete_ab}) > 1:
                raise RuntimeError("baseline/optimized exact output parity mismatch")
        report["completed_ab_output_parity"] = "PASS" if len(complete_ab) == 6 else "NOT_RUN"
        best = "direct"
        for strategy in ("reference", best):
            for rate in (2250000, 2500000, 3000000):
                if not twice(f"sweep-{strategy}-{rate}", rate, strategy):
                    report["not_run"].append(
                        {
                            "status": "NOT_RUN",
                            "strategy": strategy,
                            "higher_than_failed_rate": rate,
                            "reason": "lower rate violated a fixed gate",
                        }
                    )
                    break
        passing = [
            r["requested_rate_points_per_second"]
            for r in report["trials"]
            if r["status"] == "PASS" and r.get("geometry_mode") == best
        ]
        target = max(passing) if passing else 2000000
        for index, source in enumerate(a.sources[1:], 1):
            twice(f"scene{index}-{best}-{target}", target, best, source=source)
        # Keep the input identical when contrasting eviction capacity effects.
        # Fast translation creates genuinely new world voxels, not repeated local fixtures.
        for capacity in (4096, 65536):
            for rate in (50000, 25000, 10000):
                if twice(
                    f"churn-{capacity}-{rate}",
                    rate,
                    best,
                    points=256,
                    churn=64000,
                    map_cells=capacity,
                    seconds=120 if capacity == 65536 else 60,
                ):
                    break
        # Original target size/rate at full-map churn: stop visibly on queue overload.
        trial("churn-65536-2m-16384", 2000000, best, churn=64000, map_cells=65536)
        report["model_hashes_after"] = {p.name: sha(p) for p in a.model.iterdir()}
        report["model_unchanged"] = report["model_hashes_after"] == before
        report["status"] = "PASS" if report["model_unchanged"] else "FAIL"
        report["scope"] = (
            "Campaign executed; individual trial FAILs are retained. No general throughput claim."
        )
    finally:
        (a.out / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--native", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--sources", type=Path, nargs="+", required=True)
    p.add_argument("--source-commit", required=True)
    p.add_argument("--out", type=Path, required=True)
    campaign(p.parse_args())
