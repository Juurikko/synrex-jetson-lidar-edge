# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
"""Sequential bounded window/scene campaign; every neural call is native TensorRT."""

import argparse
import json
from pathlib import Path

from sustained_campaign import run_trial, sha


def campaign(a):
    a.out.mkdir(parents=True, exist_ok=False)
    files = {p.name: sha(p) for p in a.model.iterdir()}
    report = {
        "schema": "synrex.window_campaign.v1",
        "source_commit": a.source_commit,
        "status": "FAIL",
        "model_hashes_before": files,
        "trials": [],
        "not_run": [],
        "scope": (
            "Per-trial gates apply. Window sizes have different context; "
            "label parity is required only at identical boundaries."
        ),
    }

    def trial(label, rate, points=4096, source=None, native=None, **kwargs):
        receipt = run_trial(
            native or a.native,
            a.model,
            source or a.sources[0],
            a.out / f"{label}.json",
            rate=rate,
            mode=kwargs.pop("mode", "points-map"),
            points=points,
            **kwargs,
        )
        receipt["experiment_label"] = label
        report["trials"].append(receipt)
        (a.out / "progress.json").write_text(json.dumps(report, indent=2) + "\n")
        return receipt

    def repeated(label, rate, points=4096, source=None, **kwargs):
        runs = [trial(label + "-t1", rate, points, source, **kwargs)]
        if runs[0]["status"] == "PASS":
            runs.append(trial(label + "-t2", rate, points, source, **kwargs))
        return all(r["status"] == "PASS" for r in runs) and len(runs) == 2

    try:
        # Close the requested low-rate A/B sweep without changing the reference binary.
        for rate in (600000, 700000, 750000):
            repeated(f"reference-{rate}", rate, native=a.reference)
        for rate in (600000, 700000, 750000):
            repeated(f"optimized-{rate}", rate)
        for rate in (1600000, 1750000, 1900000, 2000000):
            if not repeated(f"4096-{rate}", rate):
                report["not_run"].append(
                    {
                        "status": "NOT_RUN",
                        "scope": "higher 4096 rates",
                        "reason": "bounded lower-rate gate failed; historical 2M failures retained",
                    }
                )
                break
        passed_2m = []
        for points in (8192, 16384, 32768):
            if not repeated(f"{points}-1500000", 1500000, points):
                continue
            if repeated(f"{points}-2000000", 2000000, points):
                passed_2m.append(points)
        best = min(passed_2m) if passed_2m else 4096
        report["selected_window_points"] = best
        if passed_2m:
            for rate in (1600000, 1750000, 1900000):
                if not repeated(f"selected-{best}-{rate}", rate, best):
                    break
        for index, source in enumerate(a.sources[1:], 1):
            for rate in (1000000, 1500000, 2000000):
                if not repeated(f"scene{index}-{best}-{rate}", rate, best, source):
                    break
        # Motion, spatial overlap, alternating occlusion and arrival jitter: fresh
        # inference still runs for every emitted observation; no map-label reuse.
        repeated(
            "temporal-motion-occlusion-jitter",
            1000000,
            best,
            a.sources[-1],
            motion=250,
            occlusion=25,
            overlap=50,
            jitter=1000,
        )
        trial("explicit-cancellation", 1000000, best, cancel=8000)
        trial("cuda-host-timeline", 1500000 if not passed_2m else 2000000, best, timeline=True)
        trial("model-only", 1000000, best, mode="model-only")
        trial("optional-json-encoding", 50000, best, mode="json")
        report["model_hashes_after"] = {p.name: sha(p) for p in a.model.iterdir()}
        report["model_unchanged"] = report["model_hashes_after"] == files
        report["status"] = "PASS" if report["model_unchanged"] else "FAIL"
    finally:
        (a.out / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--native", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--sources", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--source-commit", required=True)
    campaign(p.parse_args())
