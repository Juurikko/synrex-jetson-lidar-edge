# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
"""Bounded scene-C rate fallback and lower-rate full-capacity churn confirmation."""

import argparse
import json
from pathlib import Path

from sustained_campaign import run_trial, sha

p = argparse.ArgumentParser()
p.add_argument("--native", type=Path, required=True)
p.add_argument("--model", type=Path, required=True)
p.add_argument("--sources", type=Path, nargs="+", required=True)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--skip-scenes", action="store_true")
p.add_argument("--churn-rate", type=int, default=5000)
p.add_argument("--churn-points", type=int, default=64)
a = p.parse_args()
a.out.mkdir(exist_ok=False)
before = {f.name: sha(f) for f in a.model.iterdir()}
report = {
    "schema": "synrex.geometry_post.v1",
    "source_commit": "3f97a490860529505b1ebfad03e2726e2081dbe2",
    "status": "FAIL",
    "trials": [],
}


def trial(label, source, rate, points=16384, seconds=60, churn=0, capacity=65536):
    r = run_trial(
        a.native,
        a.model,
        source,
        a.out / (label + ".json"),
        rate=rate,
        mode="points-map",
        geometry="direct",
        verify_map=True,
        points=points,
        seconds=seconds,
        churn=churn,
        map_cells=capacity,
    )
    r["experiment_label"] = label
    report["trials"].append(r)
    (a.out / "progress.json").write_text(json.dumps(report, indent=2) + "\n")
    return r


try:
    for rate in () if a.skip_scenes else (2500000, 2250000, 2000000):
        first = trial("sceneC-" + str(rate) + "-t1", a.sources[-1], rate)
        if first["status"] == "PASS":
            second = trial("sceneC-" + str(rate) + "-t2", a.sources[-1], rate)
            if second["status"] == "PASS":
                break
    for i in (1, 2):
        r = trial(
            f"churn65536-{a.churn_rate}-{a.churn_points}-t{i}",
            a.sources[0],
            a.churn_rate,
            points=a.churn_points,
            seconds=120,
            churn=64000,
        )
        if r["status"] != "PASS":
            break
    report["model_unchanged"] = before == {f.name: sha(f) for f in a.model.iterdir()}
    report["status"] = "PASS" if report["model_unchanged"] else "FAIL"
finally:
    (a.out / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")
