# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
"""Long target-rate confirmations and independent fresh point-only regressions."""

import argparse
import json
from pathlib import Path

from sustained_campaign import run_trial, sha

p = argparse.ArgumentParser()
p.add_argument("--native", type=Path, required=True)
p.add_argument("--model", type=Path, required=True)
p.add_argument("--sources", type=Path, nargs="+", required=True)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--source-commit", required=True)
a = p.parse_args()
a.out.mkdir(exist_ok=False)
before = {p.name: sha(p) for p in a.model.iterdir()}
r = {
    "schema": "synrex.long_confirmations.v1",
    "source_commit": a.source_commit,
    "status": "FAIL",
    "trials": [],
    "model_hashes_before": before,
}


def trial(label, rate, points, source, mode="points-map", seconds=30, warmup=5, expected="PASS"):
    t = run_trial(
        a.native,
        a.model,
        source,
        a.out / (label + ".json"),
        rate=rate,
        mode=mode,
        points=points,
        seconds=seconds,
        warmup=warmup,
    )
    t["experiment_label"] = label
    t["expected_status"] = expected
    r["trials"].append(t)
    (a.out / "progress.json").write_text(json.dumps(r, indent=2) + "\n")
    return t["status"] == expected


try:
    for i, source in enumerate((a.sources[0], a.sources[0], a.sources[-1])):
        if not trial("long-2m-" + str(i + 1), 2000000, 16384, source, seconds=60):
            break
    for i in (1, 2):
        trial("point-only-1m-" + str(i), 1000000, 4096, a.sources[0], mode="points")
    trial("json-compatible-window", 50000, 4096, a.sources[0], mode="json")
    trial(
        "short-duration-final-gate-regression",
        1000000,
        16384,
        a.sources[0],
        seconds=2,
        warmup=0,
        expected="FAIL",
    )
    r["model_hashes_after"] = {p.name: sha(p) for p in a.model.iterdir()}
    r["model_unchanged"] = r["model_hashes_after"] == before
    r["status"] = (
        "PASS"
        if r["model_unchanged"] and all(t["status"] == t["expected_status"] for t in r["trials"])
        else "FAIL"
    )
finally:
    (a.out / "receipt.json").write_text(json.dumps(r, indent=2) + "\n")
