# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
# ruff: noqa: E402
"""Actual seven-class model semantics for disclosed held-out real scenes."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import laspy
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

from dense_training import targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "qualified", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    q = json.loads((args.qualified / "qualification.json").read_text())
    candidates = [max(q["files"], key=lambda r: r["semantic_support"][i]) for i in [3, 4, 5, 6]]
    chosen = {r["source"]: r for r in candidates}
    colors = [
        "#8b9298",
        "#a47c48",
        "#30985b",
        "#e95f50",
        "#3979d0",
        "#bf71cc",
        "#f1b639",
        "#d1d5db",
    ]
    names = [
        "Other / ID0",
        "Ground / ID1",
        "Aggregate vegetation / ID2",
        "Building / ID3",
        "Water / ID4",
        "Bridge / ID5",
        "Permanent structure / ID6",
    ]
    fig, axes = plt.subplots(len(chosen), 2, figsize=(11, 5 * len(chosen)), squeeze=False)
    records = []
    for row, (name, info) in enumerate(chosen.items()):
        path = args.data / "test" / name
        stem = path.stem
        data = laspy.read(path, laz_backend=laspy.LazBackend.Lazrs)
        sidecar = args.qualified / (stem + "-semantics.json")
        predicted = np.asarray(json.loads(sidecar.read_text())["semantic_ids"])
        truth = targets(data.classification)
        truth[truth < 0] = 7
        order = np.argsort(np.asarray(data.z), kind="stable")
        for col, classes in enumerate([truth, predicted]):
            axes[row, col].scatter(
                np.asarray(data.x)[order] - data.header.mins[0],
                np.asarray(data.y)[order] - data.header.mins[1],
                c=classes[order],
                cmap=ListedColormap(colors),
                vmin=-0.5,
                vmax=7.5,
                s=1,
                linewidths=0,
                rasterized=True,
            )
            axes[row, col].set(
                aspect="equal",
                xlabel="Local easting (m)",
                ylabel="Local northing (m)",
                title=("IGN reference semantics" if col == 0 else "Native FP16 semantic sidecar")
                + f"\n{stem} · {len(data):,} points",
            )
        records.append(
            {
                "source": name,
                "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "output_sha256": info["independent"]["output_sha256"],
                "semantic_sidecar_sha256": hashlib.sha256(sidecar.read_bytes()).hexdigest(),
                "points": len(data),
                "semantic_support": info["semantic_support"],
            }
        )
    fig.suptitle("SynRex BYOM · frozen compact U-Net on held-out real FRACTAL", fontsize=15)
    fig.legend(
        handles=[
            Line2D([0], [0], marker="o", color=c, linestyle="", label=n)
            for c, n in zip(colors, names, strict=False)
        ],
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.022),
    )
    fig.text(
        0.5,
        0.008,
        "IGN FRACTAL · Etalab-2.0 · updated 2025-04-05, retrieved 2026-10-26. "
        "All points; no interpolation.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.065, 1, 0.98))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)
    result = {
        "selection": (
            "Highest reference support for building, water, bridge and permanent structure; "
            "chosen for display after model freeze; all 256 test files scored"
        ),
        "projection": "All original points increasing elevation; plan-view, no interpolation",
        "files": records,
        "sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(),
        "data_license": "Etalab-2.0 IGN FRACTAL; original presentation CC-BY-4.0 Tuomas Pietilä",
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
