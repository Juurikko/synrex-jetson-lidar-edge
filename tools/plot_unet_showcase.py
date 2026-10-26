# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
"""Plot genuine held-out IGN classes and native classified points, without smoothing.

Requires separate visualization dependencies matplotlib 3.10.8, laspy, lazrs, numpy.
Plot output carries IGN Etalab-2.0 attribution; code remains MIT.
"""

import argparse
import hashlib
import json
from pathlib import Path

import laspy
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--classified", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted((args.data / "test").glob("*.laz"))
    # Explicit reproducible display selection, never used for model selection or metrics.
    most_buildings = max(paths, key=lambda p: np.count_nonzero(laspy.read(p).classification == 6))
    selected = list(dict.fromkeys([paths[0], most_buildings]))
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
    codes = [1, 2, 5, 6, 9, 17, 64]
    names = [
        "Other (1)",
        "Ground (2)",
        "Vegetation (5)",
        "Building (6)",
        "Water (9)",
        "Bridge (17)",
        "IGN structure (64)",
    ]
    cmap = ListedColormap(colors)
    fig, axes = plt.subplots(len(selected), 2, figsize=(11, 5 * len(selected)), squeeze=False)
    records = []
    for row, source in enumerate(selected):
        data = laspy.read(source, laz_backend=laspy.LazBackend.Lazrs)
        output = args.classified / (source.stem + "-classified.laz")
        predicted = laspy.read(output, laz_backend=laspy.LazBackend.Lazrs)
        order = np.argsort(np.asarray(data.z), kind="stable")
        # All points, sorted by elevation for plan-view visibility; no interpolation.
        for col, labels in enumerate(
            [np.asarray(data.classification), np.asarray(predicted.classification)]
        ):
            labels = labels.copy()
            labels[np.isin(labels, [3, 4])] = 5
            classes = np.full(len(labels), 7)
            for index, code in enumerate(codes):
                classes[labels == code] = index
            axes[row, col].scatter(
                np.asarray(data.x)[order] - data.header.mins[0],
                np.asarray(data.y)[order] - data.header.mins[1],
                c=classes[order],
                cmap=cmap,
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
                title=("IGN ground truth" if col == 0 else "Native TensorRT FP16 U-Net")
                + f"\n{source.stem} · {len(data):,} points",
            )
        records.append(
            {
                "source": source.name,
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "points": len(data),
            }
        )
    fig.suptitle("SynRex BYOM: real held-out FRACTAL LiDAR", fontsize=15)
    handles = [
        Line2D([0], [0], color=c, marker="o", linestyle="", label=n)
        for c, n in zip(colors, names, strict=False)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.035))
    fig.text(
        0.5,
        0.012,
        "Data: IGN FRACTAL · Etalab-2.0 · source updated 2025-04-05, retrieved 2026-10-26. "
        "Pilot; rare classes unqualified.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.09, 1, 0.96))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)
    receipt = {
        "selection": (
            "first sorted test tile and tile with most ground-truth building points; "
            "all 32 tiles scored in qualification"
        ),
        "projection": "all source points, increasing elevation, local XY; no interpolation",
        "files": records,
        "sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(),
        "data_license": "Etalab-2.0, IGN FRACTAL; original presentation CC-BY-4.0 Tuomas Pietilä",
    }
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
