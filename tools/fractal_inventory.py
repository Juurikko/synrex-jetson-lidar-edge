# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
# ruff: noqa: E402
"""Audit selected FRACTAL classes, input format and cross-split geographic overlap.

Invoke on train/validation before training; do not open reserved test data early.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import laspy
import numpy as np

from dense_training import targets
from verify_laz import metadata_regions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--splits", nargs="+", default=["train", "val"], choices=["train", "val", "test"]
    )
    args = parser.parse_args()
    selected = json.loads((args.data / "dataset.json").read_text())
    receipt = {
        "status": "PASS",
        "transformation": "NONE; exact original compressed source retained",
        "semantic_channel_order": [1, 2, 5, 6, 9, 17, 64],
        "splits": {},
    }
    for split in args.splits:
        rows, semantic_counts, raw_counts = [], np.zeros(7, dtype=np.int64), {}
        ignored = 0
        for record in selected["files"][split]["tiles"]:
            path = args.data / split / Path(record["member"]).name
            assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
            metadata_regions(path)  # independent fail-closed format contract
            data = laspy.read(path, laz_backend=laspy.LazBackend.Lazrs)
            truth = targets(data.classification)
            valid = truth >= 0
            count = np.bincount(truth[valid], minlength=7)
            semantic_counts += count
            ignored += int(np.count_nonzero(~valid))
            for code, n in zip(*np.unique(data.classification, return_counts=True), strict=True):
                raw_counts[str(int(code))] = raw_counts.get(str(int(code)), 0) + int(n)
            rows.append(
                {
                    "source_path": record["member"],
                    "sha256": record["sha256"],
                    "points": len(data),
                    "semantic_counts": count.tolist(),
                    "min_xy": data.header.mins[:2].tolist(),
                    "max_xy": data.header.maxs[:2].tolist(),
                    "las_version": str(data.header.version),
                    "point_format": data.header.point_format.id,
                    "record_bytes": data.header.point_format.size,
                }
            )
        receipt["splits"][split] = {
            "tiles": len(rows),
            "points": sum(r["points"] for r in rows),
            "semantic_point_counts": semantic_counts.tolist(),
            "raw_las_class_counts": raw_counts,
            "ignored_artifacts": ignored,
            "files": rows,
        }
    overlap = []
    for i, a in enumerate(args.splits):
        for b in args.splits[i + 1 :]:
            for x in receipt["splits"][a]["files"]:
                for y in receipt["splits"][b]["files"]:
                    if np.all(
                        np.minimum(x["max_xy"], y["max_xy"]) > np.maximum(x["min_xy"], y["min_xy"])
                    ):
                        overlap.append([x["source_path"], y["source_path"]])
    if overlap:
        raise ValueError(f"cross-split positive-area XY overlap: {overlap[:5]}")
    receipt["cross_split_positive_area_xy_overlap"] = False
    args.out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        json.dumps(
            {k: {a: b for a, b in v.items() if a != "files"} for k, v in receipt["splits"].items()},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
