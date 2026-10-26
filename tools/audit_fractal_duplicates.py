# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
"""Detect duplicate selected tiles, including reordered clouds with changed headers."""

import argparse
import hashlib
import json
from pathlib import Path

import laspy
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    root = a.data
    manifest = json.loads((root / "dataset.json").read_text())
    seen, coords, rows = {}, {}, []
    for split, selection in manifest["files"].items():
        for row in selection["tiles"]:
            source = root / split / Path(row["member"]).name
            h = hashlib.sha256(source.read_bytes()).hexdigest()
            if h != row["sha256"] or h in seen:
                raise ValueError(f"source hash mismatch or duplicate: {source.name}")
            seen[h] = str(source.relative_to(root))
            cloud = laspy.read(source, laz_backend=laspy.LazBackend.Lazrs)
            xyz = np.column_stack((np.asarray(cloud.x), np.asarray(cloud.y), np.asarray(cloud.z)))
            xyz = xyz[np.lexsort((xyz[:, 2], xyz[:, 1], xyz[:, 0]))].astype("<f8")
            c = hashlib.sha256(xyz.tobytes()).hexdigest()
            if c in coords:
                raise ValueError(f"duplicate unordered XYZ: {source.name}, {coords[c]}")
            coords[c] = str(source.relative_to(root))
            rows.append(
                dict(
                    split=split,
                    source=source.name,
                    sha256=h,
                    unordered_xyz_sha256=c,
                    points=len(cloud),
                )
            )
    a.out.write_text(
        json.dumps(
            dict(
                status="PASS",
                tiles=len(rows),
                unique_file_hashes=len(seen),
                unique_unordered_xyz_hashes=len(coords),
                files=rows,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
