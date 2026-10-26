# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
# ruff: noqa: E402
"""Independent seven-class release projection, binding and LAZ audit (laz-rs)."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import laspy
import numpy as np

from verify_semantics import verify_semantics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["data", "qualified", "taxonomy", "out"]:
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    # Independent explicit release policy; not inferred from the model's own lookup.
    expected_codes = [1, 2, 200, 6, 9, 17, 201]
    expected_sources = [[1], [2], [3, 4, 5], [6], [9], [17], [64]]
    taxonomy = json.loads(a.taxonomy.read_text())
    assert [r["semantic_id"] for r in taxonomy["classes"]] == list(range(7))
    assert [r["source_las_codes"] for r in taxonomy["classes"]] == expected_sources
    assert [r["output_las_code"] for r in taxonomy["classes"]] == expected_codes
    assert all(64 <= c <= 255 for c in expected_codes if c in [200, 201])
    counts = np.zeros(7, np.int64)
    rows = []
    selection = json.loads((a.data / "dataset.json").read_text())["files"]["test"]["tiles"]
    for row in selection:
        source = a.data / "test" / Path(row["member"]).name
        stem = source.stem
        sidecar = a.qualified / (stem + "-semantics.json")
        output = a.qualified / (stem + "-classified.laz")
        receipt = verify_semantics(
            source,
            output,
            sidecar,
            a.qualified / "unet.srxmodel",
            a.qualified / (stem + "-binding.json"),
            a.taxonomy,
        )
        ids = np.asarray(json.loads(sidecar.read_text())["semantic_ids"], dtype=np.uint8)
        cloud = laspy.read(output, laz_backend=laspy.LazBackend.Lazrs)
        assert np.array_equal(np.asarray(cloud.classification), np.asarray(expected_codes)[ids])
        counts += np.bincount(ids, minlength=7)
        rows.append(dict(source=source.name, **receipt))
    assert np.all(counts > 0)
    result = dict(
        status="PASS",
        files=rows,
        predicted_semantic_counts=counts.tolist(),
        output_las_codes=expected_codes,
        all_seven_model_classes_preserved=True,
        bridge_policy=(
            "Semantic ID5 is IGN bridge; LAS17 is a bridge-deck projection, "
            "not an assertion of subcomponent accuracy"
        ),
        user_defined_codes={
            "200": "aggregate vegetation",
            "201": "permanent aboveground structure",
        },
    )
    a.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
