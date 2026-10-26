# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
"""Stage/verify a separate indexed COPC view without changing native LAZ outputs."""

import argparse
import hashlib
import json
from pathlib import Path

import laspy
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--classified", type=Path, required=True)
    p.add_argument("--sidecar", type=Path, required=True)
    p.add_argument("--staging", type=Path)
    p.add_argument("--copc", type=Path)
    p.add_argument("--receipt", type=Path, required=True)
    a = p.parse_args()
    if (a.staging is None) == (a.copc is None):
        raise ValueError("Choose staging generation or COPC verification")
    original = laspy.read(a.classified, laz_backend=laspy.LazBackend.Lazrs)
    semantic = json.loads(a.sidecar.read_text())
    if any(
        type(i) is not int or not 0 <= i < len(semantic["las_classes"])
        for i in semantic["semantic_ids"]
    ):
        raise ValueError("invalid full semantic channel ID")
    ids = np.asarray(semantic["semantic_ids"], dtype=np.uint8)
    require(0 < len(ids) == len(original) <= np.iinfo(np.uint32).max, "point count mismatch")
    require(
        np.array_equal(original.classification, np.asarray(semantic["las_classes"])[ids]),
        "classification/semantic mapping mismatch",
    )
    result = dict(
        schema="synrex.qgis_copc_view.v1",
        status="PASS",
        original_classified_sha256=sha(a.classified),
        original_semantic_sidecar_sha256=sha(a.sidecar),
        points=len(original),
        source_order="SynRexSourceIndex is zero-based original LAZ source record index",
        semantic_definition="SynRexSemanticID retains original model output channel",
    )
    if a.staging is not None:
        if a.staging.exists():
            raise FileExistsError(a.staging)
        require(
            not {"SynRexSourceIndex", "SynRexSemanticID"}.intersection(
                original.point_format.dimension_names
            ),
            "reserved SynRex attributes already present",
        )
        original.add_extra_dims(
            [
                laspy.ExtraBytesParams(
                    name="SynRexSourceIndex",
                    type=np.uint32,
                    description="Original source record index",
                ),
                laspy.ExtraBytesParams(
                    name="SynRexSemanticID",
                    type=np.uint8,
                    description="Original model output channel",
                ),
            ]
        )
        original["SynRexSourceIndex"] = np.arange(len(original), dtype=np.uint32)
        original["SynRexSemanticID"] = ids
        original.write(a.staging)
        result["staging_sha256"] = sha(a.staging)
    else:
        view = laspy.read(a.copc, laz_backend=laspy.LazBackend.Lazrs)
        index = np.asarray(view["SynRexSourceIndex"])
        require(len(view) == len(original), "COPC point count mismatch")
        require(
            index.dtype == np.dtype("uint32")
            and np.array_equal(np.sort(index), np.arange(len(original))),
            "source index coverage is not a uint32 permutation",
        )
        if not any(v.user_id == "copc" and v.record_id == 1 for v in view.header.vlrs):
            raise ValueError("COPC info VLR missing")
        order = np.argsort(index)
        restored = np.empty(len(original), dtype=original.points.array.dtype)
        for field in original.points.array.dtype.names:
            restored[field] = view.points.array[field][order]
        require(restored.tobytes() == original.points.array.tobytes(), "point bytes changed")
        require(np.array_equal(view["SynRexSemanticID"][order], ids), "semantic IDs changed")
        require(np.array_equal(view.header.scales, original.header.scales), "scales changed")
        require(np.array_equal(view.header.offsets, original.header.offsets), "offsets changed")
        result.update(
            copc_sha256=sha(a.copc),
            all_original_point_bytes_exact_after_index_restore=True,
            semantic_ids_exact_after_index_restore=True,
            metadata_scope=(
                "COPC viewing derivative with rewritten/index metadata; "
                "original LAZ and full metadata stay separate and unchanged"
            ),
        )
    with a.receipt.open("x") as f:
        f.write(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
