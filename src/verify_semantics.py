# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
"""Independent source/model-bound semantic sidecar and classified LAZ verification."""

import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np

from verify_laz import verify


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_semantics(source, output, sidecar, package, binding=None, taxonomy=None):
    source, output, sidecar, package = map(Path, (source, output, sidecar, package))
    manifest_path = package / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    info = json.loads(sidecar.read_text())
    if info["schema"] != "synrex.point_semantics.v1":
        raise ValueError("unsupported semantic sidecar")
    if info["source_sha256"] != digest(source) or info["model_manifest_sha256"] != digest(
        manifest_path
    ):
        raise ValueError("source/model binding mismatch")
    mapping = manifest["contract"]["las_classes"]
    if (
        info["las_classes"] != mapping
        or not 2 <= len(mapping) <= 256
        or any(type(i) is not int or not 0 <= i <= 255 for i in mapping)
    ):
        raise ValueError("semantic mapping mismatch")
    ids = info["semantic_ids"]
    if (
        type(info["point_count"]) is not int
        or len(ids) != info["point_count"]
        or not 0 < len(ids) <= 20000000
    ):
        raise ValueError("semantic point count mismatch")
    if info["index_order"] != "original source point record order":
        raise ValueError("unsupported semantic point order")
    if any(type(i) is not int or not 0 <= i < len(mapping) for i in ids):
        raise ValueError("invalid semantic channel ID")
    for entry in manifest["artifacts"].values():
        path = package / entry["path"]
        if path.name != entry["path"] or path.is_symlink() or digest(path) != entry["sha256"]:
            raise ValueError("package artifact mismatch")
    labels = np.asarray(mapping, dtype=np.uint8)[np.asarray(ids, dtype=np.uint8)].tobytes()
    if hashlib.sha256(labels).hexdigest() != info["mapped_labels_sha256"]:
        raise ValueError("mapped labels digest mismatch")
    with tempfile.TemporaryDirectory(prefix="synrex-semantic-verify-") as scratch:
        expected = Path(scratch) / "labels.u8"
        expected.write_bytes(labels)
        receipt = verify(source, output, expected)
    if binding is not None:
        envelope = json.loads(Path(binding).read_text())
        expected = {
            "schema": "synrex.semantic_output_binding.v1",
            "source_sha256": digest(source),
            "output_sha256": digest(output),
            "semantic_sidecar_sha256": digest(sidecar),
            "model_manifest_sha256": digest(manifest_path),
            "points": len(ids),
            "index_order": info["index_order"],
        }
        if taxonomy is None or envelope.get("taxonomy_sha256") != digest(Path(taxonomy)):
            raise ValueError("taxonomy binding mismatch")
        if any(envelope.get(key) != value for key, value in expected.items()):
            raise ValueError("semantic output binding mismatch")
        receipt["output_hash_binding_verified"] = True
    receipt.update(
        semantic_schema=info["schema"],
        semantic_sidecar_sha256=digest(sidecar),
        semantic_model_binding=True,
        full_model_channel_ids_preserved=True,
    )
    return receipt
