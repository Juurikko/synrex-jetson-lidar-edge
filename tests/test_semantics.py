# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
"""Independent semantic binding/identity tests using supplied-label writer fixtures."""

import hashlib
import json
import subprocess

import numpy as np
import pytest
from test_byom import package
from test_preservation import WRITER

from synthetic_laz import create
from verify_semantics import verify_semantics


@pytest.mark.parametrize(
    "corruption", [None, "source", "model", "ids", "count", "order", "mapping"]
)
def test_semantic_binding_and_invalid_sidecars(tmp_path, corruption):
    source, _ = create(tmp_path, 257)
    pkg, _, _ = package(tmp_path)
    ids = (np.arange(257) % 3).astype(np.uint8)
    labels = np.asarray([2, 5, 6], dtype=np.uint8)[ids].tobytes()
    label_file = tmp_path / "supplied.u8"
    label_file.write_bytes(labels)
    output = tmp_path / "out.laz"
    subprocess.run(
        [str(WRITER), str(source), str(label_file), str(output)], check=True, capture_output=True
    )
    doc = {
        "schema": "synrex.point_semantics.v1",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "model_manifest_sha256": hashlib.sha256((pkg / "manifest.json").read_bytes()).hexdigest(),
        "point_count": 257,
        "index_order": "original source point record order",
        "las_classes": [2, 5, 6],
        "semantic_ids": ids.tolist(),
        "mapped_labels_sha256": hashlib.sha256(labels).hexdigest(),
    }
    if corruption == "source":
        doc["source_sha256"] = "0" * 64
    elif corruption == "model":
        doc["model_manifest_sha256"] = "0" * 64
    elif corruption == "ids":
        doc["semantic_ids"][1] = 1.5
    elif corruption == "count":
        doc["point_count"] = 257.0
    elif corruption == "order":
        doc["index_order"] = "reversed"
    elif corruption == "mapping":
        doc["las_classes"] = [2, 5, 201]
    sidecar = tmp_path / "semantics.json"
    sidecar.write_text(json.dumps(doc))
    if corruption:
        with pytest.raises(ValueError):
            verify_semantics(source, output, sidecar, pkg)
    else:
        result = verify_semantics(source, output, sidecar, pkg)
        assert result["points"] == 257 and result["full_model_channel_ids_preserved"]


def test_output_binding_rejects_tampering(tmp_path):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from seal_semantics import seal

    test_semantic_binding_and_invalid_sidecars(tmp_path, None)
    source = next(tmp_path.glob("*.laz"))
    # Use the exact source recorded in the native-independent fixture sidecar.
    doc = json.loads((tmp_path / "semantics.json").read_text())
    source = next(
        p
        for p in tmp_path.glob("*.laz")
        if hashlib.sha256(p.read_bytes()).hexdigest() == doc["source_sha256"]
    )
    pkg = next(tmp_path.glob("*.srxmodel"))
    taxonomy = tmp_path / "taxonomy.json"
    taxonomy.write_text(
        json.dumps(
            {
                "classes": [
                    {"semantic_id": i, "output_las_code": code} for i, code in enumerate([2, 5, 6])
                ]
            }
        )
    )
    output, sidecar, binding = (
        tmp_path / "out.laz",
        tmp_path / "semantics.json",
        tmp_path / "bind.json",
    )
    seal(source, output, sidecar, pkg, taxonomy, binding)
    assert verify_semantics(source, output, sidecar, pkg, binding, taxonomy)[
        "output_hash_binding_verified"
    ]
    envelope = json.loads(binding.read_text())
    envelope["output_sha256"] = "0" * 64
    binding.write_text(json.dumps(envelope))
    with pytest.raises(ValueError, match="output binding"):
        verify_semantics(source, output, sidecar, pkg, binding, taxonomy)
