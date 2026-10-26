# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
"""Verify training features against known observations and native source identities."""

import json
import subprocess

import laspy
import numpy as np
from test_byom import NATIVE, ROOT, package

from dense_training import blocks, targets


def test_fractal_artifacts_are_ignored_and_vegetation_is_merged():
    actual = targets(np.array([1, 2, 3, 4, 5, 6, 9, 17, 64, 65, 66, 255]))
    assert actual.tolist() == [0, 1, 2, 2, 2, 3, 4, 5, 6, -100, -100, -100]


def test_training_geometry_source_identity_and_measured_features(tmp_path):
    spec = json.loads((ROOT / "examples/unet/dense_voxel.json").read_text())
    header = laspy.LasHeader(point_format=8, version="1.4")
    header.scales = [0.01, 0.01, 0.01]
    data = laspy.LasData(header)
    data.x = [33.25, 0, 0.5, 1, 32, 0.25]
    data.y = [0, 0, 0.5, 0, 0, 0.25]
    data.z = [0, 0, 0.5, 32, 0, 0.25]
    data.intensity = np.array([100, 1000, 3000, 200, 400, 2000], dtype=np.uint16)
    data.classification = np.array([6, 2, 3, 17, 64, 2], dtype=np.uint8)
    source = tmp_path / "fractal-shaped.laz"
    data.write(source, laz_backend=laspy.LazBackend.Lazrs)
    prepared = list(blocks(source, spec))
    # A model package only supplies the geometry contract here; no GPU inference.
    pkg, _, _ = package(tmp_path)
    manifest = pkg / "manifest.json"
    obj = json.loads(manifest.read_text())
    obj["contract"] = spec
    manifest.write_text(json.dumps(obj))
    output = tmp_path / "plan.json"
    subprocess.run(
        [str(NATIVE), "plan", "--model", str(pkg), "--input", str(source), "--out", str(output)],
        check=True,
    )
    actual = json.loads(output.read_text())["blocks"]
    expected = []
    for b in prepared:
        members = sorted(
            zip(b["ids"].tolist(), b["linear"].tolist(), strict=True), key=lambda p: (p[1], p[0])
        )
        expected.append({"origin": b["origin"], "members": [list(p) for p in members]})
    assert actual == expected
    first = prepared[0]
    features = first["features"].reshape(4, -1)
    np.testing.assert_array_equal(
        features[:, 0], np.array([1, 3 / 16, 0.25 / 32, 2000 / 65535], dtype=np.float32)
    )
    assert first["labels"].reshape(-1)[0] == 1  # majority ground, vegetation minority
    assert np.all(features[:, 1] == 0)  # empty remains zero, no label feature leakage
    assert set(np.concatenate([b["ids"] for b in prepared])) == set(range(6))


def test_engine_precision_rejects_unknown_policy(tmp_path):
    pkg, _, _ = package(tmp_path)
    engine = tmp_path / "bad.engine"
    result = subprocess.run(
        [
            str(NATIVE),
            "build-engine",
            "--model",
            str(pkg),
            "--out",
            str(engine),
            "--precision",
            "int8",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0 and "unsupported engine precision" in result.stderr
    assert not engine.exists()


def test_features_do_not_depend_on_reference_labels(tmp_path):
    spec = json.loads((ROOT / "examples/unet/dense_voxel.json").read_text())
    data = laspy.LasData(laspy.LasHeader(point_format=8, version="1.4"))
    data.x, data.y, data.z = [0, 0.25, 1.5], [0, 0.5, 2.5], [0, 0.75, 3.5]
    data.intensity = np.array([100, 200, 300], dtype=np.uint16)
    data.classification = np.array([2, 5, 6], dtype=np.uint8)
    a = tmp_path / "a.laz"
    data.write(a, laz_backend=laspy.LazBackend.Lazrs)
    data.classification = np.array([64, 9, 17], dtype=np.uint8)
    b = tmp_path / "b.laz"
    data.write(b, laz_backend=laspy.LazBackend.Lazrs)
    aa, bb = list(blocks(a, spec)), list(blocks(b, spec))
    assert len(aa) == len(bb)
    for x, y in zip(aa, bb, strict=True):
        np.testing.assert_array_equal(x["features"], y["features"])
        np.testing.assert_array_equal(x["ids"], y["ids"])
    assert any(np.any(x["labels"] != y["labels"]) for x, y in zip(aa, bb, strict=True))
