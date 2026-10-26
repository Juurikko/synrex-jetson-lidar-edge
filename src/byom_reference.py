# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
"""Original dense synthetic scene and independent development-only ONNX CPU oracle.

The native classifier never imports this module. The oracle uses brute-force block
enumeration, independently of the native point-driven membership algorithm.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import laspy
import numpy as np
import onnxruntime as ort

from synthetic_laz import create


def create_dense_scene(directory: Path, points: int = 50013) -> Path:
    if points < 23:
        raise ValueError("at least 23 fixture points required")
    source, _ = create(directory / "record-fixture", points)
    data = laspy.read(source, laz_backend=laspy.LazBackend.Lazrs)
    rng = np.random.default_rng(20261027)
    # Invented compact geometry with shuffled source order, exact tile boundaries,
    # coincident source records and feature bands that survive voxel averaging.
    data.X = rng.integers(0, 1400, points, dtype=np.int32)
    data.Y = rng.integers(0, 900, points, dtype=np.int32)
    data.Z = rng.integers(0, 900, points, dtype=np.int32)
    data.X[:8] = [0, 700, 700, 800, 600, 1200, 300, 1300]
    data.Y[:8] = [0, 0, 0, 600, 0, 0, 100, 100]
    data.Z[:8] = [0, 0, 0, 400, 0, 0, 100, 100]
    data.intensity = np.where(data.X < 400, 0, np.where(data.X < 900, 32768, 65535)).astype(
        np.uint16
    )
    output = directory / "dense-synthetic.laz"
    if output.exists():
        raise FileExistsError(output)
    data.write(output, laz_backend=laspy.LazBackend.Lazrs)
    return output


def expected_labels(source: Path, spec: dict, model: Path) -> tuple[np.ndarray, dict]:
    """Brute-force NumPy geometry + real ONNX Runtime CPU inference; qualification only."""
    data = laspy.read(source, laz_backend=laspy.LazBackend.Lazrs)
    xyz = np.column_stack([np.asarray(data.x), np.asarray(data.y), np.asarray(data.z)])
    minimum = xyz.min(axis=0)
    g = spec["geometry"]
    extent = np.asarray(g["block_size_xyz"])
    stride = extent * (1 - np.asarray(g["overlap_xyz"]))
    dims = np.asarray(g["dimensions_xyz"])
    maximum = np.floor((xyz.max(axis=0) - minimum) / stride).astype(int)
    labels = np.full(len(xyz), -1, dtype=np.int32)
    session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    blocks, memberships = 0, 0
    for index in itertools.product(*(range(int(v) + 1) for v in maximum)):
        local = xyz - (minimum + np.asarray(index) * stride)
        ids = np.flatnonzero(np.all((local >= 0) & (local < extent), axis=1))
        if not len(ids):
            continue
        blocks += 1
        memberships += len(ids)
        vox = np.minimum(
            np.floor(local[ids] / np.asarray(g["voxel_size_xyz"])).astype(int), dims - 1
        )
        linear = (vox[:, 2] * dims[1] + vox[:, 1]) * dims[0] + vox[:, 0]
        tensor = np.zeros(spec["input"]["shape"], dtype=spec["input"]["dtype"])
        flat = tensor.reshape(len(spec["features"]), -1)
        for voxel in np.unique(linear):
            selected = ids[linear == voxel]
            values = {
                "occupancy": 1,
                "count": len(selected),
                "mean_intensity": np.asarray(data.intensity)[selected].astype(np.float64).mean(),
                "local_x": local[selected, 0].mean(),
                "local_y": local[selected, 1].mean(),
                "local_z": local[selected, 2].mean(),
            }
            for channel, feature in enumerate(spec["features"]):
                flat[channel, voxel] = (
                    np.float32(values[feature["kind"]]) - np.float32(feature["offset"])
                ) * np.float32(feature["scale"])
        scores = session.run([spec["output"]["name"]], {spec["input"]["name"]: tensor})[0]
        scores = scores.reshape(len(spec["las_classes"]), -1)
        prediction = np.asarray(spec["las_classes"], dtype=np.uint8)[
            scores[:, linear].argmax(axis=0)
        ]
        if g["overlap_policy"] == "first_wins":
            take = labels[ids] < 0
            labels[ids[take]] = prediction[take]
        else:
            labels[ids] = prediction
    if not (labels >= 0).all():
        raise ValueError("oracle left source points uncovered")
    result = labels.astype(np.uint8)
    unique, counts = np.unique(result, return_counts=True)
    return result, {
        "backend": "ONNX Runtime CPU independent oracle, not native execution",
        "points": len(xyz),
        "blocks": blocks,
        "memberships": memberships,
        "class_histogram": {str(k): int(v) for k, v in zip(unique, counts, strict=True)},
    }
