# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
"""Fixed, label-free airborne geometry baseline; deliberately not a learned model."""

import laspy
import numpy as np
from scipy.ndimage import minimum_filter


def predict(path):
    data = laspy.read(path, laz_backend=laspy.LazBackend.Lazrs)
    xyz = np.column_stack([np.asarray(data.x), np.asarray(data.y), np.asarray(data.z)])
    xyz -= xyz.min(0)

    def grid(width):
        xy = np.floor(xyz[:, :2] / width).astype(int)
        shape = xy.max(0) + 1
        return xy[:, 1] * shape[0] + xy[:, 0], shape

    cells, shape = grid(5.0)
    ground = np.full(int(np.prod(shape)), np.inf)
    np.minimum.at(ground, cells, xyz[:, 2])
    terrain = minimum_filter(ground.reshape(shape[::-1]), size=5, mode="nearest").reshape(-1)
    height = xyz[:, 2] - terrain[cells]
    fine, dimensions = grid(2.0)
    count = np.bincount(fine, minlength=int(np.prod(dimensions)))
    mean = np.bincount(fine, weights=xyz[:, 2], minlength=len(count)) / np.maximum(count, 1)
    square = np.bincount(fine, weights=xyz[:, 2] ** 2, minlength=len(count)) / np.maximum(count, 1)
    deviation = np.sqrt(np.maximum(square - mean**2, 0))[fine]
    output = np.full(len(data), 2, dtype=np.int64)  # vegetation
    output[height <= 0.5] = 1  # ground
    output[(height > 2) & (deviation < 0.5)] = 3  # building, flat elevated surfaces
    return output
