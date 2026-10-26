# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
"""Independent NumPy training/validation implementation of the public dense contract.

Not part of the native execution path. No prior SegFormer source is used here.
"""

import itertools

import laspy
import numpy as np

LAS_CLASSES = np.array([1, 2, 5, 6, 9, 17, 64], dtype=np.uint8)


def targets(raw):
    raw = np.asarray(raw).copy()
    raw[np.isin(raw, [3, 4])] = 5
    result = np.full(raw.shape, -100, dtype=np.int64)
    for index, value in enumerate(LAS_CLASSES):
        result[raw == value] = index
    # IGN sensor artifacts 65/66 and any unknown codes are excluded from scoring.
    return result


def blocks(path, spec):
    data = laspy.read(path, laz_backend=laspy.LazBackend.Lazrs)
    xyz = np.column_stack([np.asarray(data.x), np.asarray(data.y), np.asarray(data.z)])
    minimum = xyz.min(0)
    geometry = spec["geometry"]
    if any(geometry["overlap_xyz"]):
        raise ValueError("training pilot supports the configured nonoverlapping geometry only")
    extent = np.array(geometry["block_size_xyz"])
    dimensions = np.array(geometry["dimensions_xyz"])
    voxel_size = np.array(geometry["voxel_size_xyz"])
    maximum = np.floor((xyz.max(0) - minimum) / extent).astype(int)
    truth = targets(data.classification)
    for index in itertools.product(*(range(int(v) + 1) for v in maximum)):
        origin = minimum + np.array(index) * extent
        local = xyz - origin
        ids = np.flatnonzero(((local >= 0) & (local < extent)).all(1))
        if not len(ids):
            continue
        coordinates = np.minimum(np.floor(local[ids] / voxel_size).astype(int), dimensions - 1)
        linear = (coordinates[:, 2] * dimensions[1] + coordinates[:, 1]) * dimensions[
            0
        ] + coordinates[:, 0]
        size = int(np.prod(dimensions))
        counts = np.bincount(linear, minlength=size)
        occupied = counts > 0
        features = np.zeros((len(spec["features"]), size), dtype=np.float32)
        for channel, f in enumerate(spec["features"]):
            kind = f["kind"]
            if kind == "occupancy":
                value = np.ones(size)
            elif kind == "count":
                value = counts
            else:
                values = (
                    np.asarray(data.intensity)[ids]
                    if kind == "mean_intensity"
                    else local[ids, "xyz".index(kind[-1])]
                )
                value = np.bincount(linear, weights=values, minlength=size) / np.maximum(counts, 1)
            features[channel, occupied] = (
                value[occupied].astype(np.float32) - np.float32(f["offset"])
            ) * np.float32(f["scale"])
        valid = truth[ids] >= 0
        votes = np.bincount(linear[valid] * 7 + truth[ids][valid], minlength=size * 7).reshape(
            size, 7
        )
        labels = votes.argmax(1).astype(np.int64)
        labels[votes.sum(1) == 0] = -100
        yield {
            "features": features.reshape([len(spec["features"]), *dimensions[::-1]]),
            "labels": labels.reshape(dimensions[::-1]),
            "ids": ids,
            "linear": linear,
            "truth": truth[ids],
            "origin": origin.tolist(),
            "points": len(data),
        }


def metrics(confusion):
    confusion = np.asarray(confusion, dtype=np.int64)
    support = confusion.sum(1)
    union = support + confusion.sum(0) - confusion.diagonal()
    iou = np.divide(confusion.diagonal(), union, out=np.full(7, np.nan), where=union > 0)
    return {
        "confusion": confusion.tolist(),
        "support": support.tolist(),
        "iou": [None if not np.isfinite(x) else float(x) for x in iou],
        "mean_iou_present": float(np.nanmean(iou)),
        "ground_vegetation_building_mean_iou": float(np.nanmean(iou[[1, 2, 3]])),
        "accuracy": float(confusion.trace() / max(confusion.sum(), 1)),
    }


def add_confusion(confusion, truth, predicted):
    valid = truth >= 0
    confusion += np.bincount(truth[valid] * 7 + predicted[valid], minlength=49).reshape(7, 7)
