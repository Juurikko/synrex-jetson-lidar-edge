# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
# Developer and project lead: Tuomas Pietilä.
# AI-assisted software engineering, including OpenAI Codex.
"""Invented records and rule-assigned labels; no pretrained model or real survey data."""

from __future__ import annotations

import datetime
from pathlib import Path

import laspy
import numpy as np


def create(directory: Path, points: int = 257, seed: int = 20261026) -> tuple[Path, Path]:
    if points < 1:
        raise ValueError("positive point count required")
    directory.mkdir(parents=True, exist_ok=True)
    source, labels = directory / "synthetic.laz", directory / "labels.u8"
    if source.exists() or labels.exists():
        raise FileExistsError("synthetic outputs must not exist")
    rng = np.random.default_rng(seed)
    header = laspy.LasHeader(point_format=7, version="1.4")
    header.creation_date = datetime.date(2026, 10, 26)
    header.system_identifier = "SynRex invented synthetic data"
    header.generating_software = "SynRex synthetic integrity test"
    header.scales = [0.01, 0.01, 0.01]
    header.offsets = [1000.0, 2000.0, 30.0]
    for name in ["Amplitude", "Reflectance", "Deviation"]:
        header.add_extra_dim(laspy.ExtraBytesParams(name=name, type=np.int16))
    header.vlrs.append(
        laspy.VLR(
            "SynRexSynthetic", 42, "Invented scene notice", b"No real coordinates or model output"
        )
    )
    header.global_encoding.wkt = True
    header.vlrs.append(
        laspy.vlrs.known.WktCoordinateSystemVlr(
            'LOCAL_CS["SynRex invented local frame",LOCAL_DATUM["Synthetic",0],'
            'UNIT["metre",1],AXIS["x",EAST],AXIS["y",NORTH]]'
        )
    )
    header.extra_vlr_bytes = b"synthetic-padding\x00\xff\x01"
    data = laspy.LasData(header)
    for axis in ["X", "Y", "Z"]:
        setattr(data, axis, rng.integers(-10000, 10000, points, dtype=np.int32))
    for name in ["intensity", "point_source_id", "red", "green", "blue"]:
        setattr(data, name, rng.integers(0, 65536, points, dtype=np.uint16))
    data.gps_time = rng.uniform(1_000_000.0, 2_000_000.0, points)
    data.scan_angle = rng.integers(-5000, 5000, points, dtype=np.int16)
    data.return_number = rng.integers(1, 4, points, dtype=np.uint8)
    data.number_of_returns = np.full(points, 3, dtype=np.uint8)
    data.scanner_channel = rng.integers(0, 4, points, dtype=np.uint8)
    for name in [
        "synthetic",
        "key_point",
        "withheld",
        "overlap",
        "scan_direction_flag",
        "edge_of_flight_line",
    ]:
        setattr(data, name, rng.integers(0, 2, points, dtype=np.uint8))
    for name in ["Amplitude", "Reflectance", "Deviation"]:
        setattr(data, name, rng.integers(-30000, 30000, points, dtype=np.int16))
    data.classification = np.ones(points, dtype=np.uint8)
    data.write(source, laz_backend=laspy.LazBackend.Lazrs)
    # A synthetic height rule exercises class-byte replacement; this is not inference.
    predicted = np.where(data.Z >= 0, 6, 2).astype(np.uint8)
    labels.write_bytes(predicted.tobytes())
    return source, labels
