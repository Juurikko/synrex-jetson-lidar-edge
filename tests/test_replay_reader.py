# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
"""Independent laz-rs checks of label-free bounded native source decoding."""

import hashlib
import json
import os
import struct
import subprocess
from pathlib import Path

import laspy
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
BUILD = Path(os.environ.get("SYNREX_TEST_NATIVE_DIR", ROOT / "build")).resolve()
EXECUTABLE = BUILD / "synrex-replay"


def cloud(path, count, point_format, labels=1, scales=None, offsets=None):
    header = laspy.LasHeader(point_format=point_format, version="1.4")
    header.scales = scales if scales is not None else [0.001, 0.01, 0.0001]
    header.offsets = offsets if offsets is not None else [600000000.0, -400000000.0, 123.4567]
    header.global_encoding.gps_time_type = laspy.header.GpsTimeType.STANDARD
    data = laspy.LasData(header)
    data.X = np.arange(count, dtype=np.int32) * 17 - 132
    data.Y = np.arange(count, dtype=np.int32) * -31 + 19
    data.Z = np.arange(count, dtype=np.int32) * 71 - 9000
    data.intensity = np.arange(count, dtype=np.uint16) * 501 + 53
    data.return_number = np.arange(count, dtype=np.uint8) % 3 + 1
    data.number_of_returns = np.full(count, 3, dtype=np.uint8)
    data.gps_time = 1000000.0 + np.arange(count) * 0.125
    data.classification = np.full(count, labels, dtype=np.uint8)
    data.write(path, laz_backend=laspy.LazBackend.Lazrs)
    return laspy.read(path, laz_backend=laspy.LazBackend.Lazrs)


def manifest(path, sources):
    value = {
        "schema": "synrex.distinct_replay.v1",
        "workload": "file_order_synthetic",
        "boundary_policy": "reset_each_file",
        "sources": [
            {"path": source.name, "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
             "records": records, "coordinate_frame": f"tile_{index}",
             "crs": "UNSPECIFIED:fixture"}
            for index, (source, records) in enumerate(sources)
        ],
    }
    path.write_text(json.dumps(value) + "\n")
    return value


def inspect(path, window=8, receipt=None):
    receipt = receipt or path.parent / "receipt.json"
    completed = subprocess.run(
        [str(EXECUTABLE), "inspect", "--manifest", str(path), "--window-points", str(window),
         "--receipt", str(receipt)], capture_output=True, text=True, timeout=15,
    )
    return completed, json.loads(completed.stdout)


def expected_digests(data):
    features, observations = hashlib.sha256(), hashlib.sha256()
    for index in range(len(data)):
        xyz = [int(data[axis][index]) * scale + offset
               for axis, scale, offset in zip("XYZ", data.header.scales, data.header.offsets,
                                               strict=True)]
        features.update(struct.pack("<dddH", *xyz, int(data.intensity[index])))
        observations.update(struct.pack(
            "<iiiHBBd", int(data.X[index]), int(data.Y[index]), int(data.Z[index]),
            int(data.intensity[index]), int(data.return_number[index]),
            int(data.number_of_returns[index]), float(data.gps_time[index]),
        ))
    return features.hexdigest(), observations.hexdigest()


@pytest.mark.parametrize("point_format", [6, 7, 8])
def test_distinct_reader_independent_precision_attributes_and_final_partial(tmp_path, point_format):
    sources = [tmp_path / "a.laz", tmp_path / "b.laz"]
    counts = [23, 11]
    originals = [cloud(path, count, point_format)
                 for path, count in zip(sources, counts, strict=True)]
    path = tmp_path / "manifest.json"
    manifest(path, list(zip(sources, counts, strict=True)))
    completed, receipt = inspect(path)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert receipt["status"] == "PASS"
    assert receipt["execution"] == "CPU_SOURCE_DECODE_ONLY"
    assert receipt["classification"] == "NOT_RUN"
    assert receipt["records"] == 34
    assert receipt["processed_records"] == receipt["decoded_records"] == 34
    assert receipt["raw_records_decoded"] == receipt["verified_records"] == 34
    assert receipt["windows"] == 5
    assert receipt["output_window_high_water_points"] == 8
    assert receipt["manifest_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert receipt["physical_capture_age_seconds"] is None
    assert receipt["offered_rate_pps"] is None
    assert 0 < receipt["decode_verify_digest_seconds"] <= receipt["elapsed_seconds"] < 15
    for row, original in zip(receipt["sources"], originals, strict=True):
        features, observations = expected_digests(original)
        assert row["runtime_features_sha256"] == features
        assert row["source_observations_sha256"] == observations
        assert row["decoded_records"] == len(original)
        assert row["source_verified_after_decode"]
        assert row["physical_eof_verified"] is False
        assert row["scale"] == list(original.header.scales)
        assert row["offset"] == list(original.header.offsets)
        assert row["global_encoding"] == original.header.global_encoding.value
        assert row["coordinate_frame"] == f"tile_{row['source_index']}"
        assert row["crs"] == "UNSPECIFIED:fixture"


def test_reader_classification_erasure_leaves_all_runtime_fields_identical(tmp_path):
    results = []
    for name, labels in [("original", 1), ("erased", 255)]:
        folder = tmp_path / name
        folder.mkdir()
        source = folder / "cloud.laz"
        cloud(source, 23, 7, labels)
        path = folder / "manifest.json"
        manifest(path, [(source, 23)])
        completed, receipt = inspect(path)
        assert completed.returncode == 0, receipt
        results.append(receipt["sources"][0])
    assert results[0]["source_sha256"] != results[1]["source_sha256"]
    for name in ["runtime_features_sha256", "source_observations_sha256"]:
        assert results[0][name] == results[1][name]


@pytest.mark.parametrize("point_format", [6, 7, 8])
def test_reader_defined_int32_predictor_wrap_matches_independent_decoder(tmp_path, point_format):
    source = tmp_path / "extreme.laz"
    data = cloud(source, 11, point_format, scales=[0.001] * 3, offsets=[0.0] * 3)
    extreme = np.array(
        [-2147483648, 2147483647, -1, 0, 10000] * 2 + [-2147483648], dtype=np.int32,
    )
    data.X, data.Y, data.Z = extreme, extreme[::-1], np.roll(extreme, 2)
    data.write(source, laz_backend=laspy.LazBackend.Lazrs)
    original = laspy.read(source, laz_backend=laspy.LazBackend.Lazrs)
    path = tmp_path / "manifest.json"
    manifest(path, [(source, 11)])
    completed, receipt = inspect(path)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    features, observations = expected_digests(original)
    assert receipt["sources"][0]["runtime_features_sha256"] == features
    assert receipt["sources"][0]["source_observations_sha256"] == observations
    assert receipt["verified_records"] == 11


@pytest.mark.parametrize("failure", ["count", "truncated", "scale"])
def test_reader_incomplete_or_invalid_decode_cannot_claim_success(tmp_path, failure):
    source = tmp_path / "source.laz"
    cloud(source, 23, 8)
    if failure == "truncated":
        raw = source.read_bytes()
        point_offset = struct.unpack_from("<I", raw, 96)[0]
        source.write_bytes(raw[:point_offset + 10])
    if failure == "scale":
        raw = bytearray(source.read_bytes())
        struct.pack_into("<d", raw, 131, float("nan"))
        source.write_bytes(raw)
    path = tmp_path / "manifest.json"
    manifest(path, [(source, 24 if failure == "count" else 23)])
    completed, receipt = inspect(path)
    assert completed.returncode == 1, receipt
    assert receipt["status"] == "FAIL"
    assert receipt["classification"] == "NOT_RUN"
    assert "error" in receipt


@pytest.mark.parametrize("window", [0, 65537, "-1", "8x", "999999999999999999999"])
def test_reader_configuration_is_admitted_before_source(tmp_path, window):
    completed, receipt = inspect(tmp_path / "absent.json", window)
    assert completed.returncode == 1
    assert receipt["status"] == "FAIL"
    assert "window" in receipt["error"]


def test_reader_existing_receipt_is_preserved(tmp_path):
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_bytes(b"prior evidence\n")
    completed, receipt = inspect(tmp_path / "absent.json", receipt=receipt_path)
    assert completed.returncode == 1
    assert receipt["status"] == "FAIL"
    assert receipt_path.read_bytes() == b"prior evidence\n"


@pytest.mark.parametrize("failure", ["count", "partial_window_coordinate"])
def test_reader_failed_second_source_retains_verified_prefix_and_partial_counts(tmp_path, failure):
    first, second = tmp_path / "first.laz", tmp_path / "second.laz"
    cloud(first, 23, 6)
    scales = [25000000.0, 0.01, 0.0001] if failure == "partial_window_coordinate" else None
    data = cloud(second, 23, 8, scales=scales)
    if failure == "partial_window_coordinate":
        data.X = np.arange(23, dtype=np.int32)
        data.write(second, laz_backend=laspy.LazBackend.Lazrs)
    path = tmp_path / "manifest.json"
    manifest(path, [(first, 23), (second, 24 if failure == "count" else 23)])
    completed, receipt = inspect(path)
    assert completed.returncode == 1, completed.stdout + completed.stderr
    assert receipt["status"] == "FAIL" and "records" not in receipt
    assert receipt["verified_records"] == 23
    assert [row["status"] for row in receipt["sources"]] == ["PASS", "FAIL"]
    failed = receipt["sources"][1]
    assert failed["source_verified_after_decode"] is False
    assert failed["runtime_features_sha256"] is None
    assert failed["source_observations_sha256"] is None
    if failure == "count":
        assert receipt["processed_records"] == receipt["decoded_records"] == 23
        assert failed["raw_records_decoded"] == 0
    else:
        assert receipt["processed_records"] == 23 + 16
        assert receipt["decoded_records"] == 23 + 17
        assert receipt["raw_records_decoded"] == 23 + 18
        assert failed["processed_records"] == 16
        assert failed["decoded_records"] == 17
        assert failed["raw_records_decoded"] == 18
