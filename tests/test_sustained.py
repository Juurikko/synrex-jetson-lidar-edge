# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
"""CPU-only temporal protocol regressions; no inference or hardware qualification claims."""

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "build/synrex-stream"


def validate(tmp_path, records):
    source = tmp_path / "stream.jsonl"
    source.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    result = subprocess.run(
        [str(NATIVE), "validate", "--input", str(source)], capture_output=True, text=True
    )
    return result, json.loads(result.stdout or result.stderr)


def records(tmp_path):
    path = tmp_path / "fixture.jsonl"
    subprocess.run(
        [
            str(NATIVE),
            "fixture",
            "--out",
            str(path),
            "--frames",
            "4",
            "--points",
            "16",
            "--chunk",
            "4",
        ],
        check=True,
        capture_output=True,
    )
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_entire_missing_frame_is_visible(tmp_path):
    source = records(tmp_path)
    result, report = validate(tmp_path, source[:4] + source[8:])
    assert result.returncode != 0
    assert report["assembly"]["missing_sequences"] == 4
    assert report["status"] == "FAIL"


def test_duplicate_packets_do_not_duplicate_classifiable_points(tmp_path):
    source = records(tmp_path)
    result, report = validate(tmp_path, source[:3] + [source[2]] + source[3:])
    assert result.returncode != 0
    assert report["assembly"]["assembled_points"] == 64
    assert report["assembly"]["incoming_points"] == 68
    assert report["assembly"]["late_points"] == 4
    assert report["assembly"]["dropped_points"] == 4


def test_out_of_order_packets_are_counted(tmp_path):
    source = records(tmp_path)
    source[1], source[2] = source[2], source[1]
    result, report = validate(tmp_path, source)
    assert result.returncode != 0
    assert report["assembly"]["partial_frames"] > 0
    # The swapped packet and unassemblable frame tail are all rejected as late.
    assert report["assembly"]["late_points"] == 12
    assert report["assembly"]["incoming_points"] == (
        report["assembly"]["assembled_points"] + report["assembly"]["dropped_points"]
    )


def test_motion_and_source_overlap_preserve_protocol_identity(tmp_path):
    source = records(tmp_path)
    for r in source:
        for point in r["points"]:
            point["xyz"][0] += r["frame_id"] * 0.125
    result, report = validate(tmp_path, source)
    assert result.returncode == 0
    assert report["assembly"]["assembled_points"] == 64


def test_nonmonotonic_frame_time_fails(tmp_path):
    source = records(tmp_path)
    for r in source[4:8]:
        r["frame_start_ns"] -= 200_000_000
        r["frame_end_ns"] -= 200_000_000
        for point in r["points"]:
            point["timestamp_ns"] -= 200_000_000
    result, report = validate(tmp_path, source)
    assert result.returncode != 0
    assert report["assembly"]["late_points"] == 16


def test_sustained_cli_rejects_unsafe_configuration_before_gpu(tmp_path):
    result = subprocess.run(
        [
            str(ROOT / "build/synrex-sustain"),
            "--model",
            "unused",
            "--rate",
            "-1",
            "--receipt",
            str(tmp_path / "receipt.json"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert json.loads(result.stderr)["status"] == "FAIL"
