# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
"""Native streaming CLI integration; no Python implementation of production inference."""

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "build/synrex-stream"
FIXTURE = ROOT / "examples/streaming/deterministic.jsonl"


def invoke(*args):
    return subprocess.run([str(NATIVE), *map(str, args)], capture_output=True, text=True)


def test_checked_fixture():
    result = invoke("validate", "--input", FIXTURE)
    assert result.returncode == 0, result.stderr
    receipt = json.loads(result.stdout)
    assert receipt["status"] == "PASS"
    assert receipt["assembly"]["incoming_points"] == 14
    assert receipt["assembly"]["assembled_points"] == 14
    assert receipt["GPU_INFERENCE"] == "NOT_RUN"


def test_fixture_is_exactly_reproducible_and_refuses_overwrite(tmp_path):
    output = tmp_path / "fixture.jsonl"
    args = (
        "fixture",
        "--out",
        output,
        "--frames",
        "2",
        "--points",
        "7",
        "--chunk",
        "4",
        "--seed",
        "7",
        "--taxonomy",
        "synrex.fractal7.v1",
    )
    assert invoke(*args).returncode == 0
    assert output.read_bytes() == FIXTURE.read_bytes()
    assert invoke(*args).returncode != 0
    assert output.read_bytes() == FIXTURE.read_bytes()


def test_incomplete_frame_cannot_pass(tmp_path):
    output = tmp_path / "partial.jsonl"
    output.write_text(FIXTURE.read_text().splitlines()[0] + "\n")
    result = invoke("validate", "--input", output)
    assert result.returncode != 0
    receipt = json.loads(result.stdout)
    assert receipt["status"] == "FAIL"
    assert receipt["assembly"]["partial_frames"] == 1
    assert receipt["assembly"]["dropped_points"] == 4


def test_unknown_schema_cannot_pass(tmp_path):
    output = tmp_path / "unknown.jsonl"
    record = json.loads(FIXTURE.read_text().splitlines()[0])
    record["schema"] = "synrex.point_stream.v999"
    output.write_text(json.dumps(record) + "\n")
    result = invoke("validate", "--input", output)
    assert result.returncode != 0
    assert json.loads(result.stderr)["status"] == "FAIL"


def test_duplicate_cli_option_rejected(tmp_path):
    result = invoke("fixture", "--out", tmp_path / "unused", "--frames", "2", "--frames", "3")
    assert result.returncode != 0
    assert not (tmp_path / "unused").exists()
