# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
"""Explicit v3 is file-only until a generic streaming state path is implemented."""

import copy
import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

from test_taxonomy_package import SPEC, cpu_package, v3_package

ROOT = Path(__file__).resolve().parents[1]
NATIVE = Path(os.environ.get("SYNREX_TEST_NATIVE_DIR", ROOT / "build")).resolve()
GUARD = "package v3 is file-only; legacy streaming requires package v2"
MAPPINGS = [[0, 2, 5, 6], [1, 2, 5, 6, 9, 17, 64], [2, 5, 6]]


def invoke(executable, *arguments):
    return subprocess.run(
        [str(NATIVE / executable), *map(str, arguments)],
        capture_output=True, text=True, timeout=15,
    )


def assert_guard(result, tmp_path):
    assert result.returncode != 0, result.stdout
    diagnostic = result.stdout + result.stderr
    assert GUARD in diagnostic, diagnostic
    assert "backend not built" not in diagnostic
    assert "_ready=1" not in result.stdout
    assert not (tmp_path / "bundle").exists()
    assert not (tmp_path / "events.jsonl").exists()
    for receipt in tmp_path.glob("*receipt.json"):
        assert json.loads(receipt.read_text())["status"] != "PASS"


@pytest.mark.parametrize("mapping", MAPPINGS)
@pytest.mark.parametrize("mode", ["receive", "send", "subscribe", "subscribe-export"])
def test_every_live_legacy_endpoint_rejects_v3_before_source_socket_or_export(
    tmp_path, mapping, mode
):
    model, _, _ = v3_package(tmp_path, mapping)
    command = "subscribe" if mode == "subscribe-export" else mode
    arguments = [command, "--model", model, "--receipt", tmp_path / "live-receipt.json"]
    if mode in ("send", "subscribe-export"):
        arguments += ["--source", tmp_path / "absent-source.laz"]
    if mode == "subscribe-export":
        arguments += ["--taxonomy", tmp_path / "absent-taxonomy.json",
                      "--export-bundle", tmp_path / "bundle"]
    assert_guard(invoke("synrex-live", *arguments), tmp_path)


@pytest.mark.parametrize("mapping", MAPPINGS)
def test_text_replay_rejects_v3_before_output_creation(tmp_path, mapping):
    model, _, _ = v3_package(tmp_path, mapping)
    assert_guard(invoke(
        "synrex-stream", "replay", "--input", ROOT / "examples/streaming/deterministic.jsonl",
        "--model", model, "--out", tmp_path / "events.jsonl",
        "--receipt", tmp_path / "stream-receipt.json",
    ), tmp_path)


@pytest.mark.parametrize("mapping", MAPPINGS)
@pytest.mark.parametrize("mode", ["points", "points-map", "json", "model-only"])
def test_every_sustained_mode_rejects_v3_before_source_or_execution(tmp_path, mapping, mode):
    model, _, _ = v3_package(tmp_path, mapping)
    assert_guard(invoke(
        "synrex-sustain", "--model", model, "--mode", mode,
        "--source", tmp_path / "absent-source.laz", "--receipt", tmp_path / "sustain-receipt.json",
    ), tmp_path)


@pytest.mark.parametrize("mapping", MAPPINGS)
@pytest.mark.parametrize("mode", ["legacy", "session", "export"])
def test_direct_native_legacy_apis_reject_v3(tmp_path, mapping, mode):
    model, _, _ = v3_package(tmp_path, mapping)
    assert_guard(invoke("synrex-package-policy-tests", mode, model, tmp_path / "bundle"), tmp_path)


@pytest.mark.parametrize("mapping", [[2, 5, 6], [1, 2, 5, 6, 9, 17, 64]])
def test_v2_legacy_package_policy_positive_controls(tmp_path, mapping):
    spec = copy.deepcopy(SPEC)
    spec["las_classes"] = mapping
    spec["output"]["shape"][1] = len(mapping)
    model, _, _ = cpu_package(tmp_path, spec, "v2")
    result = invoke("synrex-package-policy-tests", "legacy", model, tmp_path / "bundle")
    assert result.returncode == 0, result.stderr
    expected = hashlib.sha256((model / "manifest.json").read_bytes()).hexdigest()
    assert result.stdout.strip() == expected
    # Only the CPU backend positive reaches its sentinel; never initialize a v2 GPU session.
    backend = invoke("synrex-package-policy-tests", "backend", model, tmp_path / "bundle")
    assert backend.returncode == 0, backend.stderr
    if backend.stdout.strip() == "cpu":
        result = invoke("synrex-package-policy-tests", "session", model, tmp_path / "bundle")
        assert result.returncode != 0
        assert "CUDA/TensorRT backend not built" in result.stderr
        assert GUARD not in result.stderr


def test_capability_policy_precedes_contract_and_artifacts(tmp_path):
    model = tmp_path / "envelope"
    model.mkdir()
    (model / "manifest.json").write_text(json.dumps({"schema": "synrex.srxmodel.v3"}))
    result = invoke("synrex-package-policy-tests", "legacy", model, tmp_path / "bundle")
    assert_guard(result, tmp_path)
