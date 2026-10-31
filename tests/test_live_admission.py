# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
"""Live admission and terminal failure behavior on CPU, without a GPU session."""

import hashlib
import json
import os
import socket
import struct
import subprocess
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NATIVE = Path(os.environ.get("SYNREX_TEST_NATIVE_DIR", ROOT / "build")).resolve()
CPU_MODEL = b"CPU_SUBSCRIBER_MODEL_FREE_ONNX_PLACEHOLDER_NO_INFERENCE"
CPU_LICENSE = b"MIT; CPU wire fixture only; no weights or inference qualification.\n"
CPU_SPEC = json.loads((ROOT / "examples/dense_voxel.json").read_text())
CPU_MANIFEST = (json.dumps({
    "schema": "synrex.srxmodel.v2",
    "contract": CPU_SPEC,
    "artifacts": {
        "onnx": {"path": "model.onnx", "sha256": hashlib.sha256(CPU_MODEL).hexdigest()},
        "license": {"path": "MODEL_LICENSE.txt", "sha256": hashlib.sha256(CPU_LICENSE).hexdigest()},
    },
    "target": None,
}, sort_keys=True) + "\n").encode()


def json_payload(value):
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode()


def run_subscriber_peer(tmp_path, messages, validation_mode="decoded"):
    """Send independently encoded fixtures to the actual CPU subscriber executable."""
    model = tmp_path / "model"
    model.mkdir()
    (model / "manifest.json").write_bytes(CPU_MANIFEST)
    (model / "model.onnx").write_bytes(CPU_MODEL)
    (model / "MODEL_LICENSE.txt").write_bytes(CPU_LICENSE)
    receipt = tmp_path / "receipt.json"
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    listener.settimeout(10)
    port = listener.getsockname()[1]
    errors = []

    def peer():
        try:
            with listener:
                conn, _ = listener.accept()
                with conn:
                    conn.settimeout(10)
                    for kind, payload in messages:
                        header = struct.pack(
                            "<8sHHIQ32s",
                            b"SRXWIRE1",
                            2,
                            kind,
                            len(payload),
                            0,
                            hashlib.sha256(payload).digest(),
                        )
                        conn.sendall(header + payload)
        except (BrokenPipeError, ConnectionResetError):
            # A malformed snapshot can be rejected before the terminal record arrives.
            pass
        except Exception as error:
            errors.append(error)

    thread = threading.Thread(target=peer)
    thread.start()
    try:
        result = subprocess.run(
            [
                str(NATIVE / "synrex-live"),
                "subscribe",
                "--receipt",
                str(receipt),
                "--model",
                str(model),
                "--output-port",
                str(port),
                "--validation-mode",
                validation_mode,
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
    finally:
        thread.join(timeout=12)
    assert not thread.is_alive()
    assert not errors
    assert "subscriber_ready=1" in result.stdout, result.stderr + result.stdout
    # stdout also contains readiness markers; only the receipt is a JSON document.
    return result, json.loads(receipt.read_text())


def classified_prefix_fixture():
    """Wire-v2 layout from transport.cpp; all classes and counters are synthetic CPU data."""
    identity = ["cpu-sensor", "cpu-stream", "cpu-world", "LOCAL:cpu", "synrex.synthetic3.v1"]

    def identifier(value):
        encoded = value.encode("ascii")
        return struct.pack("<H", len(encoded)) + encoded

    # Five identifiers, world coordinates, UNIX ns, metres, complete frame,
    # sequence/frame/start/end/replay times, point count, and one 48-byte point.
    source = b"".join(identifier(value) for value in identity)
    source += struct.pack("<BBBBQQQQQI", 3, 1, 1, 1, 0, 0, 1000, 1000, 0, 1)
    source += struct.pack("<QQdddHBBB3x", 17, 1000, -1.25, 2.5, 3.75, 0, 0, 0, 0)
    model_hash = hashlib.sha256(CPU_MANIFEST).hexdigest()
    # synthetic3 channel 2 maps exactly to LAS code 6. Its sole 64-byte map delta
    # contains floor(xyz), semantic ID, absent eviction, timestamp and zero victim.
    tail = identifier(model_hash) + struct.pack("<BBI", 2, 6, 1)
    tail += struct.pack("<qqqBB6xQqqq", -2, 2, 3, 2, 0, 1000, 0, 0, 0)
    classified = source + struct.pack("<QQQ", 1, 2, 1) + tail
    canonical_classified = source + struct.pack("<QQQ", 0, 0, 1) + tail
    snapshot = {
        "identity": identity,
        "schema": "synrex.semantic_map.v1",
        "resolution_metres": 1.0,
        "capacity": 65536,
        "mode": "map_delta_output",
        "cells": [{
            "key": [-2, 2, 3],
            "semantic_id": 2,
            "timestamp_ns": 1000,
            "classification_age_ns": 0,
            "confidence": None,
            "confidence_age_ns": None,
        }],
    }
    terminal = {
        "status": "PASS",
        "output_points": 1,
        "received_frames": 1,
        "input_sha256": hashlib.sha256(source).hexdigest(),
        "output_sha256": hashlib.sha256(canonical_classified).hexdigest(),
        "map_snapshot_sha256": hashlib.sha256(json_payload(snapshot)).hexdigest(),
        "test_scope": "CPU peer fixture; no neural inference evidence",
    }
    return classified, snapshot, terminal


@pytest.mark.parametrize("mode", ["receive", "subscribe"])
@pytest.mark.parametrize("capacity", [0, 65537, 262144, 18446744073709551615])
def test_capacity_rejected_before_construction(tmp_path, mode, capacity):
    receipt = tmp_path / "receipt.json"
    product = tmp_path / "must-not-create"
    result = subprocess.run(
        [
            str(NATIVE / "synrex-live"),
            mode,
            "--receipt",
            str(receipt),
            "--map-cells",
            str(capacity),
            "--model",
            str(tmp_path / "absent-model"),
            "--source",
            str(tmp_path / "absent-source"),
            "--export-bundle",
            str(product),
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 1, result.stderr
    data = json.loads(receipt.read_text())
    assert data["status"] == "FAIL"
    assert "legacy map capacity must be in 1..65536" in data["error"]
    assert "initialization_seconds" not in data
    assert not product.exists()
    # The diagnostic precedes CPU package/source reads and the no-GPU sentinel.
    assert "CUDA/TensorRT" not in data["error"]
    assert "absent" not in data["error"]


@pytest.mark.parametrize("mode", ["receive", "subscribe"])
@pytest.mark.parametrize("points,rate", [(0, 1000000), (16384, 0), (16384, 16383)])
def test_point_rate_admission_before_initialization(tmp_path, mode, points, rate):
    result = subprocess.run(
        [
            str(NATIVE / "synrex-live"),
            mode,
            "--receipt",
            str(tmp_path / "receipt.json"),
            "--points",
            str(points),
            "--rate",
            str(rate),
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 1, result.stderr
    assert "legacy live point/rate budget" in json.loads(result.stdout)["error"]


@pytest.mark.parametrize("fault", ["missing", "corrupt", "empty_success"])
def test_required_snapshot_cannot_be_omitted_or_forged(tmp_path, fault):
    """A fake CPU peer cannot qualify success by sending only a PASS end marker."""
    messages = []
    snapshot_hash = "0" * 64
    if fault != "missing":
        snapshot = json_payload({
            "identity": None,
            "schema": "synrex.semantic_map.v1",
            "resolution_metres": 1.0,
            "capacity": 65536,
            "mode": "map_delta_output",
            "cells": [] if fault == "empty_success" else [{}],
        })
        messages.append((3, snapshot))
        snapshot_hash = hashlib.sha256(snapshot).hexdigest()
    messages.append((4, json_payload({
        "status": "PASS",
        "output_points": 0,
        "received_frames": 0,
        "map_snapshot_sha256": snapshot_hash,
    })))
    result, data = run_subscriber_peer(tmp_path, messages)
    assert result.returncode == 1, result.stderr
    assert data["status"] == "FAIL"
    assert "subscriber" in data["error"]
    assert data.get("required_snapshot_validated") is not True


@pytest.mark.parametrize("validation_mode", ["decoded", "reference"])
@pytest.mark.parametrize("fault", ["none", "missing", "digest", "semantic", "receiver_fail"])
def test_nonempty_subscriber_terminal_contract(tmp_path, validation_mode, fault):
    """Exercise the native subscriber after a valid prefix; never claim real inference."""
    classified, snapshot, terminal = classified_prefix_fixture()
    messages = [(2, classified)]
    if fault == "semantic":
        snapshot["cells"][0]["semantic_id"] = 1
        # Rebind the digest so semantic rejection cannot be a transport checksum failure.
        terminal["map_snapshot_sha256"] = hashlib.sha256(json_payload(snapshot)).hexdigest()
    if fault != "missing":
        messages.append((3, json_payload(snapshot)))
    if fault == "digest":
        terminal["map_snapshot_sha256"] = "0" * 64
    if fault == "receiver_fail":
        terminal["status"] = "FAIL"
    messages.append((4, json_payload(terminal)))
    result, data = run_subscriber_peer(tmp_path, messages, validation_mode)
    assert data["validated_prefix"]["frames"] == 1, data
    assert data["validated_prefix"]["points"] == 1, data
    assert data["validated_prefix"]["map_updates"] == 1, data
    if fault in {"none", "receiver_fail"}:
        assert data["required_snapshot_validated"] is True
        assert data["frames"] == data["points"] == data["model_executions"] == 1
        assert data["map_snapshot_sha256"] == terminal["map_snapshot_sha256"]
        assert data["input_sha256"] == terminal["input_sha256"]
        assert data["output_sha256"] == terminal["output_sha256"]
        assert data["receiver"]["test_scope"] == "CPU peer fixture; no neural inference evidence"
        assert data["status"] == terminal["status"]
        assert result.returncode == (0 if fault == "none" else 1), result.stderr
    else:
        expected_error = {
            "missing": "subscriber complete output coverage",
            "digest": "subscriber/receiver full snapshot digest parity",
            "semantic": "subscriber full snapshot parity",
        }[fault]
        assert result.returncode == 1, result.stderr
        assert data["status"] == "FAIL"
        assert expected_error in data["error"], data
        assert data.get("required_snapshot_validated") is not True
