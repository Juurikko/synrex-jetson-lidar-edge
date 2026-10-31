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
    model = tmp_path / "model"
    model.mkdir()
    (model / "manifest.json").write_text('{"fixture":"no-inference"}\n')
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

                    def send(kind, value):
                        payload = json.dumps(value, separators=(",", ":")).encode()
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
                        return hashlib.sha256(payload).hexdigest()

                    snapshot_hash = "0" * 64
                    if fault != "missing":
                        snapshot_hash = send(
                            3,
                            {
                                "identity": None,
                                "schema": "synrex.semantic_map.v1",
                                "resolution_metres": 1.0,
                                "capacity": 65536,
                                "mode": "map_delta_output",
                                "cells": [] if fault == "empty_success" else [{}],
                            },
                        )
                    send(
                        4,
                        {
                            "status": "PASS",
                            "output_points": 0,
                            "received_frames": 0,
                            "map_snapshot_sha256": snapshot_hash,
                        },
                    )
        except (BrokenPipeError, ConnectionResetError):
            # Early rejection is expected for the corrupt snapshot.
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
                str(tmp_path / "receipt.json"),
                "--model",
                str(model),
                "--output-port",
                str(port),
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
    finally:
        thread.join(timeout=12)
    assert not thread.is_alive()
    assert not errors
    assert result.returncode == 1, result.stderr
    data = json.loads(result.stdout)
    assert data["status"] == "FAIL"
    assert "subscriber" in data["error"]
    assert data.get("required_snapshot_validated") is not True
