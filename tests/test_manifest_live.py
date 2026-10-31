# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
"""Actual multi-process manifest transport with a clearly identified CPU backend."""

import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

import pytest
from test_generic_live import FIXTURE, PRODUCTION, SESSION, package, port, start_receiver
from test_replay_reader import cloud, manifest


def arguments(mode, model, receipt, port_number, manifest_path, rate=1000000, wire="binary-v3"):
    return [mode, "--wire", wire, "--model", str(model), "--receipt", str(receipt),
            "--session", SESSION, "--manifest", str(manifest_path), "--points", "8",
            "--rate", str(rate), "--port", str(port_number), "--io-ms", "1200",
            "--startup-ms", "3000", "--map-cells", "32", "--queue-frames", "2"]


def pipeline(tmp_path, count=4, failure=None, rate=1000000, sources=2, wire="binary-v3",
             raw_bits=False, queue_frames=2, timing="enabled", reuse_inputs=False,
             receipt_dir=None, profile_backend=None):
    receipt_dir = receipt_dir or tmp_path
    model = tmp_path / "model" if reuse_inputs else package(tmp_path, count)
    other_model = package(tmp_path, count, changed=True) if failure == "taxonomy" else model
    paths = [tmp_path / f"source-{i}.laz" for i in range(sources)]
    counts = [23, 11][:sources]
    if failure == "cancel":
        counts[0] = 1001
    for index, (path, records) in enumerate(zip(paths, counts, strict=True)):
        if not reuse_inputs:
            cloud(path, records, 6 + index, offsets=[1000.0 * index, -2000.0 * index, 12.5])
    if raw_bits:
        import laspy
        import numpy as np
        for path in paths:
            data = laspy.read(path, laz_backend=laspy.LazBackend.Lazrs)
            data.return_number = np.zeros(len(data.points), dtype=np.uint8)
            data.number_of_returns = np.zeros(len(data.points), dtype=np.uint8)
            gps = np.resize(np.array([0x8000000000000000, 0x7ff8000000000001,
                                     0xffffffffffffffff], dtype=np.uint64), len(data.points))
            data.gps_time = gps.view(np.float64)
            data.write(path, laz_backend=laspy.LazBackend.Lazrs)
    if failure == "late-decode":
        cloud(paths[-1], counts[-1], 7, offsets=[1000000001.0, 0.0, 0.0])
    manifest_path = tmp_path / "manifest.json"
    if not reuse_inputs:
        manifest(manifest_path, list(zip(paths, counts, strict=True)))
    input_port, output_port = port(), port()
    while input_port == output_port:
        output_port = port()
    environment = dict(os.environ)
    if failure == "inference":
        environment["SYNREX_FIXTURE_RUN"] = "fail"
    receive_args = arguments("receive", model, receipt_dir / "receive.json", input_port,
                             manifest_path, rate, wire)
    receive_args += ["--output-port", str(output_port), "--timing-instrumentation", timing]
    if profile_backend is not None:
        receive_args += ["--profile-backend", profile_backend]
    children = {"receive": start_receiver([str(FIXTURE), *receive_args], environment)}
    try:
        for mode, model_path, port_number in [
            ("subscribe", other_model, output_port), ("send", model, input_port)
        ]:
            if mode == "subscribe" and failure == "missing-subscriber":
                continue
            command = arguments(mode, model_path, receipt_dir / f"{mode}.json", port_number,
                                manifest_path, rate, wire)
            command[command.index("--queue-frames") + 1] = str(queue_frames)
            command += ["--timing-instrumentation", timing]
            if mode == "subscribe" and failure == "output-map":
                command[command.index("--map-cells") + 1] = "1"
            children[mode] = subprocess.Popen(
                [str(FIXTURE), *command], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, env=environment,
            )
        if failure == "cancel":
            sender = children["send"]
            deadline = time.monotonic() + 5
            while len(list(Path(f"/proc/{sender.pid}/task").iterdir())) < 2:
                assert sender.poll() is None and time.monotonic() < deadline
                time.sleep(0.005)
            time.sleep(0.15)
            sender.send_signal(signal.SIGTERM)
        result = {}
        for mode, child in children.items():
            stdout, stderr = child.communicate(timeout=20)
            result[mode] = {"code": child.returncode, "receipt": json.loads(stdout),
                            "backend": json.loads(stderr.splitlines()[-1])}
        return result, manifest_path
    finally:
        for child in children.values():
            if child.poll() is None:
                child.kill()
                child.communicate(timeout=5)


@pytest.mark.parametrize("classes", [4, 7, 256])
@pytest.mark.parametrize("wire", ["binary-v3", "binary-source-v1"])
def test_manifest_cli_finite_distinct_full_pipeline_and_per_source_reset(tmp_path, classes, wire):
    results, manifest_path = pipeline(tmp_path, count=classes, wire=wire)
    for mode, result in results.items():
        receipt = result["receipt"]
        assert result["code"] == 0, results
        assert receipt["status"] == "PASS"
        assert receipt["binary_wire"] == wire
        if wire == "binary-source-v1":
            assert "raw GPS bits" in receipt["source_timestamp_transport"]
        assert receipt["manifest_sha256"] == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        assert receipt["declared_records"] == 34
        assert receipt["offered_finite_duration_seconds"] == 34 / 1000000
        assert receipt["capacity_qualification"] == "NOT_RUN"
        assert receipt["physical_capture_age_seconds"] is None
        assert receipt["multi_file_preservation"] == "NOT_SUPPORTED"
        assert receipt["boundary_policy"] == "reset_each_file"
        assert result["backend"]["live_after_return"] == 0
        assert result["backend"]["initializations"] == (2 if mode == "receive" else 0)
        assert result["backend"]["runs"] == (5 if mode == "receive" else 0)
        assert [row["wire_begin"] for row in receipt["sources"]] == [0, 23]
        assert [row["map_epoch"] for row in receipt["sources"]] == [1, 2]
        assert [row["frames"] for row in receipt["sources"]] == [3, 2]
        for row in receipt["sources"]:
            assert row["status"] == "PASS"
            assert row["coverage"]["state"] == "SUCCEEDED"
            assert row["coverage"]["accepted_points"] == row["expected_records"]
            assert row["coverage"]["declared_frame_coverage_complete"]
    for index in range(2):
        assert len({result["receipt"]["sources"][index]["descriptor_sha256"]
                    for result in results.values()}) == 1
    assert results["receive"]["receipt"]["accepted_records"] == 34
    assert results["receive"]["receipt"]["freshly_classified_records"] == 34
    assert results["receive"]["receipt"]["emitted_records"] == 34
    assert results["receive"]["receipt"]["independently_validated_records"] == 0
    assert results["receive"]["receipt"]["subscriber_snapshot_acknowledged_records"] == 34
    assert results["subscribe"]["receipt"]["independently_validated_records"] == 34
    assert results["send"]["receipt"]["emitted_records"] == 34
    for mode in ["send", "subscribe"]:
        for row in results[mode]["receipt"]["sources"]:
            assert row["coverage"]["exact_attribute_points"] == row["expected_records"]
            assert row["decoder"]["source_verified_after_decode"]
    for row in results["send"]["receipt"]["sources"]:
        queue = row["decode_queue"]
        assert queue["capacity"] == 2
        assert queue["high_water_items"] <= 2
        assert queue["queued_items"] == queue["queued_bytes"] == 0
        assert queue["high_water_bytes"] <= queue["byte_limit"]
        assert queue["high_water_allocation_bytes"] <= queue["byte_limit"]


@pytest.mark.parametrize("failure", ["taxonomy", "inference", "late-decode"])
def test_manifest_cli_faults_keep_failed_trial_and_verified_prefix(tmp_path, failure):
    results, _ = pipeline(tmp_path, failure=failure)
    for result in results.values():
        assert result["code"] == 1, results
        assert result["receipt"]["status"] == "FAIL"
        assert result["backend"]["live_after_return"] == 0
    if failure == "taxonomy":
        assert results["receive"]["backend"]["initializations"] == 0
    if failure == "late-decode":
        for result in results.values():
            rows = result["receipt"]["sources"]
            assert [row["status"] for row in rows] == ["PASS", "FAIL"]
        assert results["receive"]["receipt"]["freshly_classified_records"] == 23
        assert results["subscribe"]["receipt"]["independently_validated_records"] == 23
        assert results["send"]["receipt"]["emitted_records"] == 23


def test_manifest_cli_finite_pacing_counts_partial_window_accumulation(tmp_path):
    results, _ = pipeline(tmp_path, rate=80, sources=1)
    producer = results["send"]["receipt"]
    assert all(row["code"] == 0 for row in results.values()), results
    assert producer["offered_finite_duration_seconds"] == 23 / 80
    assert producer["producer_elapsed_seconds"] >= 23 / 80
    assert producer["producer_elapsed_seconds"] < 5


def test_manifest_cli_input_delivery_success_is_separate_from_output_success(tmp_path):
    results, _ = pipeline(tmp_path, failure="output-map", sources=1)
    producer = results["send"]
    assert producer["code"] == 0, results
    assert producer["receipt"]["status"] == "PASS"
    assert "input delivery acknowledged only" in producer["receipt"]["scope"]
    assert "output excluded" in producer["receipt"]["timed_boundary"]
    for mode in ["receive", "subscribe"]:
        assert results[mode]["code"] == 1, results
        assert results[mode]["receipt"]["status"] == "FAIL"


@pytest.mark.parametrize("failure", ["missing-subscriber", "cancel"])
def test_manifest_cli_missing_peer_and_active_decode_cancellation_fail(tmp_path, failure):
    results, _ = pipeline(tmp_path, failure=failure, sources=1,
                          rate=80 if failure == "cancel" else 1000000)
    for result in results.values():
        assert result["code"] == 1, results
        assert result["receipt"]["status"] == "FAIL"
        assert result["backend"]["live_after_return"] == 0
        if failure == "missing-subscriber":
            assert result["backend"]["initializations"] == 0
    if failure == "cancel":
        queue = results["send"]["receipt"]["sources"][0]["decode_queue"]
        assert queue["failed"]
        assert queue["queued_items"] == queue["queued_allocation_bytes"] == 0
        assert not queue["producer_waiting"]
        assert queue["producer_waits"] > 0


@pytest.mark.parametrize("key,value", [
    ("--queue-frames", "3"), ("--rate", "7"), ("--points", "0"),
    ("--wire", "binary-v2"), ("--map-cells", "65537"), ("--frame-bytes", "10"),
    ("--timing-instrumentation", "maybe"),
    ("--export-bundle", "forbidden-export"), ("--source", "forbidden-source"),
])
def test_manifest_cli_invalid_options_reject_before_manifest_package_or_gpu(tmp_path, key, value):
    command = arguments("receive", tmp_path / "absent-model", tmp_path / "receipt.json",
                        39183, tmp_path / "absent-manifest.json")
    command += ["--output-port", "39184"]
    if key in command:
        command[command.index(key) + 1] = value
    else:
        command += [key, value]
    completed = subprocess.run([str(PRODUCTION), *command], capture_output=True, text=True,
                               timeout=5)
    receipt = json.loads(completed.stdout)
    assert completed.returncode == 1
    assert receipt["status"] == "FAIL"
    assert "cannot open" not in receipt["error"]
    assert "generic_listening" not in completed.stderr


def test_manifest_cli_header_count_admitted_before_package_or_gpu(tmp_path):
    source = tmp_path / "source.laz"
    cloud(source, 23, 7)
    path = tmp_path / "manifest.json"
    manifest(path, [(source, 24)])
    command = arguments("receive", tmp_path / "absent-model", tmp_path / "receipt.json",
                        39183, path)
    command += ["--output-port", "39184"]
    result = subprocess.run([str(PRODUCTION), *command], capture_output=True, text=True, timeout=5)
    receipt = json.loads(result.stdout)
    assert result.returncode == 1
    assert "declared/header" in receipt["error"]
    assert "generic_listening" not in result.stderr


def test_manifest_cli_prior_receipt_is_immutable(tmp_path):
    receipt = tmp_path / "receipt.json"
    receipt.write_bytes(b"prior receipt\n")
    command = arguments("receive", tmp_path / "absent", receipt, 39183, tmp_path / "absent.json")
    result = subprocess.run([str(PRODUCTION), *command], capture_output=True, text=True, timeout=5)
    assert result.returncode == 1
    assert receipt.read_bytes() == b"prior receipt\n"


def test_manifest_source_codec_preserves_raw_gps_nan_bits_and_paired_zero_returns(tmp_path):
    results, _ = pipeline(tmp_path, wire="binary-source-v1", raw_bits=True)
    assert all(result["code"] == 0 for result in results.values()), results
    for mode in ["send", "subscribe"]:
        for row in results[mode]["receipt"]["sources"]:
            assert row["coverage"]["raw_source_attribute_points"] == row["expected_records"]
            assert row["coverage"]["source_attribute_schema"] == "synrex.las14.source_attributes.v1"


def test_source_wire_unknown_extra_attributes_reject_before_package_or_gpu(tmp_path):
    import laspy
    import numpy as np
    source = tmp_path / "extra.laz"
    cloud(source, 3, 6)
    data = laspy.read(source, laz_backend=laspy.LazBackend.Lazrs)
    data.add_extra_dim(laspy.ExtraBytesParams(name="unadmitted", type=np.uint16))
    data.write(source, laz_backend=laspy.LazBackend.Lazrs)
    path = tmp_path / "manifest.json"
    manifest(path, [(source, 3)])
    command = arguments("receive", tmp_path / "absent-model", tmp_path / "receipt.json",
                        39183, path, wire="binary-source-v1")
    command += ["--output-port", "39184"]
    result = subprocess.run([str(PRODUCTION), *command], capture_output=True, text=True, timeout=5)
    receipt = json.loads(result.stdout)
    assert result.returncode == 1
    assert "extra" in receipt["error"].lower(), receipt
    assert "cannot open" not in receipt["error"]
    assert "generic_listening" not in result.stderr
