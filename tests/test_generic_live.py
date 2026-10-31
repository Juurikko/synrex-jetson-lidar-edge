# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
"""Real generic CLI processes/sockets with an opaque CPU backend, never inference evidence."""

import hashlib
import json
import os
import selectors
import socket
import subprocess
from pathlib import Path

import laspy
import numpy as np
import pytest

from synthetic_laz import create

ROOT = Path(__file__).resolve().parents[1]
BUILD = Path(os.environ.get("SYNREX_TEST_NATIVE_DIR", ROOT / "build")).resolve()
FIXTURE = BUILD / "synrex-generic-live-cpu-fixture"
PRODUCTION = BUILD / "synrex-generic-live"
SESSION = "c" * 32


def digest(data):
    return hashlib.sha256(data).hexdigest()


def package(tmp_path, count, changed=False):
    model = tmp_path / ("changed" if changed else "model")
    model.mkdir()
    spec = json.loads((ROOT / "examples/dense_voxel.json").read_text())
    mapping = [1, 2, 200, 6, 9, 17, 201] if count == 7 else [i % 256 for i in range(count)]
    spec["las_classes"] = mapping
    spec["output"]["shape"][1] = count
    taxonomy = (
        json.dumps(
            {
                "schema": "synrex.taxonomy.v1",
                "taxonomy_id": "example.same.id",
                "namespace": "example.cpu",
                "unknown_semantic_id": 255 if count == 256 else None,
                "provenance": {"source": "CPU fixture", "author": "SynRex tests", "license": "MIT"},
                "classes": [
                    {
                        "semantic_id": i,
                        "key": f"class_{i}",
                        "name": ("Changed " if changed else "Meaning ") + str(i),
                        "las_code": code,
                    }
                    for i, code in enumerate(mapping)
                ],
            },
            indent=2,
        )
        + "\n\n"
    ).encode()
    artifacts = {
        "engine": b"Opaque CPU fixture: NEVER executed",
        "license": b"MIT CPU fixture\n",
        "taxonomy": taxonomy,
    }
    names = {"engine": "model.engine", "license": "MODEL_LICENSE.txt", "taxonomy": "taxonomy.json"}
    for key, value in artifacts.items():
        (model / names[key]).write_bytes(value)
    manifest = {
        "schema": "synrex.srxmodel.v3",
        "contract": spec,
        "artifacts": {
            key: {"path": names[key], "sha256": digest(value)} for key, value in artifacts.items()
        },
        "target": {
            "architecture": "CPU-test-only",
            "gpu_name": "NO GPU",
            "sm": 89,
            "tensorrt": "10.test",
            "cuda_runtime": 12010,
        },
    }
    (model / "manifest.json").write_text(json.dumps(manifest))
    return model


def port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def arguments(mode, model, receipt, port_number):
    return [
        mode,
        "--wire",
        "binary-v3",
        "--model",
        str(model),
        "--receipt",
        str(receipt),
        "--session",
        SESSION,
        "--points",
        "8",
        "--port",
        str(port_number),
        "--io-ms",
        "800",
        "--startup-ms",
        "2000",
        "--map-cells",
        "32",
    ]


def start_receiver(command, environment):
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=environment
    )
    with selectors.DefaultSelector() as selector:
        selector.register(process.stderr, selectors.EVENT_READ)
        if not selector.select(5):
            process.kill()
            raise AssertionError("receiver did not listen within five seconds")
        line = process.stderr.readline()
    assert line == "generic_listening=1\n", line + str(process.communicate(timeout=5))
    return process


def pipeline(
    tmp_path,
    count=4,
    changed=False,
    fault=False,
    records=23,
    export=True,
    subscriber=True,
    trace_bytes=None,
    trace_failure=False,
):
    model = package(tmp_path, count)
    expected_model = package(tmp_path, count, changed=True) if changed else model
    source, _ = create(tmp_path, 23)
    source_hash = digest(source.read_bytes())
    input_port, output_port = port(), port()
    while output_port == input_port:
        output_port = port()
    env = dict(os.environ)
    if fault:
        env["SYNREX_FIXTURE_RUN"] = "fail"
    if trace_failure:
        env["SYNREX_FIXTURE_TRACE"] = "fail"
    receiver_args = arguments("receive", model, tmp_path / "receive.json", input_port)
    receiver_args += [
        "--output-port",
        str(output_port),
        "--frames",
        "3",
        "--records",
        str(records),
        "--source-sha256",
        source_hash,
    ]
    if trace_bytes is not None:
        receiver_args += [
            "--tensor-trace-root",
            str(tmp_path / "diagnostic"),
            "--tensor-trace-bytes",
            str(trace_bytes),
            "--trace-source-commit",
            "a" * 40,
        ]
    receiver = start_receiver([str(FIXTURE), *receiver_args], env)
    children = {"receive": receiver}
    try:
        if subscriber:
            sub_args = arguments(
                "subscribe", expected_model, tmp_path / "subscribe.json", output_port
            )
            sub_args += ["--frames", "3", "--records", str(records), "--source-sha256", source_hash]
            if records == 23:
                sub_args += ["--source", str(source)]
                if export:
                    sub_args += ["--export-bundle", str(tmp_path / "bundle")]
            children["subscribe"] = subprocess.Popen(
                [str(FIXTURE), *sub_args],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
            )
        producer_args = arguments("send", model, tmp_path / "send.json", input_port)
        index = producer_args.index("--map-cells")
        del producer_args[index : index + 2]
        producer_args += ["--source", str(source), "--rate", "1000000"]
        children["send"] = subprocess.Popen(
            [str(FIXTURE), *producer_args],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        output = {}
        for mode, process in children.items():
            stdout, stderr = process.communicate(timeout=12)
            output[mode] = {
                "code": process.returncode,
                "receipt": json.loads(stdout),
                "fixture": json.loads(stderr.splitlines()[-1]),
            }
        return output, source, model
    finally:
        for process in children.values():
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=5)


@pytest.mark.parametrize("count", [4, 7, 256])
def test_generic_cli_single_source_full_pipeline_cpu_only(tmp_path, count):
    output, source, model = pipeline(tmp_path, count=count)
    for row in output.values():
        assert row["code"] == 0, output
        assert row["receipt"]["status"] == "PASS"
        assert row["receipt"]["records"] == 23
        assert row["receipt"]["frames"] == 3
        assert row["receipt"]["physical_capture_age_seconds"] is None
        assert row["fixture"]["live_after_return"] == 0
    assert output["receive"]["fixture"]["initializations"] == 1
    assert output["receive"]["fixture"]["runs"] == 3
    assert output["send"]["fixture"]["initializations"] == 0
    assert output["subscribe"]["fixture"]["initializations"] == 0
    assert output["subscribe"]["receipt"]["source_records_independently_checked"]
    assert len({row["receipt"]["descriptor_sha256"] for row in output.values()}) == 1
    bundle = tmp_path / "bundle"
    assert (bundle / "taxonomy.json").read_bytes() == (model / "taxonomy.json").read_bytes()
    before = laspy.read(source, laz_backend=laspy.LazBackend.Lazrs)
    after = laspy.read(bundle / "classified.laz", laz_backend=laspy.LazBackend.Lazrs)
    for name in before.point_format.dimension_names:
        if name != "classification":
            assert np.array_equal(before[name], after[name]), name
    semantic = json.loads((bundle / "semantics.json").read_text())
    assert max(semantic["semantic_ids"]) == count - 1


@pytest.mark.parametrize(
    "failure", ["changed-meaning", "inference", "record-count", "no-subscriber"]
)
def test_generic_cli_failure_never_publishes_success_or_export(tmp_path, failure):
    output, _, _ = pipeline(
        tmp_path,
        changed=failure == "changed-meaning",
        fault=failure == "inference",
        records=24 if failure == "record-count" else 23,
        subscriber=failure != "no-subscriber",
    )
    assert output["receive"]["code"] == 1, output
    assert output["receive"]["receipt"]["status"] == "FAIL"
    assert all(row["receipt"]["status"] == "FAIL" for row in output.values()), output
    assert not (tmp_path / "bundle").exists()
    if failure in {"changed-meaning", "no-subscriber"}:
        assert output["receive"]["fixture"]["initializations"] == 0


@pytest.mark.parametrize(
    "key,value",
    [
        ("--wire", "binary-v2"),
        ("--points", "0"),
        ("--points", "65537"),
        ("--frame-bytes", "10"),
        ("--map-cells", "65537"),
        ("--source-sha256", "x" * 64),
        ("--frames", "4"),
        ("--records", "0"),
        ("--port", "65536"),
    ],
)
def test_generic_cli_invalid_configuration_rejects_before_package_or_socket(tmp_path, key, value):
    command = arguments("receive", tmp_path / "absent-model", tmp_path / "receipt.json", 39183)
    command += [
        "--output-port",
        "39184",
        "--frames",
        "3",
        "--records",
        "23",
        "--source-sha256",
        "a" * 64,
    ]
    if key in command:
        command[command.index(key) + 1] = value
    else:
        command += [key, value]
    completed = subprocess.run(
        [str(PRODUCTION), *command], capture_output=True, text=True, timeout=5
    )
    receipt = json.loads(completed.stdout)
    assert completed.returncode == 1
    assert receipt["status"] == "FAIL"
    assert "manifest" not in receipt["error"]
    assert "generic_listening=1" not in completed.stderr


def test_generic_cli_existing_receipt_never_changes(tmp_path):
    receipt = tmp_path / "receipt.json"
    receipt.write_bytes(b"immutable prior receipt\n")
    before = receipt.read_bytes()
    command = arguments("receive", tmp_path / "absent", receipt, 39183)
    completed = subprocess.run(
        [str(PRODUCTION), *command], capture_output=True, text=True, timeout=5
    )
    assert completed.returncode == 1
    assert receipt.read_bytes() == before


def test_generic_live_budgeted_diagnostic_cpu_forwarding(tmp_path):
    output, source, _ = pipeline(tmp_path, trace_bytes=8 * 1024 * 1024)
    assert all(result["code"] == 0 for result in output.values())
    trace = output["receive"]["receipt"]["session_identity"]["diagnostic_trace"]
    assert trace["completed_windows"] == 3
    assert trace["reserved_bytes"] <= trace["total_byte_limit"]
    root = tmp_path / "diagnostic"
    admission = json.loads((root / "admission.json").read_text())
    assert admission["executable_sha256"] == digest(FIXTURE.read_bytes())
    assert admission["declared_source_commit"] == "a" * 40
    assert "declaration" in admission["source_commit_scope"]
    for index in range(3):
        window = root / f"window-{index}"
        lineage = json.loads((window / "lineage.json").read_text())
        assert lineage["sequence"] == index
        assert lineage["source_stream_id"] == digest(source.read_bytes())
        assert lineage["backend_manifest_sha256"] == digest((window / "manifest.json").read_bytes())
        timing = json.loads((window / "timing.json").read_text())
        completed = json.loads((window / "output-timing.json").read_text())
        assert (
            0 < timing["receiver_frame_available_steady_ns"] <= timing["service_started_steady_ns"]
        )
        assert timing["service_started_steady_ns"] <= timing["trace_payload_published_steady_ns"]
        assert (
            timing["trace_payload_published_steady_ns"]
            <= timing["classification_validated_steady_ns"]
        )
        assert timing["classification_validated_steady_ns"] <= completed["output_ready_steady_ns"]
        assert completed["output_ready_steady_ns"] <= completed["output_send_started_steady_ns"]
        assert (
            completed["output_send_started_steady_ns"]
            <= completed["output_send_completed_steady_ns"]
        )
    receiver = output["receive"]["receipt"]
    assert (
        completed["output_send_completed_steady_ns"]
        <= receiver["input_completion_observed_steady_ns"]
    )
    producer = output["send"]["receipt"]
    assert (
        receiver["output_completion_observed_steady_ns"]
        >= receiver["input_completion_observed_steady_ns"]
    )
    assert (
        producer["replay_anchor_steady_ns"]
        < producer["first_scheduled_frame_available_steady_ns"]
        <= producer["last_scheduled_frame_available_steady_ns"]
    )
    assert producer["replay_clock_boot_id"] == receiver["receiver_clock_boot_id"]

    assert (
        sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
        <= trace["total_byte_limit"]
    )


def test_generic_live_trace_failure_keeps_partial_evidence_never_completion(tmp_path):
    output, _, _ = pipeline(tmp_path, trace_bytes=8 * 1024 * 1024, trace_failure=True)
    assert output["receive"]["code"] == 1
    assert output["receive"]["receipt"]["status"] == "FAIL"
    assert "output_completion" not in output["receive"]["receipt"]
    assert not (tmp_path / "bundle").exists()
    assert (tmp_path / "diagnostic/admission.json").is_file()
    assert not (tmp_path / "diagnostic/window-0/lineage.json").exists()
    assert output["receive"]["fixture"]["runs"] == 0


@pytest.mark.parametrize(
    "failure",
    [
        "partial",
        "bytes",
        "source",
        "existing",
        "symlink",
        "parent_alias",
        "alias_parent_dotdot",
        "dot",
        "parent_dotdot",
    ],
)
def test_generic_live_trace_admission_precedes_package_gpu_and_listen(tmp_path, failure):
    root = tmp_path / "trace"
    if failure == "existing":
        root.mkdir()
    if failure == "symlink":
        root.symlink_to(tmp_path / "absent-target")
    if failure == "parent_alias":
        (tmp_path / "real").mkdir()
        (tmp_path / "alias").symlink_to(tmp_path / "real", target_is_directory=True)
        root = tmp_path / "alias/trace"
    if failure == "alias_parent_dotdot":
        (tmp_path / "outside/dir").mkdir(parents=True)
        (tmp_path / "alias").symlink_to(tmp_path / "outside/dir", target_is_directory=True)
        root = tmp_path / "alias/../trace"
    if failure == "dot":
        root = str(tmp_path) + "/./trace"
    if failure == "parent_dotdot":
        (tmp_path / "actual").mkdir()
        root = tmp_path / "actual/../trace"
    command = arguments("receive", tmp_path / "absent", tmp_path / "receipt.json", 39183)
    command += [
        "--output-port",
        "39184",
        "--frames",
        "3",
        "--records",
        "23",
        "--source-sha256",
        "a" * 64,
    ]
    command += [
        "--tensor-trace-bytes",
        "524288" if failure == "bytes" else "8388608",
        "--trace-source-commit",
        "unbound" if failure == "source" else "a" * 40,
    ]
    if failure != "partial":
        command += ["--tensor-trace-root", str(root)]
    result = subprocess.run([str(PRODUCTION), *command], capture_output=True, text=True, timeout=5)
    assert result.returncode == 1
    assert "diagnostic" in json.loads(result.stdout)["error"]
    assert "generic_listening=1" not in result.stderr

    # In particular alias/../trace must not create the outside destination.
    assert not (tmp_path / "outside/trace").exists()


@pytest.mark.parametrize("budget", [524289, 557056])
def test_generic_live_contract_minimum_exact_frames_before_listen(tmp_path, budget):
    model = package(tmp_path, 4)
    command = arguments("receive", model, tmp_path / "receipt.json", 39183)
    command += [
        "--output-port",
        "39184",
        "--frames",
        "3",
        "--records",
        "23",
        "--source-sha256",
        "a" * 64,
        "--tensor-trace-root",
        str(tmp_path / "trace"),
        "--tensor-trace-bytes",
        str(budget),
        "--trace-source-commit",
        "a" * 40,
    ]
    result = subprocess.run([str(PRODUCTION), *command], capture_output=True, text=True, timeout=5)
    assert result.returncode == 1
    assert "required minimum tensor windows before backend" in json.loads(result.stdout)["error"]
    assert "generic_listening=1" not in result.stderr
    assert not (tmp_path / "trace").exists()
