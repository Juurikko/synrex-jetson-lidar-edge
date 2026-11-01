# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
"""CPU feature and hostile-artifact oracles; no learned inference executes."""

import importlib.util
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
LOADER = importlib.util.spec_from_file_location(
    "runtime_frozen_reference", ROOT / "tools/runtime_frozen_reference.py"
)
reference = importlib.util.module_from_spec(LOADER)
LOADER.loader.exec_module(reference)
SPEC = json.loads((ROOT / "examples/unet/dense_voxel.json").read_text())


def block_bytes(features=None, ids=None, linear=None):
    buffer = io.BytesIO()
    np.savez(
        buffer,
        features=np.zeros((1, 4, 32, 32, 32), np.float32) if features is None else features,
        ids=np.array([0, 1], np.int64) if ids is None else ids,
        linear=np.array([0, 1], np.int64) if linear is None else linear,
    )
    return buffer.getvalue()


def test_boundary_membership_and_features():
    xyz = np.array(
        [[-1.0, -1.0, -1.0], [-1.0, -1.0, 0.0], [31.0, -1.0, -1.0], [30.999999, -1.0, -1.0]]
    )
    result = list(reference.feature_blocks(xyz, np.array([0, 65535, 32768, 1], np.uint16), SPEC))
    assert [row["ids"].tolist() for row in result] == [[0, 1, 3], [2]]
    assert [row["linear"].tolist() for row in result] == [[0, 1024, 31], [0]]
    tensor = result[0]["features"].reshape(4, -1)
    np.testing.assert_array_equal(tensor[:, 0], [1, 1 / 16, 0, 0])
    np.testing.assert_array_equal(tensor[:, 1024], [1, 1 / 16, 1 / 32, 1])
    assert np.count_nonzero(tensor[0]) == 3


def test_reduction_and_no_truth_input():
    xyz = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    result = list(reference.feature_blocks(xyz, np.array([0, 65535], np.uint16), SPEC))[0]
    np.testing.assert_array_equal(
        result["features"].reshape(4, -1)[:, 0], [1, 2 / 16, 0.25 / 32, 0.5]
    )
    assert set(result) == {"features", "ids", "linear", "origin"}


def test_independent_slow_voxel_reduction():
    rng = np.random.default_rng(2718)
    xyz = rng.uniform(-4, 43, (257, 3)).astype(np.float64)
    intensity = rng.integers(0, 65536, len(xyz), dtype=np.uint16)
    coverage = np.zeros(len(xyz), int)
    for block in reference.feature_blocks(xyz, intensity, SPEC):
        local = xyz[block["ids"]] - block["origin"]
        coordinates = np.floor(local).astype(int)
        ids = block["ids"]
        independent_linear = coordinates[:, 2] * 1024 + coordinates[:, 1] * 32 + coordinates[:, 0]
        np.testing.assert_array_equal(independent_linear, block["linear"])
        tensor = block["features"].reshape(4, -1)
        for voxel in set(independent_linear):
            selected = np.flatnonzero(independent_linear == voxel)
            values = [
                1,
                len(selected) / 16,
                local[selected, 2].mean() / 32,
                intensity[ids[selected]].astype(float).mean()
                * float(np.float32(SPEC["features"][3]["scale"])),
            ]
            # Baseline normalization casts the reduced value before multiplying.
            values[2] = np.float32(local[selected, 2].mean()) * np.float32(1 / 32)
            values[3] = np.float32(intensity[ids[selected]].astype(float).mean()) * np.float32(
                SPEC["features"][3]["scale"]
            )
            np.testing.assert_array_equal(tensor[:, voxel], np.array(values, np.float32))
        coverage[ids] += 1
    assert np.all(coverage == 1)


def test_window_slices_exclude_future_minimum():
    xyz = np.array([[100.0, 100.0, 100.0], [101.0, 100.0, 101.0], [-999.0, -999.0, -999.0]])
    prefix = list(reference.feature_blocks(xyz[:2], np.zeros(2, np.uint16), SPEC))
    assert prefix[0]["origin"] == [100.0, 100.0, 100.0]
    assert prefix[0]["linear"].tolist() == [0, 1025]
    # This tests feature slice isolation only; no executable model/native prefix probe.


@pytest.mark.parametrize(
    "xyz",
    [
        np.empty((0, 3)),
        np.array([[np.nan, 0, 0]]),
        np.array([[0.0, 0, 0], [1e9, 1e9, 1e9]]),
        np.zeros((reference.MAX_POINTS + 1, 3)),
        np.zeros((3, 2)),
        np.zeros((2, 3), np.float32),
    ],
)
def test_invalid_coordinates(xyz):
    with pytest.raises(ValueError):
        list(reference.feature_blocks(xyz, np.zeros(len(xyz), np.uint16), SPEC))


@pytest.mark.parametrize("intensity", [np.array([1.0, 2.0]), np.array([0], np.uint16)])
def test_invalid_intensity(intensity):
    with pytest.raises(ValueError):
        list(reference.feature_blocks(np.zeros((2, 3)), intensity, SPEC))


@pytest.mark.parametrize(
    "changes",
    [
        {"features": np.zeros((1, 4, 1, 1, 1), np.float32)},
        {"features": np.full((1, 4, 32, 32, 32), np.nan, np.float32)},
        {"ids": np.array([0, 0], np.int64)},
        {"ids": np.array([-1, 1], np.int64)},
        {"ids": np.array([0, 2], np.int64)},
        {"ids": np.array([0, 1], np.int32)},
        {"linear": np.array([0, 32768], np.int64)},
        {"linear": np.array([0], np.int64)},
    ],
)
def test_invalid_block_payload(changes):
    with pytest.raises(ValueError):
        reference.load_block(block_bytes(**changes), 2)


def test_archive_inflation_bound(monkeypatch):
    monkeypatch.setattr(reference, "MAX_ARTIFACT_BYTES", 64)
    with pytest.raises(ValueError, match="inflated"):
        reference.load_block(block_bytes(), 2)


def test_strict_json_and_local_paths(tmp_path):
    for data in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}'):
        with pytest.raises(ValueError):
            reference.strict_json(data)
    for name in ("../out", "/tmp/out", "sub/out", "..", ""):
        with pytest.raises(ValueError):
            reference.local_path(tmp_path, name)
    (tmp_path / "link").symlink_to(tmp_path / "target")
    with pytest.raises(ValueError):
        reference.local_path(tmp_path, "link")


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    model_path = tmp_path / "fake.onnx"
    model_path.write_bytes(b"fixture identity only; not an executable model")
    monkeypatch.setattr(reference, "FROZEN_ONNX", reference.digest(model_path.read_bytes()))
    observations = [
        {
            "source_sha256": "a" * 64,
            "point_index": i,
            "available_ns": 0,
            "captured_ns": None,
            "xyz": [str(i), "0", "0"],
        }
        for i in range(2)
    ]
    reference.jsonl(tmp_path / "observations.jsonl", observations)
    quality = reference.load_quality(ROOT)
    records = []
    for policy in ("q0", "q1"):
        name = policy + ".npz"
        data = block_bytes()
        (tmp_path / name).write_bytes(data)
        window = quality.build_reference_windows(observations, policy)[0]
        records.append(
            {
                "policy": policy,
                "window": window,
                "blocks": [
                    {
                        "path": name,
                        "sha256": reference.digest(data),
                        "features_f32_sha256": reference.digest(
                            np.zeros((1, 4, 32, 32, 32), np.float32).tobytes()
                        ),
                    }
                ],
            }
        )
    monkeypatch.setattr(reference, "checkout_commit", lambda _checkout: "1" * 40)
    manifest = {
        "schema": reference.SCHEMA,
        "state": "PREPARED_NOT_INFERRED",
        "points": 2,
        "clock_boot_id": reference.boot_identity(),
        "source_files": {
            str(path): reference.digest(path.read_bytes())
            for path in reference.source_paths(ROOT).values()
        },
        "source_checkout_commit": "1" * 40,
        "spec_sha256": reference.FROZEN_SPEC,
        "spec_path": str(ROOT / "examples/unet/dense_voxel.json"),
        "source_checkout": str(ROOT),
        "source_sha256": "a" * 64,
        "onnx": str(model_path),
        "onnx_sha256": reference.FROZEN_ONNX,
        "observations": {
            "path": "observations.jsonl",
            "sha256": reference.digest((tmp_path / "observations.jsonl").read_bytes()),
        },
        "windows": records,
    }
    return tmp_path, manifest


def write_manifest(directory, manifest):
    (directory / "PREPARED.json").write_text(json.dumps(manifest))
    return reference.digest((directory / "PREPARED.json").read_bytes())


def test_preflight_accepts_complete_identity_fixture_without_model_load(prepared):
    directory, manifest = prepared
    actual, model_bytes = reference.preflight(directory, write_manifest(directory, manifest))
    assert actual == manifest
    assert model_bytes == b"fixture identity only; not an executable model"


@pytest.mark.parametrize(
    "mutation",
    [
        "manifest_hash",
        "model",
        "observations",
        "block",
        "duplicate_coverage",
        "missing_coverage",
        "window",
        "source",
        "path",
        "blocks",
        "points",
        "aggregate_bytes",
        "boot",
        "missing_role",
        "extra_role",
        "substituted_dependency",
        "spec_sha",
        "spec_file",
        "commit",
    ],
)
def test_preflight_rejects_before_inference(prepared, monkeypatch, mutation):
    directory, manifest = prepared
    if mutation in ("model", "observations", "block"):
        name = {"model": "fake.onnx", "observations": "observations.jsonl", "block": "q0.npz"}[
            mutation
        ]
        (directory / name).write_bytes(b"changed")
    elif mutation in ("duplicate_coverage", "missing_coverage"):
        data = block_bytes(ids=np.array([0], np.int64), linear=np.array([0], np.int64))
        (directory / "q0.npz").write_bytes(data)
        blocks = manifest["windows"][0]["blocks"]
        blocks[0]["sha256"] = reference.digest(data)
        if mutation == "duplicate_coverage":
            (directory / "extra.npz").write_bytes(data)
            blocks.append(dict(blocks[0], path="extra.npz"))
    elif mutation == "window":
        manifest["windows"][1]["window"]["origin"][0] = "-1"
    elif mutation == "source":
        manifest["source_files"][next(iter(manifest["source_files"]))] = "0" * 64
    elif mutation == "path":
        manifest["windows"][0]["blocks"][0]["path"] = "../q0.npz"
    elif mutation == "blocks":
        monkeypatch.setattr(reference, "MAX_BLOCKS", 1)
    elif mutation == "points":
        manifest["points"] = True
    elif mutation == "missing_role":
        del manifest["source_files"][str(ROOT / "tools/runtime_comparison.py")]
    elif mutation == "extra_role":
        manifest["source_files"][str(directory / "arbitrary")] = "0" * 64
    elif mutation == "substituted_dependency":
        path = ROOT / "tools/runtime_quality.py"
        copied = directory / "runtime_quality.py"
        copied.write_bytes(path.read_bytes())
        del manifest["source_files"][str(path)]
        manifest["source_files"][str(copied)] = reference.digest(copied.read_bytes())
    elif mutation == "spec_sha":
        manifest["spec_sha256"] = "0" * 64
    elif mutation == "spec_file":
        copied = directory / "wrong_spec.json"
        copied.write_text("{}")
        manifest["spec_path"] = str(copied)
    elif mutation == "commit":
        manifest["source_checkout_commit"] = "2" * 40
    elif mutation == "boot":
        manifest["clock_boot_id"] = "0" * 36
    elif mutation == "aggregate_bytes":
        monkeypatch.setattr(reference, "MAX_PREPARED_BYTES", 32)
    identity = write_manifest(directory, manifest)
    if mutation == "manifest_hash":
        identity = "0" * 64
    # Import must never be reached by a rejected input. No learned model executes.
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    with pytest.raises(ValueError):
        reference.infer(
            SimpleNamespace(
                prepared=directory,
                prepared_sha256=identity,
                registered_cpu_job_id="cpu_fixture",
                out=directory / "output",
            )
        )
    assert not (directory / "output").exists()


def test_missing_registration_and_output_overwrite_rejected(prepared):
    directory, manifest = prepared
    identity = write_manifest(directory, manifest)
    for job_id, output in (("", directory / "new"), ("job", directory)):
        with pytest.raises(ValueError):
            reference.infer(
                SimpleNamespace(
                    prepared=directory,
                    prepared_sha256=identity,
                    registered_cpu_job_id=job_id,
                    out=output,
                )
            )


def test_npy_huge_shape_rejected_before_array_allocation(monkeypatch):
    import zipfile

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name in ("features", "ids", "linear"):
            header = io.BytesIO()
            np.lib.format.write_array_header_1_0(
                header, {"shape": (10**12,), "fortran_order": False, "descr": "<i8"}
            )
            archive.writestr(name + ".npy", header.getvalue())
    monkeypatch.setattr(np, "load", lambda *_args, **_kwargs: pytest.fail("allocation reached"))
    with pytest.raises(ValueError):
        reference.load_block(buffer.getvalue(), 2)


@pytest.mark.parametrize("providers", [["CPUExecutionProvider"], ["CUDAExecutionProvider"]])
def test_execution_plumbing_uses_only_a_mock_session(prepared, monkeypatch, providers):
    """Synthetic scores verify plumbing; this is never trained inference evidence."""
    import time

    directory, manifest = prepared
    identity = write_manifest(directory, manifest)
    model_captures, prediction_close_events = [], []

    class Session:
        def __init__(self, model_bytes, options, providers):
            assert options.intra_op_num_threads == 2 and options.inter_op_num_threads == 1
            assert providers == ["CPUExecutionProvider"]
            model_captures.append(model_bytes)

        def get_providers(self):
            return providers

        def run(self, output_names, feed):
            assert output_names == ["scores"]
            assert feed["features"].shape == (1, 4, 32, 32, 32)
            return [np.zeros((1, 7, 32, 32, 32), np.float32)]

    fake_runtime = SimpleNamespace(
        SessionOptions=SimpleNamespace, InferenceSession=Session, __version__="SYNTHETIC_FIXTURE"
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_runtime)
    original_writer = reference.jsonl

    def timestamped_writer(path, rows):
        original_writer(path, rows)
        prediction_close_events.append(time.monotonic_ns())

    monkeypatch.setattr(reference, "jsonl", timestamped_writer)
    args = SimpleNamespace(
        prepared=directory,
        prepared_sha256=identity,
        registered_cpu_job_id="cpu_fixture",
        out=directory / "output",
    )
    if providers != ["CPUExecutionProvider"]:
        with pytest.raises(ValueError, match="exclusive CPU"):
            reference.infer(args)
        assert not args.out.exists()
        return
    reference.infer(args)
    receipt = json.loads((args.out / "REFERENCE_RECEIPT.json").read_text())
    assert model_captures == [b"fixture identity only; not an executable model"]
    assert receipt["onnxruntime_version"] == "SYNTHETIC_FIXTURE"
    for window, closed in zip(receipt["windows"], prediction_close_events, strict=True):
        assert window["output_ns"] >= closed
        predictions = [
            json.loads(line)
            for line in (args.out / window["predictions"]["path"]).read_text().splitlines()
        ]
        assert [row["point_index"] for row in predictions] == [0, 1]
        assert [row["label"] for row in predictions] == [0, 0]


@pytest.mark.parametrize(
    "version,format_id,record_bytes,points,accepted",
    [
        ("1.4", 6, 30, 50000, True),
        ("1.4", 8, 38, 2783, True),
        ("1.2", 3, 34, 2, False),
        ("1.4", 8, 65535, 2, False),
        ("1.4", 8, 38, 50001, False),
        ("1.4", 8, 38, 0, False),
    ],
)
def test_decoded_record_storage_admission(version, format_id, record_bytes, points, accepted):
    header = SimpleNamespace(
        version=version,
        point_count=points,
        point_format=SimpleNamespace(id=format_id, size=record_bytes),
    )
    if accepted:
        assert reference.admit_header(header) == points
    else:
        with pytest.raises(ValueError):
            reference.admit_header(header)


def test_moved_adapter_rejects_old_preparation_while_old_copy_exists(prepared, monkeypatch):
    directory, manifest = prepared
    identity = write_manifest(directory, manifest)
    original = ROOT / "tools/runtime_frozen_reference.py"
    copied = directory / "moved_adapter.py"
    copied.write_bytes(original.read_bytes())
    loader = importlib.util.spec_from_file_location("moved_frozen_reference", copied)
    moved = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(moved)
    monkeypatch.setattr(moved, "FROZEN_ONNX", reference.FROZEN_ONNX)
    monkeypatch.setattr(moved, "checkout_commit", reference.checkout_commit)
    assert original.exists() and copied.read_bytes() == original.read_bytes()
    with pytest.raises(ValueError, match="executing adapter"):
        moved.preflight(directory, identity)


def test_loaded_dependency_module_roles_are_checked(monkeypatch):
    original = reference.importlib.util.module_from_spec

    def foreign_comparison(loader):
        module = original(loader)

        class ForeignLoader:
            def exec_module(self, target):
                target.comparison = SimpleNamespace(__file__="foreign_comparison.py")

        loader.loader = ForeignLoader()
        return module

    monkeypatch.setattr(reference.importlib.util, "module_from_spec", foreign_comparison)
    with pytest.raises(ValueError, match="comparison role"):
        reference.load_quality(ROOT)
