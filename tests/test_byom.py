# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
"""Native packages/geometry and opt-in physical CUDA/TensorRT LAZ integration.

The CPU oracle is independent of native geometry, kernels and compression.
GPU tests require SYNREX_RUN_GPU=1 after an allocation has been verified.
"""

from __future__ import annotations

import copy
import importlib.util
import itertools
import json
import os
import shutil
import struct
import subprocess
from pathlib import Path

import laspy
import numpy as np
import onnxruntime as ort
import pytest

from synthetic_laz import create
from verify_laz import verify

ROOT = Path(__file__).resolve().parents[1]
NATIVE = Path(os.environ.get("SYNREX_TEST_NATIVE", ROOT / "build/synrex-native")).resolve()
SPEC = json.loads((ROOT / "examples/dense_voxel.json").read_text())
module_spec = importlib.util.spec_from_file_location(
    "reference_export", ROOT / "examples/export_reference.py"
)
reference_export = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(reference_export)
GPU = pytest.mark.skipif(
    os.environ.get("SYNREX_RUN_GPU") != "1", reason="NOT_RUN: no verified GPU resource allocation"
)


def run(*args, ok=True, executable=NATIVE):
    completed = subprocess.run([str(executable), *map(str, args)], capture_output=True, text=True)
    if ok:
        assert completed.returncode == 0, completed.stderr + completed.stdout
    else:
        assert completed.returncode != 0, completed.stdout
    return completed


def package(tmp_path, spec=None, name="fixture"):
    spec = copy.deepcopy(SPEC if spec is None else spec)
    spec_path = tmp_path / f"{name}.json"
    spec_path.write_text(json.dumps(spec))
    onnx = tmp_path / f"{name}.onnx"
    reference_export.export(spec_path, onnx)
    out = tmp_path / f"{name}.srxmodel"
    run(
        "pack",
        "--spec",
        spec_path,
        "--onnx",
        onnx,
        "--license",
        ROOT / "LICENSES/MIT.txt",
        "--out",
        out,
    )
    return out, spec_path, onnx


def scene(tmp_path):
    source, _ = create(tmp_path, 23)
    data = laspy.read(source, laz_backend=laspy.LazBackend.Lazrs)
    # Nonmonotonic source order, exact boundaries, overlap and same-voxel duplicates.
    data.X = np.array(
        [
            0,
            100,
            625,
            1300,
            700,
            725,
            800,
            600,
            799,
            1200,
            2,
            301,
            410,
            550,
            900,
            1000,
            1100,
            1250,
            300,
            305,
            302,
            200,
            1400,
        ]
    )
    data.Y = np.array(
        [
            0,
            100,
            0,
            100,
            0,
            0,
            0,
            0,
            0,
            0,
            200,
            300,
            400,
            500,
            550,
            600,
            650,
            900,
            1,
            1,
            1,
            200,
            100,
        ]
    )
    data.Z = np.array(
        [
            0,
            100,
            200,
            350,
            1,
            1,
            1,
            0,
            0,
            0,
            200,
            300,
            400,
            500,
            550,
            600,
            650,
            900,
            1,
            1,
            1,
            200,
            100,
        ]
    )
    data.intensity = np.array(
        [
            0,
            65535,
            32768,
            0,
            65535,
            0,
            32768,
            65535,
            0,
            32768,
            0,
            65535,
            32768,
            0,
            65535,
            32768,
            0,
            65535,
            0,
            32768,
            65535,
            32768,
            65535,
        ],
        dtype=np.uint16,
    )
    replacement = tmp_path / "scene.laz"
    data.write(replacement, laz_backend=laspy.LazBackend.Lazrs)
    return replacement


def oracle(source, spec, onnx=None):
    data = laspy.read(source, laz_backend=laspy.LazBackend.Lazrs)
    xyz = np.column_stack([np.asarray(data.x), np.asarray(data.y), np.asarray(data.z)])
    minimum = xyz.min(axis=0)
    g = spec["geometry"]
    extent = np.asarray(g["block_size_xyz"])
    stride = extent * (1 - np.asarray(g["overlap_xyz"]))
    dimensions = np.asarray(g["dimensions_xyz"])
    grid_max = np.floor((xyz.max(axis=0) - minimum) / stride).astype(int)
    blocks = []
    labels = np.full(len(xyz), -1, dtype=np.int32)
    session = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"]) if onnx else None
    for index in itertools.product(*(range(int(v) + 1) for v in grid_max)):
        origin = minimum + np.asarray(index) * stride
        local = xyz - origin
        ids = np.flatnonzero(np.all((local >= 0) & (local < extent), axis=1))
        if not len(ids):
            continue
        v = np.minimum(
            np.floor(local[ids] / np.asarray(g["voxel_size_xyz"])).astype(int), dimensions - 1
        )
        linear = (v[:, 2] * dimensions[1] + v[:, 1]) * dimensions[0] + v[:, 0]
        members = sorted(
            zip(ids.tolist(), linear.tolist(), strict=False), key=lambda p: (p[1], p[0])
        )
        blocks.append({"origin": origin.tolist(), "members": [list(p) for p in members]})
        if session is None:
            continue
        tensor = np.zeros(spec["input"]["shape"], dtype=spec["input"]["dtype"])
        flat = tensor.reshape(len(spec["features"]), -1)
        for voxel in np.unique(linear):
            selected = ids[linear == voxel]
            values = {
                "occupancy": 1,
                "count": len(selected),
                "mean_intensity": np.asarray(data.intensity)[selected].astype(np.float64).mean(),
                "local_x": local[selected, 0].mean(),
                "local_y": local[selected, 1].mean(),
                "local_z": local[selected, 2].mean(),
            }
            for channel, f in enumerate(spec["features"]):
                flat[channel, voxel] = (
                    np.float32(values[f["kind"]]) - np.float32(f["offset"])
                ) * np.float32(f["scale"])
        scores = session.run([spec["output"]["name"]], {spec["input"]["name"]: tensor})[0].reshape(
            len(spec["las_classes"]), -1
        )
        predicted = np.asarray(spec["las_classes"], dtype=np.uint8)[
            scores[:, linear].argmax(axis=0)
        ]
        if g["overlap_policy"] == "first_wins":
            take = labels[ids] < 0
            labels[ids[take]] = predicted[take]
        else:
            labels[ids] = predicted
    if session:
        assert (labels >= 0).all()
    return blocks, labels.astype(np.uint8)


def test_reference_export_is_deterministic_nonconstant(tmp_path):
    _, spec_path, model = package(tmp_path)
    other = tmp_path / "second.onnx"
    reference_export.export(spec_path, other)
    assert model.read_bytes() == other.read_bytes()
    source = scene(tmp_path)
    _, labels = oracle(source, SPEC, model)
    assert set(labels) == {2, 5, 6}


def test_overlap_policy_changes_context_dependent_oracle(tmp_path):
    spec = copy.deepcopy(SPEC)
    spec["features"][1] = {"kind": "local_x", "offset": 0, "scale": 0.125}
    _, _, model = package(tmp_path, spec)
    source = scene(tmp_path)
    _, last = oracle(source, spec, model)
    spec["geometry"]["overlap_policy"] = "first_wins"
    _, first = oracle(source, spec, model)
    assert (first != last).any(), "fixture must expose a wrong overlap policy"


def test_signed_coordinates_fractional_grid_and_nonuniform_scale(tmp_path):
    spec = copy.deepcopy(SPEC)
    spec["geometry"]["voxel_size_xyz"] = [0.3, 0.4, 0.2]
    spec["geometry"]["dimensions_xyz"] = [7, 5, 6]
    spec["geometry"]["block_size_xyz"] = [2.1, 2.0, 1.2]
    spec["geometry"]["overlap_xyz"] = [0.3, 0.4, 0.5]
    spec["input"]["shape"] = [1, 2, 6, 5, 7]
    spec["output"]["shape"] = [1, 3, 6, 5, 7]
    pkg, _, _ = package(tmp_path, spec)
    source, _ = create(tmp_path, 31)
    data = laspy.read(source, laz_backend=laspy.LazBackend.Lazrs)
    rng = np.random.default_rng(242)
    for axis in ["X", "Y", "Z"]:
        setattr(data, axis, rng.integers(-200, 201, len(data)))
    altered = tmp_path / "signed.laz"
    data.write(altered, laz_backend=laspy.LazBackend.Lazrs)
    out = tmp_path / "signed-plan.json"
    run("plan", "--model", pkg, "--input", altered, "--out", out)
    expected, _ = oracle(altered, spec)
    assert json.loads(out.read_text())["blocks"] == expected


@pytest.mark.parametrize("policy", ["first_wins", "last_wins"])
def test_geometry_matches_independent_brute_force_source_identity(tmp_path, policy):
    spec = copy.deepcopy(SPEC)
    spec["geometry"]["overlap_policy"] = policy
    spec["geometry"]["overlap_xyz"] = [0.25, 0.5, 0.25]
    pkg, _, _ = package(tmp_path, spec)
    source = scene(tmp_path)
    out = tmp_path / "plan.json"
    run("plan", "--model", pkg, "--input", source, "--out", out)
    expected, _ = oracle(source, spec)
    actual = json.loads(out.read_text())
    assert actual["blocks"] == expected
    assert actual["points"] == 23
    assert actual["memberships"] == sum(len(b["members"]) for b in expected)
    assert {p for b in actual["blocks"] for p, _ in b["members"]} == set(range(23))


@pytest.mark.parametrize(
    "corruption",
    ["artifact", "license", "symlink", "path", "extra", "schema", "target", "duplicate_key"],
)
def test_package_validation_fails_closed(tmp_path, corruption):
    pkg, _, _ = package(tmp_path)
    manifest = pkg / "manifest.json"
    obj = json.loads(manifest.read_text())
    if corruption in {"artifact", "license"}:
        p = pkg / ("model.onnx" if corruption == "artifact" else "MODEL_LICENSE.txt")
        p.write_bytes(p.read_bytes() + b"corrupt")
    elif corruption == "symlink":
        (pkg / "model.onnx").unlink()
        (pkg / "model.onnx").symlink_to(tmp_path / "fixture.onnx")
    elif corruption == "path":
        obj["artifacts"]["onnx"]["path"] = "../fixture.onnx"
    elif corruption == "extra":
        (pkg / "unexpected").write_text("unhashed")
    elif corruption == "schema":
        obj["schema"] = "synrex.model_package.v1"
    elif corruption == "target":
        obj["target"] = {"sm": 87}
    if corruption == "duplicate_key":
        manifest.write_text(
            manifest.read_text().replace('"schema":', '"schema":"duplicate", "schema":')
        )
    else:
        manifest.write_text(json.dumps(obj))
    run("inspect", "--model", pkg, ok=False)


@pytest.mark.parametrize(
    "mutation",
    [
        "shape",
        "layout",
        "dtype",
        "mapping",
        "feature",
        "overlap",
        "extent",
        "budget",
        "unknown",
        "license",
        "fractional_shape",
    ],
)
def test_contract_rejects_unsupported_specs(tmp_path, mutation):
    spec = copy.deepcopy(SPEC)
    if mutation == "shape":
        spec["input"]["shape"][2] = 256
    elif mutation == "layout":
        spec["output"]["layout"] = "NXYZC"
    elif mutation == "dtype":
        spec["input"]["dtype"] = "int8"
    elif mutation == "mapping":
        spec["las_classes"][1] = 256
    elif mutation == "feature":
        spec["features"][0]["kind"] = "unknown"
    elif mutation == "overlap":
        spec["geometry"]["overlap_xyz"][1] = 0.99
    elif mutation == "extent":
        spec["geometry"]["block_size_xyz"][2] = 1
    elif mutation == "budget":
        spec["limits"]["max_tensor_bytes"] = 1
    elif mutation == "unknown":
        spec["anything"] = 1
    elif mutation == "license":
        spec["licensing"]["spdx"] = ""
    elif mutation == "fractional_shape":
        spec["input"]["shape"][2] = 4.5
    spec_path = tmp_path / "bad.json"
    spec_path.write_text(json.dumps(spec))
    model = tmp_path / "placeholder.onnx"
    model.write_bytes(b"must not reach ONNX parser")
    output = tmp_path / "bad.srxmodel"
    run(
        "pack",
        "--spec",
        spec_path,
        "--onnx",
        model,
        "--license",
        ROOT / "LICENSES/MIT.txt",
        "--out",
        output,
        ok=False,
    )
    assert not output.exists()
    assert not list(tmp_path.glob(".srxmodel-*"))


def test_model_free_writer_and_runtime_never_replace_outputs(tmp_path):
    pkg, spec, model = package(tmp_path)
    original = (pkg / "manifest.json").read_bytes()
    run(
        "pack",
        "--spec",
        spec,
        "--onnx",
        model,
        "--license",
        ROOT / "LICENSES/MIT.txt",
        "--out",
        pkg,
        ok=False,
    )
    assert (pkg / "manifest.json").read_bytes() == original
    source = scene(tmp_path)
    output = tmp_path / "existing.laz"
    output.write_bytes(b"keep")
    run("classify", "--model", pkg, "--input", source, "--out", output, ok=False)
    assert output.read_bytes() == b"keep"


@pytest.mark.parametrize("budget", ["max_points", "max_blocks", "max_memberships"])
def test_geometry_resource_limits(tmp_path, budget):
    spec = copy.deepcopy(SPEC)
    spec["limits"][budget] = 1
    pkg, _, _ = package(tmp_path, spec)
    source = scene(tmp_path)
    output = tmp_path / "plan.json"
    run("plan", "--model", pkg, "--input", source, "--out", output, ok=False)
    assert not output.exists()


def test_cpu_build_explicitly_refuses_inference(tmp_path):
    version = json.loads(run("version").stdout)
    if version["cuda_tensorrt"]:
        pytest.skip("CPU-only failure contract")
    pkg, _, _ = package(tmp_path)
    source = scene(tmp_path)
    output = tmp_path / "out.laz"
    result = run("classify", "--model", pkg, "--input", source, "--out", output, ok=False)
    assert "NOT_RUN" in result.stderr and not output.exists()
    run("build-engine", "--model", pkg, "--out", tmp_path / "out.engine", ok=False)


def test_relocated_installation_package_and_geometry(tmp_path):
    install = tmp_path / "install"
    subprocess.run(
        ["cmake", "--install", str(NATIVE.parent), "--prefix", str(install)],
        check=True,
        capture_output=True,
    )
    moved = tmp_path / "relocated"
    shutil.move(install, moved)
    executable = moved / "bin/synrex-native"
    assert not list(moved.rglob("*.pyc"))
    assert not list(moved.rglob("__pycache__"))
    assert (moved / "share/synrex/docs/byom/ADAPTER.md").is_file()
    assert json.loads(run("version", executable=executable).stdout)["runtime"] == "synrex-native-r0"
    pkg, _, _ = package(tmp_path)
    source = scene(tmp_path)
    run(
        "plan",
        "--model",
        pkg,
        "--input",
        source,
        "--out",
        tmp_path / "relocated-plan.json",
        executable=executable,
    )
    dynamic = subprocess.check_output(["readelf", "-d", str(executable)], text=True)
    assert str(ROOT) not in dynamic


@GPU
@pytest.mark.parametrize(
    "policy,dtype,feature,alternate_grid",
    [
        ("first_wins", "float32", "local_x", False),
        ("last_wins", "float32", "local_x", False),
        ("last_wins", "float32", "mean_intensity", False),
        ("last_wins", "float16", "local_x", True),
    ],
)
def test_real_cuda_tensorrt_to_classified_laz(tmp_path, policy, dtype, feature, alternate_grid):
    assert json.loads(run("version").stdout)["cuda_tensorrt"], (
        "GPU qualification requires GPU build"
    )
    spec = copy.deepcopy(SPEC)
    spec["geometry"]["overlap_policy"] = policy
    spec["input"]["dtype"] = spec["output"]["dtype"] = dtype
    if alternate_grid:
        spec["geometry"]["dimensions_xyz"] = [11, 3, 5]
        spec["geometry"]["block_size_xyz"] = [11, 3, 5]
        spec["input"]["shape"] = [1, 2, 5, 3, 11]
        spec["output"]["shape"] = [1, 3, 5, 3, 11]
    # Local x makes predictions depend on overlap context, exposing wrong overlap policy.
    scale = 1 / (65535 if feature == "mean_intensity" else spec["geometry"]["block_size_xyz"][0])
    spec["features"][1] = {"kind": feature, "offset": 0, "scale": scale}
    pkg, spec_path, model = package(tmp_path, spec)
    source = scene(tmp_path)
    engine = tmp_path / "model.engine"
    run("build-engine", "--model", pkg, "--out", engine)
    engine_pkg = tmp_path / "engine.srxmodel"
    run(
        "pack",
        "--spec",
        spec_path,
        "--engine",
        engine,
        "--license",
        ROOT / "LICENSES/MIT.txt",
        "--out",
        engine_pkg,
    )
    _, expected = oracle(source, spec, model)
    assert set(expected) == {2, 5, 6}
    labels = tmp_path / "expected.u8"
    labels.write_bytes(expected.tobytes())
    outputs = [tmp_path / "classified.laz", tmp_path / "repeat.laz"]
    for out in outputs:
        receipt = json.loads(
            run("classify", "--model", engine_pkg, "--input", source, "--out", out).stdout
        )
        assert receipt["backend"] == "CUDA/TensorRT"
        assert verify(source, out, labels)["points"] == len(expected)
    assert outputs[0].read_bytes() == outputs[1].read_bytes()
    # Initialization with a valid package but a wrong tensor contract must fail before output.
    manifest = engine_pkg / "manifest.json"
    original = manifest.read_text()
    bad = json.loads(original)
    bad["contract"]["input"]["name"] = "wrong_name"
    manifest.write_text(json.dumps(bad))
    run(
        "classify",
        "--model",
        engine_pkg,
        "--input",
        source,
        "--out",
        tmp_path / "bad.laz",
        ok=False,
    )
    assert not (tmp_path / "bad.laz").exists()
    bad = json.loads(original)
    bad["target"]["sm"] += 1
    manifest.write_text(json.dumps(bad))
    result = run(
        "classify",
        "--model",
        engine_pkg,
        "--input",
        source,
        "--out",
        tmp_path / "wrong-target.laz",
        ok=False,
    )
    assert "target mismatch" in result.stderr
    manifest.write_text(original)
    # Full relocated inference, with no build-tree executable invocation.
    install = tmp_path / "install"
    subprocess.run(
        ["cmake", "--install", str(NATIVE.parent), "--prefix", str(install)],
        check=True,
        capture_output=True,
    )
    moved = tmp_path / "moved"
    shutil.move(install, moved)
    relocated = tmp_path / "relocated.laz"
    run(
        "classify",
        "--model",
        engine_pkg,
        "--input",
        source,
        "--out",
        relocated,
        executable=moved / "bin/synrex-native",
    )
    assert relocated.read_bytes() == outputs[0].read_bytes()


def test_las_bad_coordinate_scale_is_rejected(tmp_path):
    pkg, _, _ = package(tmp_path)
    source = scene(tmp_path)
    raw = bytearray(source.read_bytes())
    struct.pack_into("<d", raw, 131, float("nan"))
    source.write_bytes(raw)
    output = tmp_path / "bad.json"
    run("plan", "--model", pkg, "--input", source, "--out", output, ok=False)
    assert not output.exists()
