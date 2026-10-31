# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
"""Bounded frozen R2 U-Net CPU reference; no training or GPU scheduling.

Prepare feature-only Q0/Q1 inputs, then execute through a separately registered
CPU job. Neither the registration string nor a generated trace attests admission.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import itertools
import json
import re
import subprocess
import time
import zipfile
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np

MAX_POINTS = 50_000
MAX_BLOCKS = 64
MAX_ENUMERATED_BLOCKS = 4096
MAX_SOURCE_BYTES = 1024 * 1024
MAX_ARTIFACT_BYTES = 4 * 1024 * 1024
MAX_PREPARED_BYTES = 64 * 1024 * 1024
MAX_OBSERVATION_BYTES = 16 * 1024 * 1024
MAX_MANIFEST_BYTES = 512 * 1024
SCHEMA = "synrex.runtime.frozen_cpu_reference.v2"
FROZEN_ONNX = "316dfd710b455f2324a86e18f39c7df960525fe94fe3702579030f014887fb44"
FROZEN_SPEC = "a954e28a33e3bc12d2e03ce56931acb7a60d1c7cc9c21a0ed0ffa900e22074d1"
LIMITATIONS = [
    "No truth read or quality score produced by this adapter.",
    "Q0 equals Q1 for a source with at most 16384 points; EOF equality cannot test context loss.",
    "FP32 CPU reference differs from native FP16; "
    "matched precision/logit qualification is separate.",
    "CPU feature reconstruction does not attest native preprocessing or sampled prefix causality.",
    "Batch availability follows full decode; timing is not sensor age or streaming capacity.",
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_bytes(path, limit):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "regular nonsymlink artifact required")
    require(path.stat().st_size <= limit, "artifact byte limit exceeded")
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    require(len(data) <= limit, "artifact byte limit exceeded during read")
    return data


def strict_json(data):
    def unique(pairs):
        result = {}
        for name, value in pairs:
            require(name not in result, "duplicate JSON field")
            result[name] = value
        return result

    def invalid(_):
        raise ValueError("nonfinite JSON value")

    return json.loads(data, object_pairs_hook=unique, parse_constant=invalid)


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def jsonl(path, rows):
    with Path(path).open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")


def feature_blocks(xyz, intensity, spec):
    """Feature-only binary64 floor geometry and float32 normalization oracle.

    This mirrors the existing supervised dense_training feature stages without
    exposing truth to inference. It is distinct from native discrete parity.
    """
    require(
        isinstance(xyz, np.ndarray) and xyz.ndim == 2 and xyz.shape[1] == 3,
        "Nx3 coordinates required",
    )
    require(
        xyz.dtype == np.float64 and 0 < len(xyz) <= MAX_POINTS and np.isfinite(xyz).all(),
        "bounded finite binary64 points required",
    )
    require(
        isinstance(intensity, np.ndarray)
        and intensity.dtype == np.uint16
        and intensity.shape == (len(xyz),),
        "uint16 intensity shape required",
    )
    require(
        np.isfinite(intensity).all() and ((intensity >= 0) & (intensity <= 65535)).all(),
        "invalid uint16 intensity values",
    )
    geometry = spec["geometry"]
    require(
        not any(geometry["overlap_xyz"]) and geometry["overlap_policy"] == "first_wins",
        "nonoverlapping frozen geometry required",
    )
    minimum = xyz.min(0)
    extent = np.array(geometry["block_size_xyz"])
    dimensions = np.array(geometry["dimensions_xyz"])
    voxel_size = np.array(geometry["voxel_size_xyz"])
    require(
        np.array_equal(dimensions, [32, 32, 32])
        and np.array_equal(extent, [32, 32, 32])
        and np.array_equal(voxel_size, [1, 1, 1]),
        "frozen dimensions required",
    )
    maximum = np.floor((xyz.max(0) - minimum) / extent)
    require(
        np.isfinite(maximum).all() and np.prod(maximum.astype(object) + 1) <= MAX_ENUMERATED_BLOCKS,
        "block enumeration limit exceeded",
    )
    count = 0
    for index in itertools.product(*(range(int(value) + 1) for value in maximum)):
        origin = minimum + np.array(index) * extent
        local = xyz - origin
        ids = np.flatnonzero(((local >= 0) & (local < extent)).all(1))
        if not len(ids):
            continue
        count += 1
        require(count <= MAX_BLOCKS, "nonempty block limit exceeded")
        vox = np.minimum(np.floor(local[ids] / voxel_size).astype(int), dimensions - 1)
        linear = (vox[:, 2] * dimensions[1] + vox[:, 1]) * dimensions[0] + vox[:, 0]
        size = int(np.prod(dimensions))
        counts = np.bincount(linear, minlength=size)
        occupied = counts > 0
        features = np.zeros((len(spec["features"]), size), dtype=np.float32)
        for channel, feature in enumerate(spec["features"]):
            kind = feature["kind"]
            if kind == "occupancy":
                value = np.ones(size)
            elif kind == "count":
                value = counts
            elif kind in ("mean_intensity", "local_x", "local_y", "local_z"):
                values = (
                    intensity[ids]
                    if kind == "mean_intensity"
                    else local[ids, "xyz".index(kind[-1])]
                )
                value = np.bincount(linear, weights=values, minlength=size) / np.maximum(counts, 1)
            else:
                raise ValueError("unsupported feature")
            features[channel, occupied] = (
                value[occupied].astype(np.float32) - np.float32(feature["offset"])
            ) * np.float32(feature["scale"])
        require(np.isfinite(features).all(), "nonfinite feature tensor")
        yield {
            "features": features.reshape([1, len(spec["features"]), *dimensions[::-1]]),
            "ids": ids,
            "linear": linear,
            "origin": origin.tolist(),
        }


def source_paths(source_checkout):
    """Canonical roles bind the running adapter and the dependencies it loads."""
    root = Path(source_checkout).resolve()
    return {
        "adapter": Path(__file__).resolve(),
        "supervised_baseline": root / "src/dense_training.py",
        "quality": root / "tools/runtime_quality.py",
        "comparison": root / "tools/runtime_comparison.py",
    }


def checkout_commit(source_checkout):
    actual = subprocess.check_output(
        ["git", "-C", str(source_checkout), "rev-parse", "--verify", "HEAD^{commit}"],
        text=True,
    ).strip()
    require(
        bool(re.fullmatch(r"(?:[a-f0-9]{40}|[a-f0-9]{64})", actual)),
        "actual checkout commit required",
    )
    return actual


def load_quality(source_checkout):
    paths = source_paths(source_checkout)
    spec = importlib.util.spec_from_file_location("_frozen_reference_quality", paths["quality"])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(Path(module.__file__).resolve() == paths["quality"], "loaded quality role mismatch")
    require(
        Path(module.comparison.__file__).resolve() == paths["comparison"],
        "loaded comparison role mismatch",
    )
    return module


def boot_identity():
    """Prevent a prepared monotonic timestamp from crossing a machine boot."""
    value = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    require(bool(re.fullmatch(r"[a-f0-9-]{36}", value)), "Linux boot identity unavailable")
    return value


def verify_sources(sources, source_checkout):
    expected = {str(path) for path in source_paths(source_checkout).values()}
    require(
        isinstance(sources, dict) and set(sources) == expected,
        "exact executing adapter and dependency role paths required",
    )
    for path, claimed in sources.items():
        require(
            digest(read_bytes(path, MAX_ARTIFACT_BYTES)) == claimed,
            "executable source identity changed",
        )


def verify_prepared_identity(prepared):
    verify_sources(prepared["source_files"], prepared["source_checkout"])
    require(
        prepared["source_checkout_commit"] == checkout_commit(prepared["source_checkout"]),
        "actual checkout commit changed",
    )
    require(prepared.get("spec_sha256") == FROZEN_SPEC, "frozen spec identity required")
    require(
        digest(read_bytes(prepared["spec_path"], MAX_MANIFEST_BYTES)) == FROZEN_SPEC,
        "actual frozen spec changed",
    )


def local_path(base, name):
    require(
        isinstance(name, str) and name and Path(name).name == name and name not in (".", ".."),
        "artifact basename required",
    )
    path = Path(base) / name
    require(not path.is_symlink(), "artifact symlink rejected")
    return path


def load_block(data, points):
    """Admit archive inflation and all array shapes before downstream indexing."""
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        entries = archive.infolist()
        require(
            len(entries) == 3
            and {entry.filename for entry in entries} == {"features.npy", "ids.npy", "linear.npy"},
            "exact feature archive entries required",
        )
        require(
            sum(entry.file_size for entry in entries) <= MAX_ARTIFACT_BYTES,
            "inflated block byte limit exceeded",
        )
        # A small NPY header can claim a huge shape. Check before np.load allocates.
        for entry in entries:
            with archive.open(entry) as stream:
                version = np.lib.format.read_magic(stream)
                require(version in ((1, 0), (2, 0)), "unsupported NPY header version")
                header_reader = (
                    np.lib.format.read_array_header_1_0
                    if version == (1, 0)
                    else np.lib.format.read_array_header_2_0
                )
                shape, fortran, dtype = header_reader(stream)
                require(not fortran, "C-order arrays required")
                if entry.filename == "features.npy":
                    require(
                        shape == (1, 4, 32, 32, 32) and dtype == np.float32,
                        "invalid feature header",
                    )
                else:
                    require(
                        len(shape) == 1 and 0 < shape[0] <= points and dtype == np.int64,
                        "invalid point/voxel header",
                    )
                require(
                    int(np.prod(shape)) * dtype.itemsize == entry.file_size - stream.tell(),
                    "array payload/header length mismatch",
                )
    with np.load(io.BytesIO(data), allow_pickle=False) as arrays:
        features, ids, linear = arrays["features"], arrays["ids"], arrays["linear"]
    require(
        features.shape == (1, 4, 32, 32, 32)
        and features.dtype == np.float32
        and np.isfinite(features).all(),
        "invalid FP32 feature tensor",
    )
    require(
        ids.ndim == 1
        and ids.dtype == np.int64
        and 0 < len(ids) <= points
        and ((ids >= 0) & (ids < points)).all(),
        "invalid original point indices",
    )
    require(np.all(np.diff(ids) > 0), "point indices must be strictly increasing")
    require(
        linear.shape == ids.shape
        and linear.dtype == np.int64
        and ((linear >= 0) & (linear < 32768)).all(),
        "invalid voxel indices",
    )
    return features, ids, linear


def preflight(prepared_dir, expected_sha256):
    """Verify every frozen input before creating a session or output directory."""
    raw = read_bytes(Path(prepared_dir) / "PREPARED.json", MAX_MANIFEST_BYTES)
    require(digest(raw) == expected_sha256, "prepared manifest identity mismatch")
    prepared = strict_json(raw)
    require(
        prepared.get("schema") == SCHEMA and prepared.get("state") == "PREPARED_NOT_INFERRED",
        "unsupported prepared manifest",
    )
    points = prepared.get("points")
    require(type(points) is int and 0 < points <= MAX_POINTS, "prepared point bound exceeded")
    require(prepared.get("clock_boot_id") == boot_identity(), "prepared clock boot changed")
    require(
        bool(re.fullmatch(r"[a-f0-9]{64}", prepared["source_sha256"])), "source SHA256 required"
    )
    verify_prepared_identity(prepared)
    require(prepared["onnx_sha256"] == FROZEN_ONNX, "frozen model identity required")
    model_bytes = read_bytes(prepared["onnx"], MAX_ARTIFACT_BYTES)
    require(digest(model_bytes) == FROZEN_ONNX, "frozen model changed")
    obs_info = prepared["observations"]
    obs_bytes = read_bytes(local_path(prepared_dir, obs_info["path"]), MAX_OBSERVATION_BYTES)
    require(digest(obs_bytes) == obs_info["sha256"], "observation identity mismatch")
    observations = [strict_json(line) for line in obs_bytes.splitlines()]
    quality = load_quality(prepared["source_checkout"])
    quality.validate_observations(
        observations, [{"sha256": prepared["source_sha256"], "points": points}]
    )
    expected = [
        (policy, window)
        for policy in ("q0", "q1")
        for window in quality.build_reference_windows(observations, policy)
    ]
    records = prepared["windows"]
    require(
        isinstance(records, list) and len(records) == len(expected),
        "Q0/Q1 window coverage mismatch",
    )
    total_bytes, total_blocks = len(obs_bytes) + len(raw), 0
    names = {"PREPARED.json", obs_info["path"]}
    for record, (policy, window) in zip(records, expected, strict=True):
        require(
            record["policy"] == policy and record["window"] == window, "window lineage mismatch"
        )
        require(
            isinstance(record["blocks"], list) and 1 <= len(record["blocks"]) <= MAX_BLOCKS,
            "bounded block list required",
        )
        coverage = np.zeros(window["end"] - window["begin"], dtype=np.uint8)
        for block in record["blocks"]:
            total_blocks += 1
            require(total_blocks <= MAX_BLOCKS, "aggregate block bound exceeded")
            name = block["path"]
            require(name not in names, "duplicate artifact path")
            names.add(name)
            data = read_bytes(local_path(prepared_dir, name), MAX_ARTIFACT_BYTES)
            total_bytes += len(data)
            require(total_bytes <= MAX_PREPARED_BYTES, "aggregate prepared byte limit exceeded")
            require(digest(data) == block["sha256"], "frozen feature block mismatch")
            features, ids, _ = load_block(data, len(coverage))
            require(
                digest(features.tobytes()) == block["features_f32_sha256"],
                "feature payload identity mismatch",
            )
            coverage[ids] += 1
        require(np.all(coverage == 1), "exact original-point block coverage required")
    return prepared, model_bytes


def admit_header(header):
    """Bound decoded record storage before asking laspy to allocate point arrays."""
    points = int(header.point_count)
    record_bytes = {6: 30, 7: 36, 8: 38}
    require(str(header.version) == "1.4", "LAS 1.4 source required")
    require(header.point_format.id in record_bytes, "PDRF6/7/8 source required")
    require(
        header.point_format.size == record_bytes[header.point_format.id],
        "extra point dimensions outside frozen reference admission",
    )
    require(0 < points <= MAX_POINTS, "full-source point limit exceeded; sampling forbidden")
    return points


def prepare(args):
    import laspy

    source_bytes = read_bytes(args.source, MAX_SOURCE_BYTES)
    require(digest(source_bytes) == args.source_sha256, "source identity mismatch")
    model_bytes, spec_bytes = (
        read_bytes(args.onnx, MAX_ARTIFACT_BYTES),
        read_bytes(args.spec, MAX_MANIFEST_BYTES),
    )
    require(
        digest(model_bytes) == FROZEN_ONNX and digest(spec_bytes) == FROZEN_SPEC,
        "frozen R2 ONNX/spec required",
    )
    spec = strict_json(spec_bytes)
    source_checkout = args.source_checkout.resolve()
    sources = {
        str(path): digest(read_bytes(path, MAX_ARTIFACT_BYTES))
        for path in source_paths(source_checkout).values()
    }
    # Decode the exact captured bytes that were hashed; never reopen a mutable source.
    with laspy.open(io.BytesIO(source_bytes), laz_backend=laspy.LazBackend.Lazrs) as reader:
        points = admit_header(reader.header)
        cloud = reader.read()
    require(len(cloud) == points, "decoded/header point count mismatch")
    xyz = np.column_stack([np.asarray(cloud.x), np.asarray(cloud.y), np.asarray(cloud.z)])
    intensity = np.asarray(cloud.intensity).copy()
    raw_xyz = np.column_stack([cloud.X, cloud.Y, cloud.Z])
    scales, offsets = cloud.header.scales.copy(), cloud.header.offsets.copy()
    del cloud  # No classification field is read.
    available = time.monotonic_ns()
    with localcontext() as context:
        context.prec = 100
        decimal_xyz = [
            [
                format(
                    Decimal(int(value)) * Decimal(repr(float(scale)))
                    + Decimal(repr(float(offset))),
                    "f",
                )
                for value, scale, offset in zip(row, scales, offsets, strict=True)
            ]
            for row in raw_xyz
        ]
    observations = [
        {
            "source_sha256": args.source_sha256,
            "point_index": index,
            "available_ns": available,
            "captured_ns": None,
            "xyz": row,
        }
        for index, row in enumerate(decimal_xyz)
    ]
    quality = load_quality(source_checkout)
    quality.validate_observations(observations, [{"sha256": args.source_sha256, "points": points}])
    require(not args.out.exists(), "new preparation directory required")
    args.out.mkdir()
    jsonl(args.out / "observations.jsonl", observations)
    observation_bytes = read_bytes(args.out / "observations.jsonl", MAX_OBSERVATION_BYTES)
    total_bytes, total_blocks, records = len(observation_bytes), 0, []
    for policy in ("q0", "q1"):
        for window in quality.build_reference_windows(observations, policy):
            begin, end = window["begin"], window["end"]
            blocks = []
            for number, block in enumerate(
                feature_blocks(xyz[begin:end], intensity[begin:end], spec)
            ):
                total_blocks += 1
                require(total_blocks <= MAX_BLOCKS, "aggregate block bound exceeded")
                buffer = io.BytesIO()
                np.savez(
                    buffer, features=block["features"], ids=block["ids"], linear=block["linear"]
                )
                data = buffer.getvalue()
                total_bytes += len(data)
                require(
                    len(data) <= MAX_ARTIFACT_BYTES
                    and total_bytes <= MAX_PREPARED_BYTES - MAX_MANIFEST_BYTES,
                    "prepared artifact byte bound exceeded",
                )
                name = f"{policy}-w{window['window_index']}-b{number}.npz"
                with (args.out / name).open("xb") as stream:
                    stream.write(data)
                blocks.append(
                    {
                        "path": name,
                        "sha256": digest(data),
                        "origin_binary64": block["origin"],
                        "features_f32_sha256": digest(block["features"].tobytes()),
                    }
                )
            records.append({"policy": policy, "window": window, "blocks": blocks})
    verify_sources(sources, source_checkout)
    prepared = {
        "schema": SCHEMA,
        "state": "PREPARED_NOT_INFERRED",
        "source_checkout": str(source_checkout),
        "source_checkout_commit": checkout_commit(source_checkout),
        "source_files": sources,
        "source_sha256": args.source_sha256,
        "source_bytes": len(source_bytes),
        "spec_sha256": digest(spec_bytes),
        "spec_path": str(args.spec.resolve()),
        "onnx": str(args.onnx.resolve()),
        "onnx_sha256": FROZEN_ONNX,
        "observations": {"path": "observations.jsonl", "sha256": digest(observation_bytes)},
        "points": points,
        "clock_boot_id": boot_identity(),
        "windows": records,
        "precision": (
            "binary64 LAS scaled coordinates/floor and reduction; "
            "float32 feature affine/ONNX FP32 CPU"
        ),
        "clock_scope": (
            "actual monotonic batch availability after full decode; no sensor streaming timing"
        ),
        "limitations": LIMITATIONS,
    }
    save(args.out / "PREPARED.json", prepared)
    prepared_sha256 = digest(read_bytes(args.out / "PREPARED.json", MAX_MANIFEST_BYTES))
    preflight(args.out, prepared_sha256)
    print(
        json.dumps(
            {"state": prepared["state"], "points": points, "prepared_sha256": prepared_sha256}
        )
    )


def infer(args):
    require(
        isinstance(args.registered_cpu_job_id, str)
        and re.fullmatch(r"[A-Za-z0-9_-]{1,128}", args.registered_cpu_job_id),
        "primary CPU job id required; this string does not prove registration",
    )
    require(not args.out.exists(), "new execution output directory required")
    prepared, model_bytes = preflight(args.prepared, args.prepared_sha256)
    # Lazy import and CPU provider only: fixtures and prepare never load a model.
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads = 2, 1
    session = ort.InferenceSession(model_bytes, options, providers=["CPUExecutionProvider"])
    require(session.get_providers() == ["CPUExecutionProvider"], "exclusive CPU provider required")
    args.out.mkdir()
    results = []
    for record in prepared["windows"]:
        window = record["window"]
        predictions = np.full(window["end"] - window["begin"], -1, dtype=np.int64)
        admitted, blocks = time.monotonic_ns(), []
        require(admitted >= window["cutoff_ns"], "shared monotonic clock discontinuity")
        for number, block in enumerate(record["blocks"]):
            data = read_bytes(local_path(args.prepared, block["path"]), MAX_ARTIFACT_BYTES)
            require(digest(data) == block["sha256"], "feature changed after preflight")
            tensor, ids, linear = load_block(data, len(predictions))
            logits = session.run(["scores"], {"features": tensor})[0]
            require(
                logits.dtype == np.float32
                and logits.shape == (1, 7, 32, 32, 32)
                and np.isfinite(logits).all(),
                "invalid FP32 logits",
            )
            require(np.all(predictions[ids] < 0), "duplicate point coverage")
            predictions[ids] = logits.reshape(7, -1).argmax(0)[linear]
            name = f"{record['policy']}-w{window['window_index']}-b{number}-logits.npy"
            with (args.out / name).open("xb") as stream:
                np.save(stream, logits, allow_pickle=False)
            blocks.append(
                {
                    "path": name,
                    "sha256": digest(read_bytes(args.out / name, MAX_ARTIFACT_BYTES)),
                    "logits_f32_sha256": digest(logits.tobytes()),
                }
            )
        require(np.all(predictions >= 0), "incomplete original-point output")
        rows = [
            {
                "source_sha256": window["source_sha256"],
                "point_index": window["begin"] + index,
                "window_index": window["window_index"],
                "label": int(label),
            }
            for index, label in enumerate(predictions)
        ]
        name = f"{record['policy']}-w{window['window_index']}-predictions.jsonl"
        jsonl(args.out / name, rows)
        output = time.monotonic_ns()  # Prediction file has closed before output is attested.
        results.append(
            {
                "policy": record["policy"],
                "window": window,
                "admitted_ns": admitted,
                "output_ns": output,
                "predictions": {
                    "path": name,
                    "sha256": digest(read_bytes(args.out / name, MAX_OBSERVATION_BYTES)),
                },
                "logits": blocks,
            }
        )
    verify_prepared_identity(prepared)
    require(
        digest(read_bytes(args.prepared / "PREPARED.json", MAX_MANIFEST_BYTES))
        == args.prepared_sha256,
        "prepared identity changed during execution",
    )
    save(
        args.out / "REFERENCE_RECEIPT.json",
        {
            "schema": "synrex.runtime.frozen_cpu_reference_execution.v2",
            "status": "CPU_REFERENCE_EXECUTED",
            "registered_cpu_job_id": args.registered_cpu_job_id,
            "gpu_used": False,
            "providers": session.get_providers(),
            "onnxruntime_version": ort.__version__,
            "numpy_version": np.__version__,
            "prepared_sha256": args.prepared_sha256,
            "windows": results,
            "quality_gate": "NOT_ASSESSED; no truth/native scoring",
            "limitations": LIMITATIONS,
        },
    )
    print(json.dumps({"status": "CPU_REFERENCE_EXECUTED", "windows": len(results)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    prep = commands.add_parser("prepare")
    for name in ("source", "spec", "onnx", "out"):
        prep.add_argument("--" + name, type=Path, required=True)
    prep.add_argument("--source-checkout", type=Path, default=Path(__file__).resolve().parents[1])
    prep.add_argument("--source-sha256", required=True)
    execution = commands.add_parser("infer")
    execution.add_argument("--prepared", type=Path, required=True)
    execution.add_argument("--prepared-sha256", required=True)
    execution.add_argument("--registered-cpu-job-id", required=True)
    execution.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    (prepare if args.operation == "prepare" else infer)(args)


if __name__ == "__main__":
    main()
