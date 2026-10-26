# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
"""Qualify real native GPU inference on >=50,000 original synthetic points.

The CPU oracle supplies expected values only. There is no CPU inference fallback.
--prepare-only exports and validates fixtures without making any GPU qualification claim.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))
from export_reference import export  # noqa: E402

from byom_reference import create_dense_scene, expected_labels  # noqa: E402
from verify_laz import verify  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def qualify(args) -> int:
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    records = []
    report = {
        "schema": "synrex.byom.dense_gpu_qualification.v1",
        "points_requested": args.points,
        "allocation": args.verified_gpu_allocation,
        "records": records,
        "SYNTHETIC_ONNX_EXPORT": "NOT_RUN",
        "TENSORRT_ENGINE_BUILD": "NOT_RUN",
        "REAL_GPU_INFERENCE": "NOT_RUN",
        "BYOM_CLASSIFIED_LAZ_E2E": "NOT_RUN",
        "INDEPENDENT_LAZ_VALIDATION": "NOT_RUN",
        "private_model_or_restricted_data_used": False,
    }

    def command(name, argv, *, success=True, env=None):
        start = time.perf_counter()
        result = subprocess.run(
            list(map(str, argv)), env=env, cwd=ROOT, text=True, capture_output=True
        )
        text = (result.stdout + result.stderr).replace(str(ROOT), "<workspace>")
        (out / f"{name}.log").write_text(text)
        records.append(
            {
                "check": name,
                "exit_code": result.returncode,
                "expected_success": success,
                "wall_seconds": time.perf_counter() - start,
                "log_sha256": sha(out / f"{name}.log"),
            }
        )
        if (result.returncode == 0) != success:
            raise RuntimeError(f"{name}: unexpected exit {result.returncode}: {text[-3000:]}")
        return result

    def available(name):
        sample = command(
            name,
            [
                "nvidia-smi",
                "--query-gpu=name,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
        )
        workloads = command(
            name + "-workloads",
            ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
        )
        rows = [s.strip() for s in sample.stdout.splitlines() if s.strip()]
        if len(rows) != 1:
            raise RuntimeError("qualification requires one visible NVIDIA GPU")
        gpu, free, utilization = [s.strip() for s in rows[0].split(",")]
        if workloads.stdout.strip() or int(free) < 2048 or int(utilization) > 5:
            raise RuntimeError("GPU busy or resources uncertain; no competing work started")
        report.setdefault("resource_samples", []).append(
            {
                "gpu": gpu,
                "free_mib": int(free),
                "utilization_percent": int(utilization),
                "reported_compute_processes": [],
            }
        )

    native = args.native.resolve()
    try:
        if args.points < 50000:
            raise ValueError("physical qualification requires at least 50000 points")
        revision = command("source-revision", ["git", "rev-parse", "HEAD"]).stdout.strip()
        report["source_commit"] = revision
        if args.expected_commit and revision != args.expected_commit:
            raise ValueError("source revision differs from transferred commit")
        report["native_binary_sha256"] = sha(native)
        spec = json.loads((ROOT / "examples/dense_voxel.json").read_text())
        spec_path = out / "spec.json"
        spec_path.write_text(json.dumps(spec, indent=2) + "\n")
        model = out / "reference.onnx"
        export(spec_path, model)
        report["SYNTHETIC_ONNX_EXPORT"] = "PASS"
        report["onnx_sha256"] = sha(model)
        source = create_dense_scene(out / "scene", args.points)
        expected, oracle = expected_labels(source, spec, model)
        if set(expected) != {2, 5, 6}:
            raise ValueError("synthetic predictions must exercise every explicit LAS class")
        labels = out / "expected.u8"
        labels.write_bytes(expected.tobytes())
        report["oracle"] = oracle
        report["source_sha256"] = sha(source)
        report["expected_labels_sha256"] = sha(labels)
        onnx_pkg = out / "onnx.srxmodel"
        command(
            "package-onnx",
            [
                native,
                "pack",
                "--spec",
                spec_path,
                "--onnx",
                model,
                "--license",
                ROOT / "LICENSES/MIT.txt",
                "--out",
                onnx_pkg,
            ],
        )
        if args.prepare_only:
            report["status"] = "PREPARATION_PASS_GPU_NOT_RUN"
            return 0
        if not args.verified_gpu_allocation:
            raise ValueError("verified nonconflicting GPU allocation identifier required")
        if not json.loads(command("version", [native, "version"]).stdout)["cuda_tensorrt"]:
            raise ValueError("native executable lacks the real GPU backend")
        available("before-engine-build")
        target = json.loads(command("target", [native, "target"]).stdout)
        report["target"] = target
        engine = out / "reference.engine"
        command("build-engine", [native, "build-engine", "--model", onnx_pkg, "--out", engine])
        report["TENSORRT_ENGINE_BUILD"] = "PASS"
        report["engine_sha256"] = sha(engine)
        engine_pkg = out / "engine.srxmodel"
        command(
            "package-engine",
            [
                native,
                "pack",
                "--spec",
                spec_path,
                "--engine",
                engine,
                "--license",
                ROOT / "LICENSES/MIT.txt",
                "--out",
                engine_pkg,
            ],
        )
        report["engine_manifest"] = json.loads((engine_pkg / "manifest.json").read_text())
        dynamic = command("native-dependencies", ["ldd", native]).stdout
        if "libpython" in dynamic.lower() or "libtorch" in dynamic.lower():
            raise ValueError("native runtime depends on Python or Torch")
        outputs = []
        for index in range(2):
            available(f"before-classify-{index}")
            output = out / f"classified-{index}.laz"
            env = os.environ.copy()
            # The second run cannot discover Python or any other child executable via PATH.
            if index == 1:
                env["PATH"] = "/nonexistent"
                env["LD_DEBUG"] = "libs"
            completed = command(
                f"classify-{index}",
                [native, "classify", "--model", engine_pkg, "--input", source, "--out", output],
                env=env,
            )
            receipt = json.loads(completed.stdout)
            if not (
                receipt["backend"] == "CUDA/TensorRT"
                and receipt["points"] == args.points
                and receipt["completed_inference_blocks"] == receipt["blocks"] == oracle["blocks"]
            ):
                raise ValueError("native execution/block count evidence mismatch")
            for phase in ("cuda_tensor_preparation", "tensorrt_inference", "cuda_postprocessing"):
                if receipt["timings_seconds"][phase] <= 0:
                    raise ValueError(f"missing physical CUDA event measurement: {phase}")
            if index == 1 and any(s in completed.stderr.lower() for s in ("libpython", "libtorch")):
                raise ValueError("Python/Torch library loaded during inference")
            report.setdefault("native_runs", []).append(receipt)
            report.setdefault("independent_validation", []).append(verify(source, output, labels))
            outputs.append(output)
        if sha(outputs[0]) != sha(outputs[1]):
            raise ValueError("repeat classified LAZ not byte-identical")
        report["REAL_GPU_INFERENCE"] = report["BYOM_CLASSIFIED_LAZ_E2E"] = "PASS"
        report["INDEPENDENT_LAZ_VALIDATION"] = "PASS"
        report["repeat_labels_and_laz_identical"] = True
        report["inference_with_empty_executable_path"] = "PASS"
        original = json.loads((engine_pkg / "manifest.json").read_text())
        bad_pkg = out / "bad.srxmodel"
        shutil.copytree(engine_pkg, bad_pkg)
        for kind in (
            "tensor_name",
            "tensor_shape",
            "class_mapping",
            "target",
            "sha256",
            "engine_bytes",
        ):
            bad = copy.deepcopy(original)
            (bad_pkg / "model.engine").write_bytes((engine_pkg / "model.engine").read_bytes())
            if kind == "tensor_name":
                bad["contract"]["input"]["name"] = "wrong_name"
            elif kind == "tensor_shape":
                bad["contract"]["geometry"]["dimensions_xyz"][0] = 9
                bad["contract"]["geometry"]["block_size_xyz"][0] = 9
                bad["contract"]["input"]["shape"][-1] = 9
                bad["contract"]["output"]["shape"][-1] = 9
            elif kind == "class_mapping":
                bad["contract"]["las_classes"][0] = 256
            elif kind == "target":
                bad["target"]["sm"] += 1
            elif kind == "sha256":
                bad["artifacts"]["engine"]["sha256"] = "0" * 64
            elif kind == "engine_bytes":
                (bad_pkg / "model.engine").write_bytes(b"not-a-TensorRT-engine")
                bad["artifacts"]["engine"]["sha256"] = sha(bad_pkg / "model.engine")
            (bad_pkg / "manifest.json").write_text(json.dumps(bad))
            available(f"before-negative-{kind}")
            dest = out / f"rejected-{kind}.laz"
            command(
                f"reject-{kind}",
                [native, "classify", "--model", bad_pkg, "--input", source, "--out", dest],
                success=False,
            )
            if dest.exists():
                raise ValueError("rejected classification published an output")
        command(
            "reject-overwrite",
            [native, "classify", "--model", engine_pkg, "--input", source, "--out", outputs[0]],
            success=False,
        )
        unsupported = out / "unsupported.laz"
        raw = bytearray(source.read_bytes())
        raw[104] = 134
        unsupported.write_bytes(raw)
        command(
            "reject-point-format",
            [
                native,
                "classify",
                "--model",
                engine_pkg,
                "--input",
                unsupported,
                "--out",
                out / "rejected-format.laz",
            ],
            success=False,
        )
        if sha(outputs[0]) != sha(outputs[1]) or (out / "rejected-format.laz").exists():
            raise ValueError("negative tests modified or published output")
        report["negative_cases"] = "PASS"
        available("after-qualification")
        report["status"] = "PASS"
        return 0
    except Exception as error:
        report["status"] = "FAIL"
        report["error"] = str(error)
        print(str(error), file=sys.stderr)
        return 1
    finally:
        (out / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps(
                {k: v for k, v in report.items() if k.isupper() or k in {"status", "error"}},
                indent=2,
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--points", type=int, default=50013)
    parser.add_argument("--verified-gpu-allocation")
    parser.add_argument("--expected-commit")
    parser.add_argument("--prepare-only", action="store_true")
    raise SystemExit(qualify(parser.parse_args()))
