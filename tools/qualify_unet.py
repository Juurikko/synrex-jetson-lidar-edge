# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
# ruff: noqa: E402
"""Genuine trained U-Net FP16 TensorRT/native/LAZ qualification; no mock fallback.

Python, PyTorch and TensorRT bindings are independent development oracles only.
Production classification is a native subprocess with PATH=/nonexistent.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]
import numpy as np
import onnxruntime as ort
import tensorrt as trt
import torch
from train_unet import model

from dense_training import LAS_CLASSES, add_confusion, blocks, metrics
from verify_laz import verify


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(native, *args, environment=None):
    result = subprocess.run(
        [str(native), *map(str, args)], capture_output=True, text=True, env=environment, check=True
    )
    return json.loads(result.stdout), result.stderr


class TrtOracle:
    """Execute the exact native engine on independently prepared tensors."""

    def __init__(self, engine):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine.read_bytes())
        if self.engine is None:
            raise ValueError("TensorRT oracle cannot deserialize real engine")
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()
        self.output = torch.empty((1, 7, 32, 32, 32), dtype=torch.float16, device="cuda")

    def __call__(self, tensor):
        with torch.cuda.stream(self.stream):
            source = torch.from_numpy(tensor[None]).cuda().half().contiguous()
            assert self.context.set_tensor_address("features", source.data_ptr())
            assert self.context.set_tensor_address("scores", self.output.data_ptr())
            assert self.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        return self.output.float().cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("native", "data", "trained", "upstream", "synthetic", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--resource-allocation-verified", action="store_true")
    args = parser.parse_args()
    if not args.resource_allocation_verified:
        raise ValueError("GPU resource allocation must be verified")
    args.out.mkdir(parents=True, exist_ok=False)
    native = args.native.resolve()
    before = digest(native)
    spec = ROOT / "examples/unet/dense_voxel.json"
    contract = json.loads(spec.read_text())
    notice = ROOT / "examples/unet/MODEL_NOTICE.txt"
    onnx_package = args.out / "onnx.srxmodel"
    run(
        native,
        "pack",
        "--spec",
        spec,
        "--onnx",
        args.trained / "unet-fp16.onnx",
        "--license",
        notice,
        "--out",
        onnx_package,
    )
    engine_path = args.out / "unet-fp16.engine"
    build, _ = run(
        native, "build-engine", "--model", onnx_package, "--out", engine_path, "--precision", "fp16"
    )
    (args.out / "engine-build.json").write_text(json.dumps(build, indent=2) + "\n")
    layers_text = json.dumps(build["engine_layers"])
    if "Half" not in layers_text and "FP16" not in layers_text:
        raise ValueError("engine inspector did not report FP16 layer tensors")
    package = args.out / "unet.srxmodel"
    run(
        native,
        "pack",
        "--spec",
        spec,
        "--engine",
        engine_path,
        "--license",
        notice,
        "--out",
        package,
    )
    torch.set_num_threads(8)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    net = model(args.upstream).cuda().eval()
    checkpoint = torch.load(args.trained / "best.pt", map_location="cuda", weights_only=True)
    net.load_state_dict(checkpoint["state_dict"])
    options = ort.SessionOptions()
    options.intra_op_num_threads = 8
    session = ort.InferenceSession(
        str(args.trained / "unet-fp32.onnx"), options, providers=["CPUExecutionProvider"]
    )
    oracle = TrtOracle(engine_path)
    confusion, baseline, upper = [np.zeros((7, 7), dtype=np.int64) for _ in range(3)]
    fp32_confusion = np.zeros((7, 7), dtype=np.int64)
    numeric = {
        "torch_onnx_max_abs": 0.0,
        "fp32_fp16_max_abs": 0.0,
        "squared_error": 0.0,
        "values": 0,
        "point_disagreements": 0,
        "scored_points": 0,
        "compared_points": 0,
        "blocks": 0,
    }
    files = []
    # Majority baseline is selected solely from training counts, never from test labels.
    training = json.loads((args.trained / "training.json").read_text())
    majority = int(np.argmax(training["class_voxel_counts"]))
    for source in sorted((args.data / "test").glob("*.laz")):
        labels = None
        for b in blocks(source, contract):
            x = b["features"]
            with torch.no_grad():
                reference = net(torch.from_numpy(x[None]).cuda()).cpu().numpy()
            onnx_scores = session.run(["scores"], {"features": x[None]})[0]
            predicted = oracle(x)
            numeric["torch_onnx_max_abs"] = max(
                numeric["torch_onnx_max_abs"], float(np.max(np.abs(reference - onnx_scores)))
            )
            error = reference.astype(np.float64) - predicted.astype(np.float64)
            numeric["fp32_fp16_max_abs"] = max(
                numeric["fp32_fp16_max_abs"], float(np.max(np.abs(error)))
            )
            numeric["squared_error"] += float(np.square(error).sum())
            numeric["values"] += error.size
            numeric["blocks"] += 1
            point_predictions = predicted.reshape(7, -1).argmax(0)[b["linear"]]
            fp32_predictions = reference.reshape(7, -1).argmax(0)[b["linear"]]
            numeric["point_disagreements"] += int(
                np.count_nonzero(point_predictions != fp32_predictions)
            )
            numeric["compared_points"] += len(b["ids"])
            numeric["scored_points"] += int(np.count_nonzero(b["truth"] >= 0))
            add_confusion(confusion, b["truth"], point_predictions)
            add_confusion(fp32_confusion, b["truth"], fp32_predictions)
            add_confusion(baseline, b["truth"], np.full(len(b["ids"]), majority))
            voxel_upper = b["labels"].reshape(-1)[b["linear"]].copy()
            voxel_upper[voxel_upper < 0] = 0
            add_confusion(upper, b["truth"], voxel_upper)
            if labels is None:
                labels = np.full(b["points"], 255, dtype=np.uint8)
            labels[b["ids"]] = LAS_CLASSES[point_predictions]
        assert labels is not None and not np.any(labels == 255)
        label_file = args.out / (source.stem + ".u8")
        label_file.write_bytes(labels.tobytes())
        environment = dict(os.environ, PATH="/nonexistent", PYTHONPATH="", LD_DEBUG="libs")
        output = args.out / (source.stem + "-classified.laz")
        receipt, loader = run(
            native,
            "classify",
            "--model",
            package,
            "--input",
            source,
            "--out",
            output,
            environment=environment,
        )
        if "libpython" in loader or "libtorch" in loader:
            raise ValueError("native classification loaded a Python/PyTorch library")
        (args.out / (source.stem + "-loader.log")).write_text(loader)
        assert receipt["completed_inference_blocks"] == receipt["blocks"] > 0
        assert receipt["timings_seconds"]["tensorrt_inference"] > 0
        independent = verify(source, output, label_file)
        files.append(
            {
                "source": source.name,
                "native": receipt,
                "independent": independent,
                "labels_sha256": digest(label_file),
                "predicted_class_histogram": {
                    str(int(k)): int(v)
                    for k, v in zip(*np.unique(labels, return_counts=True), strict=True)
                },
            }
        )
        print(
            f"validated {len(files)}: {source.name}, {len(labels)} original real points", flush=True
        )
    # Fresh native process repeat, source order and every nonclass byte verified again.
    source = sorted((args.data / "test").glob("*.laz"))[0]
    repeat = args.out / "repeat.laz"
    repeat_receipt, _ = run(
        native,
        "classify",
        "--model",
        package,
        "--input",
        source,
        "--out",
        repeat,
        environment=environment,
    )
    verify(source, repeat, args.out / (source.stem + ".u8"))
    assert digest(repeat) == files[0]["independent"]["output_sha256"]
    # Original analytical Conv3D and trained U-Net use the exact same executable.
    synthetic_output = args.out / "synthetic-same-runtime.laz"
    synthetic_receipt, _ = run(
        native,
        "classify",
        "--model",
        args.synthetic / "engine.srxmodel",
        "--input",
        args.synthetic / "synthetic.laz",
        "--out",
        synthetic_output,
        environment=environment,
    )
    synthetic_verify = verify(
        args.synthetic / "synthetic.laz", synthetic_output, args.synthetic / "expected.u8"
    )
    assert before == digest(native)
    numeric["rmse"] = float(np.sqrt(numeric.pop("squared_error") / numeric["values"]))
    numeric["point_label_agreement"] = (
        1 - numeric["point_disagreements"] / numeric["compared_points"]
    )
    quality = metrics(confusion)
    quality_pass = (
        quality["ground_vegetation_building_mean_iou"] >= 0.50
        and quality["accuracy"] > metrics(baseline)["accuracy"] + 0.10
    )
    numerical_pass = (
        numeric["torch_onnx_max_abs"] < 0.001
        and numeric["rmse"] < 0.03
        and numeric["point_label_agreement"] >= 0.995
    )
    times = [f["native"]["timings_seconds"] for f in files]
    points = sum(f["native"]["points"] for f in files)
    elapsed = sum(t["end_to_end"] for t in times)
    receipt = {
        "status": "PASS" if quality_pass and numerical_pass else "FAIL",
        "quality_gate": "common ground/vegetation/building mIoU >=0.50; accuracy > training-majority baseline +0.10",
        "quality_gate_pass": quality_pass,
        "numerical_gate_pass": numerical_pass,
        "native_sha256_before_and_after_model_switch": before,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "onnx_fp32_sha256": digest(args.trained / "unet-fp32.onnx"),
        "onnx_fp16_sha256": digest(args.trained / "unet-fp16.onnx"),
        "engine_sha256": digest(engine_path),
        "numerical": numeric,
        "held_out_point_metrics": quality,
        "fp32_point_metrics": metrics(fp32_confusion),
        "training_majority_baseline": metrics(baseline),
        "voxel_collision_upper_bound": metrics(upper),
        "test_points": points,
        "test_tiles": len(files),
        "class_mapping": LAS_CLASSES.tolist(),
        "native_no_python": "PASS: PATH=/nonexistent, empty PYTHONPATH, loader inspected every test invocation",
        "repeatability": {"status": "PASS", "native": repeat_receipt},
        "same_runtime_original_conv3d": {
            "native": synthetic_receipt,
            "independent": synthetic_verify,
        },
        "mean_file_timings_seconds": {k: float(np.mean([t[k] for t in times])) for k in times[0]},
        "aggregate_points_per_second_including_per_file_initialization": points / elapsed,
        "files": files,
    }
    (args.out / "qualification.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "files"}, indent=2), flush=True)
    if receipt["status"] != "PASS":
        raise SystemExit("qualification gates not met; retain failure evidence")


if __name__ == "__main__":
    main()
