# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
# ruff: noqa: E402
"""Expanded frozen-model qualification using actual native input/logit traces.

No training, hyperparameter selection or CPU inference fallback occurs here.
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
import laspy
import numpy as np
import onnxruntime as ort
import torch
from qualify_unet import TrtOracle, run
from train_unet import model

from dense_training import LAS_CLASSES, add_confusion, blocks, metrics, targets
from geometric_baseline import predict as geometry
from verify_laz import verify


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "native",
        "data",
        "trained",
        "upstream",
        "freeze",
        "inventory",
        "synthetic",
        "out",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--resource-allocation-verified", action="store_true")
    args = parser.parse_args()
    if not args.resource_allocation_verified:
        raise ValueError("verified allocation required")
    freeze = json.loads(args.freeze.read_text())
    assert freeze["weights_sha256"] == sha(args.trained / "best.pt")
    assert freeze["onnx_fp16_sha256"] == sha(args.trained / "unet-fp16.onnx")
    assert freeze["onnx_fp32_sha256"] == sha(args.trained / "unet-fp32.onnx")
    args.out.mkdir(parents=True, exist_ok=False)
    native = args.native.resolve()
    native_sha = sha(native)
    spec_path = ROOT / "examples/unet/dense_voxel.json"
    spec = json.loads(spec_path.read_text())
    notice = ROOT / "examples/unet/MODEL_NOTICE.txt"
    onnx_package, package, engine = [
        args.out / name for name in ("onnx.srxmodel", "unet.srxmodel", "unet.engine")
    ]
    run(
        native,
        "pack",
        "--spec",
        spec_path,
        "--onnx",
        args.trained / "unet-fp16.onnx",
        "--license",
        notice,
        "--out",
        onnx_package,
    )
    build, _ = run(
        native, "build-engine", "--model", onnx_package, "--precision", "fp16", "--out", engine
    )
    (args.out / "engine-build.json").write_text(json.dumps(build, indent=2) + "\n")
    assert "Half" in json.dumps(build["engine_layers"])
    run(
        native,
        "pack",
        "--spec",
        spec_path,
        "--engine",
        engine,
        "--license",
        notice,
        "--out",
        package,
    )
    torch.set_num_threads(8)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    net = model(args.upstream).cuda().eval()
    net.load_state_dict(
        torch.load(args.trained / "best.pt", map_location="cuda", weights_only=True)["state_dict"]
    )
    options = ort.SessionOptions()
    options.intra_op_num_threads = 8
    session = ort.InferenceSession(
        str(args.trained / "unet-fp32.onnx"), options, providers=["CPUExecutionProvider"]
    )
    oracle = TrtOracle(engine)
    totals = {
        name: np.zeros((7, 7), dtype=np.int64)
        for name in ["native_fp16", "torch_fp32", "geometry", "majority", "voxel_upper_bound"]
    }
    inventory = json.loads(args.inventory.read_text())
    majority = int(np.argmax(inventory["splits"]["train"]["semantic_point_counts"]))
    numeric = {
        "native_vs_torch_max_abs": 0.0,
        "torch_vs_onnx_max_abs": 0.0,
        "squared_error": 0.0,
        "logit_values": 0,
        "blocks": 0,
        "point_disagreements": 0,
        "points": 0,
        "native_input_bytes_exact": True,
        "native_vs_binding_logits_exact": True,
    }
    selected = json.loads((args.data / "dataset.json").read_text())["files"]["test"]
    assert selected["archive_url"].endswith("test-01.zip")
    records = []
    environment = dict(os.environ, PATH="/nonexistent", PYTHONPATH="", LD_DEBUG="libs")
    for row in selected["tiles"]:
        source = args.data / "test" / Path(row["member"]).name
        assert sha(source) == row["sha256"]
        stem = source.stem
        output, sidecar, trace = [
            args.out / (stem + suffix)
            for suffix in ("-classified.laz", "-semantics.json", "-trace")
        ]
        receipt, loader = run(
            native,
            "classify",
            "--model",
            package,
            "--input",
            source,
            "--out",
            output,
            "--semantics",
            sidecar,
            "--trace",
            trace,
            environment=environment,
        )
        assert "libpython" not in loader and "libtorch" not in loader
        (args.out / (stem + "-loader.log")).write_text(loader)
        native_trace = json.loads((trace / "manifest.json").read_text())
        semantic = json.loads(sidecar.read_text())
        assert semantic["source_sha256"] == sha(source)
        assert semantic["model_manifest_sha256"] == sha(package / "manifest.json")
        predicted_ids = np.asarray(semantic["semantic_ids"], dtype=np.uint8)
        reference_labels = np.full(len(predicted_ids), 255, dtype=np.uint8)
        fp32_labels = reference_labels.copy()
        upper = reference_labels.astype(np.int64)
        count = 0
        for b, actual in zip(blocks(source, spec), native_trace["blocks"], strict=True):
            count += 1
            assert b["origin"] == actual["origin"]
            input_file, logit_file = trace / actual["input"], trace / actual["logits"]
            assert (
                sha(input_file) == actual["input_sha256"]
                and sha(logit_file) == actual["logits_sha256"]
            )
            x = b["features"][None]
            native_input = np.fromfile(input_file, dtype=np.float16).reshape(spec["input"]["shape"])
            assert np.array_equal(native_input, x.astype(np.float16)), (
                "native feature bytes differ from independent contract"
            )
            native_logits = np.fromfile(logit_file, dtype=np.float16).reshape(
                spec["output"]["shape"]
            )
            with torch.no_grad():
                reference = net(torch.from_numpy(x).cuda()).cpu().numpy()
            exported = session.run(["scores"], {"features": x})[0]
            binding_logits = oracle(b["features"])
            assert np.array_equal(native_logits, binding_logits), (
                "native and binding execution of identical engine differ"
            )
            numeric["torch_vs_onnx_max_abs"] = max(
                numeric["torch_vs_onnx_max_abs"], float(np.max(np.abs(reference - exported)))
            )
            difference = native_logits.astype(np.float64) - reference.astype(np.float64)
            numeric["native_vs_torch_max_abs"] = max(
                numeric["native_vs_torch_max_abs"], float(np.max(np.abs(difference)))
            )
            numeric["squared_error"] += float(np.square(difference).sum())
            numeric["logit_values"] += difference.size
            numeric["blocks"] += 1
            predictions = native_logits.reshape(7, -1).argmax(0)[b["linear"]]
            reference_predictions = reference.reshape(7, -1).argmax(0)[b["linear"]]
            reference_labels[b["ids"]] = predictions
            fp32_labels[b["ids"]] = reference_predictions
            upper[b["ids"]] = np.maximum(b["labels"].reshape(-1)[b["linear"]], 0)
        assert count == receipt["completed_inference_blocks"] == receipt["blocks"]
        assert np.array_equal(reference_labels, predicted_ids)
        assert np.all(predicted_ids < 7)
        mapped = LAS_CLASSES[predicted_ids]
        expected = args.out / (stem + ".u8")
        expected.write_bytes(mapped.tobytes())
        independent = verify(source, output, expected)
        truth = targets(laspy.read(source, laz_backend=laspy.LazBackend.Lazrs).classification)
        for name, values in [
            ("native_fp16", predicted_ids),
            ("torch_fp32", fp32_labels),
            ("geometry", geometry(source)),
            ("majority", np.full(len(truth), majority)),
            ("voxel_upper_bound", upper),
        ]:
            add_confusion(totals[name], truth, values)
        numeric["points"] += len(truth)
        numeric["point_disagreements"] += int(np.count_nonzero(predicted_ids != fp32_labels))
        support = np.bincount(truth[truth >= 0], minlength=7).tolist()
        records.append(
            {
                "source": source.name,
                "source_sha256": sha(source),
                "semantic_support": support,
                "native": receipt,
                "independent": independent,
                "semantic_ids_sha256": sha(sidecar),
            }
        )
        print(
            f"validated {len(records)} native trace/sidecar/LAZ: {source.name}, {len(truth)} points",
            flush=True,
        )
    # Deliberately replace all reference class bytes, then demand identical neural outputs.
    first = args.data / "test" / records[0]["source"]
    changed = laspy.read(first, laz_backend=laspy.LazBackend.Lazrs)
    changed.classification = np.zeros(len(changed), dtype=np.uint8)
    tampered = args.out / "reference-labels-erased.laz"
    changed.write(tampered, laz_backend=laspy.LazBackend.Lazrs)
    altered_output, altered_sidecar = (
        args.out / "erased-classified.laz",
        args.out / "erased-semantics.json",
    )
    altered, _ = run(
        native,
        "classify",
        "--model",
        package,
        "--input",
        tampered,
        "--out",
        altered_output,
        "--semantics",
        altered_sidecar,
        environment=environment,
    )
    prior = json.loads((args.out / (first.stem + "-semantics.json")).read_text())
    assert json.loads(altered_sidecar.read_text())["semantic_ids"] == prior["semantic_ids"]
    verify(tampered, altered_output, args.out / (first.stem + ".u8"))
    repeat = args.out / "repeat.laz"
    repeated, _ = run(
        native,
        "classify",
        "--model",
        package,
        "--input",
        first,
        "--out",
        repeat,
        environment=environment,
    )
    assert sha(repeat) == records[0]["independent"]["output_sha256"]
    original = args.out / "synthetic-same-runtime.laz"
    synthetic, _ = run(
        native,
        "classify",
        "--model",
        args.synthetic / "engine.srxmodel",
        "--input",
        args.synthetic / "scene/dense-synthetic.laz",
        "--out",
        original,
        environment=environment,
    )
    original_verified = verify(
        args.synthetic / "scene/dense-synthetic.laz", original, args.synthetic / "expected.u8"
    )
    assert sha(native) == native_sha
    measured = {name: metrics(value) for name, value in totals.items()}
    learned = measured["native_fp16"]
    coverage = all(n > 0 for n in learned["support"])
    quality = (
        coverage
        and learned["mean_iou_all_seven"] >= 0.40
        and learned["macro_f1_all_seven"]
        >= max(measured[k]["macro_f1_all_seven"] for k in ["majority", "geometry"]) + 0.10
        and all(iou is not None and iou >= 0.05 for iou in learned["iou"])
    )
    numeric["rmse"] = float(np.sqrt(numeric.pop("squared_error") / numeric["logit_values"]))
    numeric["point_label_agreement"] = 1 - numeric["point_disagreements"] / numeric["points"]
    numerical = (
        numeric["rmse"] < 0.03
        and numeric["torch_vs_onnx_max_abs"] < 0.001
        and numeric["point_label_agreement"] >= 0.995
    )
    result = {
        "status": "PASS" if quality and numerical else "FAIL",
        "seven_class_quality_gate": "PASS" if quality else "FAIL",
        "numerical_gate": "PASS" if numerical else "FAIL",
        "full_seven_class_test_coverage": "PASS" if coverage else "NOT_RUN",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "native_binary_before_after_sha256": native_sha,
        "frozen_model": freeze,
        "onnx_fp16_sha256": sha(args.trained / "unet-fp16.onnx"),
        "onnx_fp32_sha256": sha(args.trained / "unet-fp32.onnx"),
        "engine_sha256": sha(engine),
        "numerical": numeric,
        "point_metrics": measured,
        "test_tiles": len(records),
        "test_points": numeric["points"],
        "files": records,
        "native_no_python": "PASS: every native loader log inspected, PATH=/nonexistent",
        "reference_label_erasure_invariance": {"status": "PASS", "native": altered},
        "repeatability": {"status": "PASS", "native": repeated},
        "same_binary_conv3d": {"native": synthetic, "independent": original_verified},
        "timing_scope": (
            "Trace/sidecar/loader I/O included; separate untraced benchmarks required"
        ),
    }
    (args.out / "qualification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "files"}, indent=2), flush=True)
    if result["status"] != "PASS":
        raise SystemExit("Predeclared gates failed; evidence retained without relabeling")


if __name__ == "__main__":
    main()
