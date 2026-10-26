# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
# ruff: noqa: E402
"""Recompute frozen test metrics/disagreements; never trains or selects a model."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]
import laspy
import numpy as np
import torch
from train_unet import model

from dense_training import blocks, targets


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def measures(matrix):
    # Independent implementation rather than reusing training metric aggregation.
    matrix = np.asarray(matrix, dtype=np.int64)
    tp = matrix.diagonal().astype(float)
    support, predicted = matrix.sum(1), matrix.sum(0)
    iou = tp / np.maximum(support + predicted - tp, 1)
    f1 = 2 * tp / np.maximum(support + predicted, 1)
    return dict(
        confusion=matrix.tolist(),
        support=support.tolist(),
        iou=iou.tolist(),
        f1=f1.tolist(),
        mean_iou=float(iou.mean()),
        macro_f1=float(f1.mean()),
        accuracy=float(tp.sum() / matrix.sum()),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["data", "trained", "upstream", "qualified", "traces", "freeze", "out"]:
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--resource-allocation-verified", action="store_true")
    a = p.parse_args()
    if not a.resource_allocation_verified:
        raise ValueError("verified idle GPU allocation required")
    freeze = json.loads(a.freeze.read_text())
    assert sha(a.trained / "best.pt") == freeze["weights_sha256"]
    spec = json.loads((a.qualified / "unet.srxmodel/manifest.json").read_text())["contract"]
    torch.set_num_threads(8)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    net = model(a.upstream).cuda().eval()
    net.load_state_dict(
        torch.load(a.trained / "best.pt", map_location="cuda", weights_only=True)["state_dict"]
    )
    matrix = {k: np.zeros((7, 7), np.int64) for k in ["native_fp16", "torch_fp32"]}
    transition = np.zeros((7, 7), np.int64)
    disagreed_truth = np.zeros(7, np.int64)
    all_disagreements = 0
    rows = []
    files = json.loads((a.data / "dataset.json").read_text())["files"]["test"]["tiles"]
    with torch.no_grad():
        for row in files:
            source = a.data / "test" / Path(row["member"]).name
            assert sha(source) == row["sha256"]
            semantics = a.qualified / (source.stem + "-semantics.json")
            info = json.loads(semantics.read_text())
            assert info["source_sha256"] == row["sha256"]
            assert info["model_manifest_sha256"] == sha(a.qualified / "unet.srxmodel/manifest.json")
            native = np.asarray(info["semantic_ids"], dtype=np.uint8)
            reference = np.full(len(native), 255, np.uint8)
            trace = a.traces / (source.stem + "-trace")
            traced = json.loads((trace / "manifest.json").read_text())
            for block, record in zip(blocks(source, spec), traced["blocks"], strict=True):
                x = block["features"][None]
                actual = trace / record["input"]
                assert sha(actual) == record["input_sha256"]
                assert actual.read_bytes() == x.astype(np.float16).tobytes()
                logits = net(torch.from_numpy(x).cuda()).cpu().numpy()
                reference[block["ids"]] = logits.reshape(7, -1).argmax(0)[block["linear"]]
                actual_logits = trace / record["logits"]
                assert sha(actual_logits) == record["logits_sha256"]
                pred = np.fromfile(actual_logits, dtype=np.float16).reshape(7, -1).argmax(0)
                assert np.array_equal(native[block["ids"]], pred[block["linear"]])
            assert np.all(reference < 7) and np.all(native < 7)
            truth = targets(laspy.read(source, laz_backend=laspy.LazBackend.Lazrs).classification)
            valid = truth >= 0
            different = native != reference
            all_disagreements += int(different.sum())
            disagreed_truth += np.bincount(truth[valid & different], minlength=7)
            transition += np.bincount(reference.astype(int) * 7 + native, minlength=49).reshape(
                7, 7
            )
            for key, values in [("native_fp16", native), ("torch_fp32", reference)]:
                matrix[key] += np.bincount(truth[valid] * 7 + values[valid], minlength=49).reshape(
                    7, 7
                )
            rows.append(
                dict(
                    source=source.name,
                    source_sha256=sha(source),
                    fp32_source_order_ids_sha256=hashlib.sha256(reference.tobytes()).hexdigest(),
                    fp16_source_order_ids_sha256=hashlib.sha256(native.tobytes()).hexdigest(),
                    point_disagreements=int(different.sum()),
                )
            )
            print("audited", len(rows), source.name, flush=True)
    support = matrix["native_fp16"].sum(1)
    measured = {key: measures(value) for key, value in matrix.items()}
    accepted = json.loads((a.traces / "qualification.json").read_text())
    for key, value in matrix.items():
        assert value.tolist() == accepted["point_metrics"][key]["confusion"]
    assert all_disagreements == accepted["numerical"]["point_disagreements"]
    result = dict(
        status="PASS",
        scope=(
            "independent aggregate implementation; frozen PyTorch rerun; "
            "actual native trace/sidecar verification"
        ),
        frozen_weights_sha256=sha(a.trained / "best.pt"),
        metrics=measured,
        disagreements=all_disagreements,
        total_points=int(transition.sum()),
        fp32_rows_fp16_columns_transition=transition.tolist(),
        disagreement_by_truth_class=disagreed_truth.tolist(),
        disagreement_rate_by_truth_class=(disagreed_truth / support).tolist(),
        iou_fp16_minus_fp32=(
            np.asarray(measured["native_fp16"]["iou"]) - measured["torch_fp32"]["iou"]
        ).tolist(),
        files=rows,
        no_training_or_model_selection=True,
    )
    a.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
