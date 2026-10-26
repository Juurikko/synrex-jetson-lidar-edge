# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
# ruff: noqa: E402
"""Train an authentic pinned MIT pytorch-3dunet UNet3D from scratch on FRACTAL.

Training/export only. The native runtime never imports this module or PyTorch.
"""

import argparse
import copy
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import torch

from dense_training import add_confusion, blocks, metrics

UPSTREAM = "7d047d2802561d075d6ce44679d7e696138c47df"


def model(source):
    actual = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != UPSTREAM or subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain"]
    ):
        raise ValueError("architecture source must be the pristine pinned MIT checkout")
    sys.path.insert(0, str(source.resolve()))
    from pytorch3dunet.unet3d.model import UNet3D

    return UNet3D(
        4,
        7,
        f_maps=[8, 16, 32],
        layer_order="cbr",
        num_levels=3,
        is_segmentation=False,
        dropout_prob=0.0,
    )


def load_split(data, split, spec):
    xs, ys, files = [], [], []
    manifest = json.loads((data / "dataset.json").read_text())
    paths = sorted(
        data / split / Path(row["member"]).name for row in manifest["files"][split]["tiles"]
    )
    for path in paths:
        files.append({"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        for b in blocks(path, spec):
            if np.count_nonzero(b["labels"] >= 0) >= 32:
                xs.append(b["features"])
                ys.append(b["labels"])
    if not xs:
        raise ValueError("empty training/validation split")
    print(f"{split}: {len(files)} tiles, {len(xs)} nontrivial blocks", flush=True)
    return torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys)), files


@torch.no_grad()
def evaluate(net, x, y, batch):
    net.eval()
    confusion = np.zeros((7, 7), dtype=np.int64)
    for start in range(0, len(x), batch):
        predicted = net(x[start : start + batch].cuda()).argmax(1).cpu().numpy()
        add_confusion(confusion, y[start : start + batch].numpy().ravel(), predicted.ravel())
    return metrics(confusion)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument(
        "--balanced-blocks",
        action="store_true",
        help="class-presence balanced sampling, based only on training labels",
    )
    parser.add_argument("--resource-allocation-verified", action="store_true")
    args = parser.parse_args()
    if not args.resource_allocation_verified:
        raise ValueError("verify idle GPU and nonconflicting allocation before training")
    args.out.mkdir(parents=True, exist_ok=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(8)
    # Upstream UNet3D uses MaxPool3d. PyTorch 2.9 has no deterministic CUDA
    # backward for it. Retain the genuine architecture, surface warnings and
    # record this reproducibility limit rather than claiming bitwise training.
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    spec = json.loads((ROOT / "examples/unet/dense_voxel.json").read_text())
    net = model(args.upstream).cuda()
    x, y, train_files = load_split(args.data, "train", spec)
    vx, vy, val_files = load_split(args.data, "val", spec)
    val_frequencies = torch.bincount(vy[vy >= 0], minlength=7)
    if args.balanced_blocks and torch.any(val_frequencies == 0):
        raise ValueError("expanded seven-class campaign requires all seven classes in validation")
    frequencies = torch.bincount(y[y >= 0], minlength=7).float()
    weights = (frequencies.sum() / frequencies.clamp_min(1)).sqrt().clamp_max(20).cuda()
    weights /= weights.mean()
    presence = torch.stack([(y == c).flatten(1).any(1) for c in range(7)], dim=1).float()
    # Half uniform, half equal probability per represented semantic class.
    # Avoid unbounded rare-class weights and retain ordinary background contexts.
    sampling = 0.5 / len(y) + 0.5 * (presence / presence.sum(0).clamp_min(1)).sum(1) / 7
    preparation = {
        "train_tiles": len(train_files),
        "val_tiles": len(val_files),
        "train_blocks": len(x),
        "val_blocks": len(vx),
        "train_voxel_counts": frequencies.tolist(),
        "val_voxel_counts": val_frequencies.tolist(),
        "train_blocks_per_class": presence.sum(0).tolist(),
        "sampling": "half uniform + half class-presence balanced"
        if args.balanced_blocks
        else "uniform",
        "spec": spec,
    }
    (args.out / "preparation.json").write_text(json.dumps(preparation, indent=2) + "\n")
    criterion = torch.nn.CrossEntropyLoss(weight=weights, ignore_index=-100)
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.003, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=0.0001
    )
    best, history = -1, []
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(args.epochs):
        net.train()
        order = (
            torch.multinomial(sampling, len(x), replacement=True)
            if args.balanced_blocks
            else torch.randperm(len(x))
        )
        losses = []
        for start in range(0, len(x), args.batch):
            ids = order[start : start + args.batch]
            bx, by = x[ids].cuda(), y[ids].cuda()
            # Horizontal symmetries preserve all four scalar features and native semantics.
            turns = random.randrange(4)
            bx, by = torch.rot90(bx, turns, (-2, -1)), torch.rot90(by, turns, (-2, -1))
            if random.random() < 0.5:
                bx, by = bx.flip(-1), by.flip(-1)
            optimizer.zero_grad(set_to_none=True)
            # Flatten occupied observations to the deterministic 1-D NLL path.
            # PyTorch 2.9 rejects CUDA NLLLoss2d with strict determinism enabled.
            logits = net(bx).permute(0, 2, 3, 4, 1).reshape(-1, 7)
            loss = criterion(logits, by.reshape(-1))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()
        val = evaluate(net, vx, vy, args.batch)
        record = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)),
            "val_voxel_metrics": val,
            "elapsed_seconds": time.perf_counter() - started,
        }
        history.append(record)
        score = val["macro_f1_all_seven"] if args.balanced_blocks else val["mean_iou_present"]
        if score > best:
            best = score
            torch.save(
                {
                    "state_dict": net.state_dict(),
                    "epoch": epoch + 1,
                    "source_commit": UPSTREAM,
                    "seed": args.seed,
                    "architecture": "UNet3D:f_maps=8,16,32;cbr;3levels;dropout=0",
                },
                args.out / "best.pt",
            )
        (args.out / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "epoch": epoch + 1,
                    "loss": record["train_loss"],
                    "selection_score": score,
                    "val_miou": val["mean_iou_present"],
                    "val_macro_f1": val["macro_f1_all_seven"],
                    "val_common3_iou": val["ground_vegetation_building_mean_iou"],
                    "elapsed": record["elapsed_seconds"],
                }
            ),
            flush=True,
        )
    checkpoint = torch.load(args.out / "best.pt", weights_only=True)
    net.load_state_dict(checkpoint["state_dict"])
    net.eval().cpu()
    example = vx[0:1]
    for precision in ("fp32", "fp16"):
        exported = copy.deepcopy(net).cuda().half() if precision == "fp16" else net
        tensor = example.cuda().half() if precision == "fp16" else example
        torch.onnx.export(
            exported,
            tensor,
            args.out / f"unet-{precision}.onnx",
            input_names=["features"],
            output_names=["scores"],
            opset_version=17,
            dynamo=False,
        )
    receipt = {
        "status": "PASS",
        "architecture_source_commit": UPSTREAM,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "seed": args.seed,
        "training_determinism": (
            "seeded; deterministic algorithms warn_only; MaxPool3d CUDA backward may vary"
        ),
        "epochs": args.epochs,
        "selected_epoch": checkpoint["epoch"],
        "selection_metric": "validation macro-F1 over all seven"
        if args.balanced_blocks
        else "validation mean IoU present",
        "balanced_blocks": args.balanced_blocks,
        "parameters": sum(p.numel() for p in net.parameters()),
        "pretrained_weights": False,
        "split_use": "train optimization; val checkpoint selection; test never loaded by training",
        "train_files": train_files,
        "val_files": val_files,
        "class_voxel_counts": frequencies.tolist(),
        "training_seconds": time.perf_counter() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    (args.out / "training.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
