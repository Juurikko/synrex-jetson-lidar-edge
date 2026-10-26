# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
# ruff: noqa: E402
"""Point-level U-Net quality and fixed baselines, using validation before model freeze."""

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

from dense_training import add_confusion, blocks, metrics, targets
from geometric_baseline import predict as geometry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "trained", "upstream", "inventory", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--resource-allocation-verified", action="store_true")
    args = parser.parse_args()
    if not args.resource_allocation_verified:
        raise ValueError("verified GPU allocation required")
    checkpoint_path = args.trained / "best.pt"
    checkpoint_sha = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    if args.split == "test":
        if (
            not args.freeze
            or json.loads(args.freeze.read_text())["weights_sha256"] != checkpoint_sha
        ):
            raise ValueError("test evaluation requires matching immutable pre-test model freeze")
    torch.set_num_threads(8)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    net = model(args.upstream).cuda().eval()
    net.load_state_dict(
        torch.load(checkpoint_path, map_location="cuda", weights_only=True)["state_dict"]
    )
    spec = json.loads((ROOT / "examples/unet/dense_voxel.json").read_text())
    manifest = json.loads((args.data / "dataset.json").read_text())
    inventory = json.loads(args.inventory.read_text())
    majority_class = int(np.argmax(inventory["splits"]["train"]["semantic_point_counts"]))
    totals = {
        k: np.zeros((7, 7), dtype=np.int64)
        for k in ["unet", "geometry", "majority", "voxel_upper_bound"]
    }
    rows = []
    for row in manifest["files"][args.split]["tiles"]:
        path = args.data / args.split / Path(row["member"]).name
        native_order = None
        upper = None
        for b in blocks(path, spec):
            with torch.no_grad():
                scores = net(torch.from_numpy(b["features"][None]).cuda()).cpu().numpy()
            if native_order is None:
                native_order = np.full(b["points"], -1, dtype=np.int64)
                upper = native_order.copy()
            native_order[b["ids"]] = scores.reshape(7, -1).argmax(0)[b["linear"]]
            upper[b["ids"]] = np.maximum(b["labels"].reshape(-1)[b["linear"]], 0)
        assert native_order is not None and np.all(native_order >= 0)
        truth = targets(laspy.read(path, laz_backend=laspy.LazBackend.Lazrs).classification)
        per_file = np.zeros((7, 7), dtype=np.int64)
        add_confusion(per_file, truth, native_order)
        totals["unet"] += per_file
        add_confusion(totals["geometry"], truth, geometry(path))
        add_confusion(totals["majority"], truth, np.full(len(truth), majority_class))
        add_confusion(totals["voxel_upper_bound"], truth, upper)
        rows.append(
            {
                "source_path": row["member"],
                "points": len(truth),
                "source_sha256": row["sha256"],
                "unet": metrics(per_file),
            }
        )
    receipt = {
        "status": "PASS",
        "split": args.split,
        "weights_sha256": checkpoint_sha,
        "points": sum(row["points"] for row in rows),
        "tiles": len(rows),
        "metrics": {k: metrics(v) for k, v in totals.items()},
        "files": rows,
        "features": "occupancy/count/local_Z/intensity only; no labels/RGB/NIR",
        "baselines": "Training majority; frozen geometry rule from protocol; no label fitting",
    }
    args.out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    main()
