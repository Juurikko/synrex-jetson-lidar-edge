# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
# ruff: noqa: E402
"""Qualify a faithful documented LAS projection of unchanged frozen semantic IDs."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from verify_semantics import verify_semantics


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("native", "data", "qualified", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--resource-allocation-verified", action="store_true")
    args = parser.parse_args()
    if not args.resource_allocation_verified:
        raise ValueError("verified allocation required")
    args.out.mkdir(parents=True, exist_ok=False)
    original = json.loads((args.qualified / "qualification.json").read_text())
    assert original["status"] == "PASS"
    native = args.native.resolve()
    before = sha(native)
    spec = ROOT / "examples/unet/dense_voxel_asprs_safe.json"
    contract = json.loads(spec.read_text())
    package = args.out / "unet.srxmodel"
    engine = args.qualified / "unet.engine"
    assert sha(engine) == original["engine_sha256"]
    subprocess.run(
        [
            str(native),
            "pack",
            "--spec",
            str(spec),
            "--engine",
            str(engine),
            "--license",
            str(ROOT / "examples/unet/MODEL_NOTICE.txt"),
            "--out",
            str(package),
        ],
        check=True,
        capture_output=True,
    )
    env = dict(os.environ, PATH="/nonexistent", PYTHONPATH="", LD_DEBUG="libs")
    records = []
    for row in original["files"]:
        source = args.data / "test" / row["source"]
        stem = source.stem
        output = args.out / (stem + "-classified.laz")
        sidecar = args.out / (stem + "-semantics.json")
        result = subprocess.run(
            [
                str(native),
                "classify",
                "--model",
                str(package),
                "--input",
                str(source),
                "--out",
                str(output),
                "--semantics",
                str(sidecar),
            ],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        assert "libtorch" not in result.stderr and "libpython" not in result.stderr
        receipt = json.loads(result.stdout)
        expected = json.loads((args.qualified / (stem + "-semantics.json")).read_text())
        actual = json.loads(sidecar.read_text())
        assert actual["semantic_ids"] == expected["semantic_ids"], (
            "neural semantic predictions changed"
        )
        labels = bytes(contract["las_classes"][i] for i in actual["semantic_ids"])
        (args.out / (stem + ".u8")).write_bytes(labels)
        independent = verify_semantics(source, output, sidecar, package)
        records.append(
            {
                "source": source.name,
                "source_sha256": sha(source),
                "semantic_support": row["semantic_support"],
                "native": receipt,
                "independent": independent,
                "semantic_ids_unchanged": True,
            }
        )
        print("safe LAS projection", len(records), source.name, flush=True)
    assert sha(native) == before
    final = dict(original)
    final.update(
        files=records,
        original_qualification_source=original["source_commit"],
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        explicit_las_mapping=contract["las_classes"],
        semantic_projection_status="PASS",
        semantic_prediction_changes=0,
        native_binary_before_after_sha256=before,
        projection_notice=(
            "Only code lookup changed; same frozen weights, engine, tensors, geometry, "
            "semantic IDs and point metrics. LAS200/201 user-defined; sidecar authoritative."
        ),
        original_neural_trace_receipt_sha256=sha(args.qualified / "qualification.json"),
    )
    shutil.copy2(
        args.qualified / "synthetic-same-runtime.laz", args.out / "synthetic-same-runtime.laz"
    )
    (args.out / "qualification.json").write_text(json.dumps(final, indent=2) + "\n")
    print("PASS", len(records), original["test_points"], "points; unchanged neural predictions")


if __name__ == "__main__":
    main()
