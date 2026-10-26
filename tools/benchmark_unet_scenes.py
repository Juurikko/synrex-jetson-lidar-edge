# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
# ruff: noqa: E402
"""Untraced application and persistent-context benchmarks on frozen real scenes."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import laspy
import numpy as np

from verify_laz import verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("native", "data", "qualified", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--resource-allocation-verified", action="store_true")
    args = parser.parse_args()
    if not args.resource_allocation_verified:
        raise ValueError("verified allocation required")
    args.out.mkdir(parents=True, exist_ok=False)
    q = json.loads((args.qualified / "qualification.json").read_text())
    ordered = sorted(
        (r for r in q["files"] if r["native"]["points"] >= 50000),
        key=lambda r: r["native"]["points"],
    )
    if len(ordered) < 3:
        raise ValueError("need three nontrivial >=50000-point real benchmark scenes")
    candidates = [
        ordered[0],
        ordered[len(ordered) // 2],
        ordered[-1],
        max(ordered, key=lambda r: r["semantic_support"][5]),
    ]
    selected = {r["source"]: r for r in candidates}
    package = args.qualified / "unet.srxmodel"
    records = []
    for name, row in selected.items():
        source = args.data / "test" / name
        directory = args.out / source.stem
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools/benchmark_native.py"),
                "--native",
                str(args.native),
                "--model",
                str(package),
                "--input",
                str(source),
                "--out",
                str(directory),
                "--runs",
                "3",
                "--resource-allocation-verified",
            ],
            stdout=(args.out / (source.stem + ".log")).open("w"),
            stderr=subprocess.STDOUT,
            check=True,
        )
        cold = json.loads((directory / "benchmark.json").read_text())
        independent = [
            verify(source, directory / f"run-{i}.laz", args.qualified / (source.stem + ".u8"))
            for i in range(3)
        ]
        warm_path = args.out / (source.stem + "-warm.json")
        completed = subprocess.run(
            [
                str(args.native.resolve()),
                "benchmark",
                "--model",
                str(package),
                "--input",
                str(source),
                "--out",
                str(warm_path),
                "--warmup",
                "3",
                "--iterations",
                "10",
            ],
            env=dict(os.environ, PATH="/nonexistent", PYTHONPATH=""),
            capture_output=True,
            text=True,
            check=True,
        )
        warm = json.loads(completed.stdout)
        assert (
            warm["source_order_labels_sha256"]
            == hashlib.sha256((args.qualified / (source.stem + ".u8")).read_bytes()).hexdigest()
        )
        cycles = [c for c in warm["cycles"] if not c["warmup"]]
        assert len(cycles) == 10 and all(
            c["point_coverage"] == row["native"]["points"] for c in cycles
        )
        with laspy.open(source, laz_backend=laspy.LazBackend.Lazrs) as reader:
            h = reader.header
            extent = (h.maxs - h.mins).tolist()
        mean_cold = {
            k: float(np.mean([c["native"]["timings_seconds"][k] for c in cold["records"]]))
            for k in cold["records"][0]["native"]["timings_seconds"]
        }
        mean_warm = {
            k: float(np.mean([c[k] for c in cycles]))
            for k in [
                "wall_seconds",
                "cuda_preparation_seconds",
                "tensorrt_seconds",
                "cuda_postprocessing_seconds",
            ]
        }
        records.append(
            {
                "source": name,
                "points": row["native"]["points"],
                "extent_xyz_m": extent,
                "semantic_support": row["semantic_support"],
                "cold": cold,
                "cold_independent": independent,
                "persistent": warm,
                "mean_cold_seconds": mean_cold,
                "mean_persistent_seconds": mean_warm,
                "mean_process_launch_to_exit_seconds": float(
                    np.mean([c["process_launch_to_exit_seconds"] for c in cold["records"]])
                ),
                "cold_points_per_second": row["native"]["points"] / mean_cold["end_to_end"],
                "persistent_processing_points_per_second_excludes_io": row["native"]["points"]
                / mean_warm["wall_seconds"],
            }
        )
        print("benchmarked", name, flush=True)
    result = {
        "status": "PASS",
        "selection": (
            "Among >=50000-point tiles: smallest, median, largest and highest bridge support; "
            "chosen after model freeze"
        ),
        "native_sha256": hashlib.sha256(args.native.read_bytes()).hexdigest(),
        "warm_scope": (
            "Persistent native engine/context/buffers/decoded cloud; includes block orchestration "
            "and reconstruction, excludes LAZ I/O and cached geometry"
        ),
        "peak_memory_scope": (
            "Exact peak requested bytes via shared TensorRT/adapter allocator; "
            "sampled maximum full device memory separately; "
            "context/untracked vendor allocations excluded from exact counter"
        ),
        "records": records,
    }
    (args.out / "scenes.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
