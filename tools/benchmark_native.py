# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
"""Bounded fresh-process native timings and sampled device-wide GPU memory.

Use only with an independently verified, nonconflicting NVIDIA GPU allocation.
Sampling is not an exact allocator peak. Python orchestrates but never classifies.
"""

import argparse
import hashlib
import json
import os
import subprocess
import threading
import time
from pathlib import Path


def memory():
    value = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True
    ).strip()
    return int(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("native", "model", "input", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--resource-allocation-verified", action="store_true")
    args = parser.parse_args()
    if not args.resource_allocation_verified or not 1 <= args.runs <= 10:
        raise ValueError("verified allocation and bounded repeat count required")
    other = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"], text=True
    ).strip()
    if other:
        raise ValueError("reported existing compute workloads; do not compete")
    args.out.mkdir(parents=True, exist_ok=False)
    records = []
    for index in range(args.runs):
        baseline = memory()
        samples = []
        stop = threading.Event()

        def sample(stop=stop, samples=samples):
            while not stop.is_set():
                samples.append({"monotonic_seconds": time.perf_counter(), "used_mib": memory()})
                stop.wait(0.02)

        monitor = threading.Thread(target=sample)
        monitor.start()
        try:
            process_start = time.perf_counter()
            result = subprocess.run(
                [
                    str(args.native.resolve()),
                    "classify",
                    "--model",
                    str(args.model),
                    "--input",
                    str(args.input),
                    "--out",
                    str(args.out / f"run-{index}.laz"),
                ],
                env=dict(os.environ, PATH="/nonexistent", PYTHONPATH=""),
                capture_output=True,
                text=True,
                check=True,
            )
            process_seconds = time.perf_counter() - process_start
        finally:
            stop.set()
            monitor.join()
        receipt = json.loads(result.stdout)
        records.append(
            {
                "native": receipt,
                "process_launch_to_exit_seconds": process_seconds,
                "baseline_device_used_mib": baseline,
                "sampled_max_device_used_mib": max(s["used_mib"] for s in samples),
                "samples": samples,
            }
        )
    hashes = {r["native"]["output_sha256"] for r in records}
    assert len(hashes) == 1
    report = {
        "status": "PASS",
        "native_sha256": hashlib.sha256(args.native.read_bytes()).hexdigest(),
        "memory_scope": (
            "nvidia-smi device-wide samples; desktop baseline retained; "
            "sampled maximum is not exact peak"
        ),
        "repeat_output_hashes_identical": True,
        "records": records,
    }
    (args.out / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
