# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
"""Development-only physical streaming qualification; inference runs in native executables.

Uses existing frozen packages. Never builds engines, changes devices or stops workloads.
Generated licensed point/model artifacts and raw logs stay in the specified output directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import threading
import time
from pathlib import Path, PureWindowsPath


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def query(args):
    return subprocess.check_output(args, text=True, timeout=15).strip()


def gpu_sample():
    row = next(
        csv.reader(
            query(
                [
                    "nvidia-smi",
                    "--query-gpu=name,utilization.gpu,memory.used,memory.free,power.draw",
                    "--format=csv,noheader,nounits",
                ]
            ).splitlines()
        )
    )
    name, utilization, used, free, power = [x.strip() for x in row]
    return {
        "gpu": name,
        "utilization_percent": int(utilization),
        "used_mib": int(used),
        "free_mib": int(free),
        "power_watts": float(power) if power != "[N/A]" else None,
    }


def available():
    workloads = query(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name",
            "--format=csv,noheader",
        ]
    )
    if workloads:
        raise RuntimeError("competing Linux compute process reported; GPU execution NOT_RUN")
    is_wsl = "microsoft" in Path("/proc/sys/kernel/osrelease").read_text().lower()
    windows_root = Path(query(["wslpath", "-u", "C:\\Windows"])) if is_wsl else None
    desktop = []
    if windows_root:
        windows = windows_root / "System32/nvidia-smi.exe"
        if not windows.is_file():
            raise RuntimeError("WSL Windows-native GPU reporting unavailable; execution NOT_RUN")
        rows = csv.reader(
            query(
                [
                    str(windows),
                    "--query-compute-apps=pid,process_name",
                    "--format=csv,noheader",
                ]
            ).splitlines()
        )
        permitted = {
            "dwm",
            "ShellExperienceHost.exe",
            "explorer.exe",
            "ChatGPT.exe",
            "SystemSettings.exe",
            "ShellHost.exe",
            "SearchHost.exe",
            "TextInputHost.exe",
        }
        for pid, process in rows:
            name = PureWindowsPath(process.strip()).name
            if name == "[Insufficient Permissions]":
                pid = int(pid)
                ps = str(windows_root / "System32/WindowsPowerShell/v1.0/powershell.exe")
                name = query([ps, "-NoProfile", "-Command", f"(Get-Process -Id {pid}).ProcessName"])
            if name not in permitted:
                raise RuntimeError(f"unallocated Windows GPU process {name}; execution NOT_RUN")
            desktop.append(name)
    for _ in range(8):
        sample = gpu_sample()
        if sample["free_mib"] < 4096 or "4090" not in sample["gpu"]:
            raise RuntimeError("RTX4090/free-memory precondition failed; GPU execution NOT_RUN")
        if sample["utilization_percent"] <= 5:
            return {
                "sample": sample,
                "windows_desktop_processes": desktop,
                "reported_linux_compute_processes": 0,
            }
        time.sleep(1)  # Only settle trailing utilization after our completed invocation.
    raise RuntimeError("GPU remained busy; execution NOT_RUN")


def qualify(a):
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    native = a.native_dir.resolve()
    binaries = {name: sha(native / name) for name in ["synrex-stream", "synrex-stream-gpu-tests"]}
    artifacts = {}
    for name, model in [("synthetic", a.synthetic_model), ("unet", a.unet_model)]:
        manifest = json.loads((model / "manifest.json").read_text())
        artifacts[name] = {p.name: sha(p) for p in model.iterdir()}
        expected = (
            "4dd03d4c2c6291e98145c64ec2fb7e174ba69d7b7e1728e84cf7357e51772f59"
            if name == "synthetic"
            else "2f20d4fd17c1e1eb3399ac450f84a2f1c7737ab69e2098ae383dc1911a1455cd"
        )
        if manifest["artifacts"]["engine"]["sha256"] != expected:
            raise ValueError("qualification requires the frozen accepted R4 engine")
    report = {
        "schema": "synrex.streaming.campaign.v1",
        "status": "FAIL",
        "source_commit": a.source_commit,
        "binaries_sha256": binaries,
        "model_package_hashes_before": artifacts,
        "commands": [],
        "AGX": "NOT_RUN",
        "THOR": "NOT_RUN",
        "SUSTAINED_B5": "NOT_RUN",
    }

    def command(name, args, expected=0, gpu=False):
        preflight = available() if gpu else None
        samples = []
        stop = threading.Event()

        def sample():
            while not stop.is_set():
                try:
                    samples.append(gpu_sample())
                except (OSError, ValueError, subprocess.SubprocessError):
                    pass
                stop.wait(0.2)

        thread = threading.Thread(target=sample) if gpu else None
        if thread:
            thread.start()
        start = time.monotonic()
        try:
            with (out / f"{name}.log").open("w") as log:
                result = subprocess.run(
                    list(map(str, args)),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env={**os.environ, "PATH": "/nonexistent"}
                    if name == "native-no-python"
                    else None,
                )
        finally:
            stop.set()
            if thread:
                thread.join()
        record = {
            "name": name,
            "status": "PASS" if result.returncode == expected else "FAIL",
            "exit_code": result.returncode,
            "expected_exit_code": expected,
            "wall_seconds": time.monotonic() - start,
            "log_sha256": sha(out / f"{name}.log"),
        }
        if preflight:
            record["resource_preflight"] = preflight
            record["device_samples"] = samples
        report["commands"].append(record)
        if record["status"] != "PASS":
            raise RuntimeError(f"{name} failed; inspect local log")

    try:
        command(
            "equivalence",
            [
                native / "synrex-stream-gpu-tests",
                a.synthetic_model,
                a.unet_model,
                out / "equivalence",
                *a.licensed_input,
            ],
            gpu=True,
        )
        report["equivalence"] = json.loads((out / "equivalence/receipt.json").read_text())
        for model in ["synthetic", "unet"]:
            source = out / f"equivalence/{model}/replay-input.jsonl"
            command(
                f"replay-{model}",
                [
                    native / "synrex-stream",
                    "replay",
                    "--input",
                    source,
                    "--model",
                    a.synthetic_model if model == "synthetic" else a.unet_model,
                    "--out",
                    out / f"{model}-events.jsonl",
                    "--receipt",
                    out / f"{model}-receipt.json",
                    "--map-cells",
                    "65536",
                ],
                gpu=True,
            )
        command(
            "native-no-python",
            [
                native / "synrex-stream",
                "replay",
                "--input",
                out / "equivalence/unet/replay-input.jsonl",
                "--model",
                a.unet_model,
                "--out",
                out / "no-python-events.jsonl",
                "--receipt",
                out / "no-python-receipt.json",
                "--map-cells",
                "65536",
            ],
            gpu=True,
        )
        for points in [256, 4096]:
            source = out / f"density-{points}.jsonl"
            command(
                f"fixture-{points}",
                [
                    native / "synrex-stream",
                    "fixture",
                    "--out",
                    source,
                    "--frames",
                    "12",
                    "--points",
                    points,
                    "--chunk",
                    "256",
                    "--seed",
                    "91",
                ],
            )
            for rate in [100000, 1000000, 10000000]:
                name = f"paced-{points}-{rate}"
                command(
                    name,
                    [
                        native / "synrex-stream",
                        "replay",
                        "--input",
                        source,
                        "--model",
                        a.unet_model,
                        "--out",
                        out / f"{name}-events.jsonl",
                        "--receipt",
                        out / f"{name}-receipt.json",
                        "--rate-pps",
                        rate,
                        "--map-cells",
                        "65536",
                    ],
                    gpu=True,
                )
        source = out / "density-4096.jsonl"
        lines = source.read_text().splitlines()
        for name, fault in [
            ("packet-loss", lines[:3] + lines[4:]),
            ("out-of-order", lines[:2] + [lines[3], lines[2]] + lines[4:]),
        ]:
            bad = out / f"{name}.jsonl"
            bad.write_text("\n".join(fault) + "\n")
            command(
                name,
                [
                    native / "synrex-stream",
                    "replay",
                    "--input",
                    bad,
                    "--model",
                    a.unet_model,
                    "--out",
                    out / f"{name}-events.jsonl",
                    "--receipt",
                    out / f"{name}-receipt.json",
                    "--map-cells",
                    "65536",
                ],
                expected=1,
                gpu=True,
            )
        report["replays"] = {p.stem: json.loads(p.read_text()) for p in out.glob("*-receipt.json")}
        report["model_packages_unchanged"] = all(
            {p.name: sha(p) for p in model.iterdir()} == artifacts[name]
            for name, model in [("synthetic", a.synthetic_model), ("unet", a.unet_model)]
        )
        report["binary_bytes_unchanged"] = all(sha(native / n) == h for n, h in binaries.items())
        if not report["model_packages_unchanged"] or not report["binary_bytes_unchanged"]:
            raise RuntimeError("frozen package or executable bytes changed")
        report["status"] = "PASS"
    except Exception as error:
        report["error"] = str(error)
        raise
    finally:
        (out / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-dir", type=Path, required=True)
    parser.add_argument("--synthetic-model", type=Path, required=True)
    parser.add_argument("--unet-model", type=Path, required=True)
    parser.add_argument("--licensed-input", type=Path, action="append", default=[])
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(qualify(args), indent=2))
