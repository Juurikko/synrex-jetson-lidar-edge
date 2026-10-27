# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
"""Bounded sustained RTX campaign. Python only controls native executables and NVML sampling."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from qualify_streaming import available, gpu_sample, query, sha


def run_trial(
    native,
    model,
    source,
    out,
    *,
    rate,
    mode,
    seconds=30,
    warmup=5,
    points=4096,
    jitter=0,
    overlap=0,
    cancel=0,
    pinned=False,
    motion=0,
    occlusion=0,
    timeline=False,
    geometry=None,
    churn=0,
    map_cells=None,
    verify_map=False,
    map_policy=None,
    map_detail=False,
):
    preflight = available()
    command = [
        str(native),
        "--model",
        str(model),
        "--source",
        str(source),
        "--receipt",
        str(out),
        "--rate",
        str(rate),
        "--seconds",
        str(seconds),
        "--warmup",
        str(warmup),
        "--mode",
        mode,
        "--points",
        str(points),
        "--jitter-us",
        str(jitter),
        "--overlap-replay",
        str(overlap),
        "--cancel-after",
        str(cancel),
    ]
    if pinned:
        command.extend(["--pinned", "1"])
    if motion:
        command.extend(["--motion-mm", str(motion)])
    if occlusion:
        command.extend(["--occlusion-percent", str(occlusion)])
    if timeline:
        command.extend(["--timeline", "1"])
    if geometry is not None:
        command.extend(["--geometry", geometry])
    if churn:
        command.extend(["--churn-mm", str(churn)])
    if map_cells is not None:
        command.extend(["--map-cells", str(map_cells)])
    if verify_map:
        command.extend(["--verify-map", "1"])
    if map_policy is not None:
        command.extend(["--map-policy", map_policy])
    if map_detail:
        command.extend(["--map-detail", "1"])
    samples = []
    log = out.with_suffix(".log")
    begin = time.monotonic()
    hardware_stop = None
    with log.open("w") as sink:
        process = subprocess.Popen(command, stdout=sink, stderr=subprocess.STDOUT)
        while process.poll() is None:
            try:
                sample = gpu_sample()
                clock_values = query(
                    [
                        "nvidia-smi",
                        "--query-gpu=clocks.current.sm,clocks.current.memory",
                        "--format=csv,noheader,nounits",
                    ]
                ).split(",")
                sample["sm_clock_mhz"], sample["memory_clock_mhz"] = [
                    int(x.strip()) for x in clock_values
                ]
                sample["wall_seconds"] = time.monotonic() - begin
                samples.append(sample)
                pids = query(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"])
                if any(int(pid.strip()) != process.pid for pid in pids.splitlines() if pid.strip()):
                    hardware_stop = "competing_compute_process"
                if sample["used_mib"] > 4096:
                    hardware_stop = "device_memory_budget"
                if time.monotonic() - begin > seconds + warmup + 15:
                    hardware_stop = "bounded_trial_timeout"
                if hardware_stop:
                    process.terminate()
                    break
            except (ValueError, OSError, subprocess.SubprocessError) as error:
                hardware_stop = f"resource_monitor_failure:{type(error).__name__}"
                process.terminate()
                break
            time.sleep(0.25)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    receipt = (
        json.loads(out.read_text())
        if out.exists()
        else {"status": "FAIL", "error": "no native receipt"}
    )
    receipt["command"] = [Path(command[0]).name, *command[1:]]
    # Preserve exact argument values except environment-specific paths.
    receipt["command"] = [
        "<model>"
        if x == str(model)
        else "<source>"
        if x == str(source)
        else out.name
        if x == str(out)
        else x
        for x in receipt["command"]
    ]
    receipt["binary_sha256"] = sha(native)
    receipt["source_sha256"] = sha(source) if str(source) != "synthetic" else None
    receipt["preflight"] = preflight
    receipt["device_samples"] = samples
    receipt["clock_policy"] = (
        "Existing dynamic device clocks; no clock, power or persistence settings changed"
    )
    receipt["exit_code"] = process.returncode
    receipt["experiment_wall_seconds"] = time.monotonic() - begin
    receipt["hardware_stop"] = hardware_stop
    receipt["log_sha256"] = sha(log)
    if hardware_stop:
        receipt["status"] = "FAIL"
    out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        json.dumps(
            {
                "trial": out.name,
                "status": receipt["status"],
                "rate": rate,
                "stop": receipt.get("stop_reason"),
                "classified_rate": receipt.get("fully_classified_emitted_points_per_second"),
            }
        ),
        flush=True,
    )
    return receipt


def campaign(a):
    a.out.mkdir(parents=True, exist_ok=False)
    model_hashes = {p.name: sha(p) for p in a.model.iterdir()}
    report = {
        "schema": "synrex.sustained_campaign.v1",
        "status": "FAIL",
        "source_commit": a.source_commit,
        "trials": [],
        "skipped_rates": [],
        "model_package_sha256_before": model_hashes,
    }
    try:
        modes = a.modes.split(",")
        rates = [int(x) for x in a.rates.split(",")]
        for mode in modes:
            best = None
            for index, rate in enumerate(rates):
                name = f"{mode}-{a.points}-{rate}-t1.json"
                result = run_trial(
                    a.native,
                    a.model,
                    a.source,
                    a.out / name,
                    rate=rate,
                    mode=mode,
                    seconds=a.seconds,
                    warmup=a.warmup,
                    points=a.points,
                    pinned=a.pinned,
                )
                report["trials"].append(result)
                (a.out / "progress.json").write_text(json.dumps(report, indent=2) + "\n")
                if result["status"] == "PASS":
                    best = rate
                else:
                    report["skipped_rates"].extend(
                        {
                            "mode": mode,
                            "rate": r,
                            "status": "NOT_RUN",
                            "reason": "lower rate violated bounded sustained gate",
                        }
                        for r in rates[index + 1 :]
                    )
                    break
            if best and a.repeat_best:
                for repeat in range(2, a.repeat_best + 1):
                    result = run_trial(
                        a.native,
                        a.model,
                        a.source,
                        a.out / f"{mode}-{a.points}-{best}-t{repeat}.json",
                        rate=best,
                        mode=mode,
                        seconds=a.seconds,
                        warmup=a.warmup,
                        points=a.points,
                        pinned=a.pinned,
                    )
                    report["trials"].append(result)
        report["model_package_unchanged"] = model_hashes == {
            p.name: sha(p) for p in a.model.iterdir()
        }
        report["status"] = "PASS" if report["model_package_unchanged"] else "FAIL"
        report["scope"] = (
            "PASS means bounded campaign executed; each native trial independently "
            "passes or fails its sustained gate"
        )
    finally:
        (a.out / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--modes", default="points-map,points")
    parser.add_argument("--rates", default="50000,100000,200000,350000,500000,750000,1000000")
    parser.add_argument("--points", type=int, default=4096)
    parser.add_argument("--seconds", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat-best", type=int, default=2)
    parser.add_argument("--pinned", action="store_true")
    args = parser.parse_args()
    campaign(args)
