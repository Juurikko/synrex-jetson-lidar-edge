# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
"""Guarded native, external TCP transport qualification. No Python inference."""

import argparse
import json
import subprocess
import time
from pathlib import Path

from qualify_streaming import available, gpu_sample, query, sha
from sustained_campaign import run_trial


def live_trial(
    a,
    out,
    source,
    rate,
    seconds,
    *,
    warmup=5,
    cells=65536,
    churn=0,
    frames=None,
    fault=None,
    stall=0,
    reconnect=0,
    model=None,
    export_bundle=False,
    validation_mode="decoded",
    cancel_after=0,
):
    out.mkdir(exist_ok=False)
    model = model or a.model
    before = {p.name: sha(p) for p in model.iterdir() if p.is_file()}
    preflight = available()
    common = [
        "--model",
        str(model),
        "--source",
        str(source),
        "--rate",
        str(rate),
        "--seconds",
        str(seconds),
        "--warmup",
        str(warmup),
        "--points",
        "16384",
        "--map-cells",
        str(cells),
        "--churn-mm",
        str(churn),
        "--validation-mode",
        validation_mode,
    ]
    if frames:
        common += ["--frames", str(frames)]
    processes, logs, commands, samples = {}, {}, {}, []
    hardware_stop = None
    controller_error = None
    begin = time.monotonic()

    def start(name, extra=()):
        command = [
            str(a.native_dir / "synrex-live"),
            name,
            *common,
            "--receipt",
            str(out / (name + ".json")),
            *extra,
        ]
        log = (out / (name + ".log")).open("w")
        logs[name] = log
        commands[name] = command
        processes[name] = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)

    def ready(name, marker):
        deadline = time.monotonic() + 15
        while marker not in (out / (name + ".log")).read_text():
            if processes[name].poll() is not None or time.monotonic() >= deadline:
                raise RuntimeError(name + " startup failed")
            time.sleep(0.05)

    try:
        start("receive", ["--geometry", "radix"])
        ready("receive", "live_ready=1")
        subscriber_options = ["--stall-ms", str(stall)]
        if export_bundle:
            subscriber_options += [
                "--export-bundle",
                str(out / "laz-bundle"),
                "--taxonomy",
                str(a.taxonomy),
                "--export-after-frame",
                str((warmup * rate + 16383) // 16384),
            ]
        start("subscribe", subscriber_options)
        ready("subscribe", "subscriber_ready=1")
        extra = ["--reconnect", str(reconnect)]
        if fault:
            extra += ["--fault", fault]
        start("send", extra)
        producer_started = time.monotonic()
        requested_cancel = False
        while any(p.poll() is None for p in processes.values()):
            if (
                cancel_after
                and not requested_cancel
                and time.monotonic() - producer_started >= cancel_after
            ):
                processes["receive"].terminate()
                requested_cancel = True
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
            sample["process_rss_bytes"] = {}
            for name, process in processes.items():
                path = Path(f"/proc/{process.pid}/statm")
                if path.exists():
                    values = path.read_text().split()
                    if len(values) > 1:
                        sample["process_rss_bytes"][name] = int(values[1]) * 4096
            sample["combined_rss_bytes"] = sum(sample["process_rss_bytes"].values())
            if sample["combined_rss_bytes"] > 1024**3:
                hardware_stop = "combined_host_memory_budget"
            sample["wall_seconds"] = time.monotonic() - begin
            samples.append(sample)
            pids = query(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"])
            owned = {p.pid for p in processes.values()}
            if any(int(pid.strip()) not in owned for pid in pids.splitlines() if pid.strip()):
                hardware_stop = "competing_compute_process"
            if sample["used_mib"] > 4096:
                hardware_stop = "device_memory_budget"
            if time.monotonic() - begin > seconds + warmup + 40:
                hardware_stop = "bounded_trial_timeout"
            if hardware_stop:
                break
            # A native correctness failure ends only this experiment, never another workload.
            if any(
                p.poll() not in (None, 0)
                for name, p in processes.items()
                if not (stall and name == "subscribe")
            ):
                time.sleep(0.3)
                break
            time.sleep(0.25)
    except (RuntimeError, OSError, ValueError) as error:
        controller_error = f"{type(error).__name__}:{error}"
    finally:
        for p in processes.values():
            if p.poll() is None:
                p.terminate()
        for p in processes.values():
            try:
                p.wait(timeout=12)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
        for log in logs.values():
            log.close()
    report = dict(
        schema="synrex.streaming_r1.transport_trial.v1",
        status="FAIL",
        source_commit=a.source_commit,
        rate=rate,
        seconds=seconds,
        warmup_seconds=warmup,
        cells=cells,
        churn_mm=churn,
        resource_preflight=preflight,
        device_samples=samples,
        hardware_stop=hardware_stop,
        controller_error=controller_error,
        requested_cancel_seconds=cancel_after,
        native={},
        source_sha256=sha(source),
        binary_sha256=sha(a.native_dir / "synrex-live"),
        model_before=before,
        model_after={p.name: sha(p) for p in model.iterdir() if p.is_file()},
    )
    for name, process in processes.items():
        path = out / (name + ".json")
        row = (
            json.loads(path.read_text())
            if path.exists()
            else {"status": "FAIL", "error": "no native receipt"}
        )
        row["exit_code"] = process.returncode
        row["log_sha256"] = sha(out / (name + ".log"))
        row["command"] = [
            "<native>"
            if x == str(a.native_dir / "synrex-live")
            else "<model>"
            if x == str(model)
            else "<source>"
            if x == str(source)
            else "<receipt>"
            if x == str(path)
            else "<" + commands[name][i - 1][2:] + ">"
            if i and commands[name][i - 1] in {"--taxonomy", "--export-bundle"}
            else x
            for i, x in enumerate(commands[name])
        ]
        report["native"][name] = row
    passed = (
        not controller_error
        and not hardware_stop
        and before == report["model_after"]
        and len(processes) == 3
    )
    passed = passed and all(
        x["status"] == "PASS" and x["exit_code"] == 0 for x in report["native"].values()
    )
    report["status"] = "PASS" if passed else "FAIL"
    report["sustained_claim"] = passed and seconds >= 60 and not frames and not fault and not stall
    (out / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({"trial": out.name, "status": report["status"], "hardware_stop": hardware_stop}),
        flush=True,
    )
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--native-dir", type=Path, required=True)
    p.add_argument("--baseline-dir", type=Path)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--synthetic-model", type=Path)
    p.add_argument("--sources", type=Path, nargs=3, required=True)
    p.add_argument("--source-commit", required=True)
    p.add_argument(
        "--phase",
        choices=[
            "smoke",
            "transport",
            "geometry",
            "long",
            "faults",
            "exports",
            "consumer-ab",
            "shutdown",
        ],
        required=True,
    )
    p.add_argument(
        "--taxonomy",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "examples/unet/semantic_classes.json",
    )
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(exist_ok=False)
    results = []

    def live(label, source, rate, seconds, **kw):
        r = live_trial(a, a.out / label, source, rate, seconds, **kw)
        results.append(r)
        (a.out / "progress.json").write_text(json.dumps(results, indent=2) + "\n")
        return r

    if a.phase == "smoke":
        live("unet-smoke", a.sources[0], 1000000, 1, frames=32, warmup=0, reconnect=8)
        if a.synthetic_model:
            live(
                "conv3d-smoke", a.sources[0], 100000, 1, frames=8, warmup=0, model=a.synthetic_model
            )
    elif a.phase == "transport":
        for source_index, cells, churn in [
            (0, 65536, 0),
            (2, 65536, 0),
            (0, 4096, 64000),
            (0, 65536, 64000),
        ]:
            for rate in (1000000, 1500000, 2000000):
                r = live(
                    f"scene{source_index}-cells{cells}-churn{churn}-rate{rate}-t1",
                    a.sources[source_index],
                    rate,
                    60,
                    cells=cells,
                    churn=churn,
                )
                if r["status"] != "PASS":
                    break
                r = live(
                    f"scene{source_index}-cells{cells}-churn{churn}-rate{rate}-t2",
                    a.sources[source_index],
                    rate,
                    60,
                    cells=cells,
                    churn=churn,
                )
                if r["status"] != "PASS":
                    break
    elif a.phase == "shutdown":
        live("requested-cancellation", a.sources[0], 1000000, 30, cancel_after=7, churn=64000)
    elif a.phase == "consumer-ab":
        for mode in ("reference", "decoded"):
            live(f"consumer-{mode}-1m", a.sources[0], 1000000, 60, validation_mode=mode)
    elif a.phase == "exports":
        live("async-export-1m", a.sources[0], 1000000, 60, export_bundle=True)
    elif a.phase == "long":
        live("full-churn-10min", a.sources[0], 1000000, 600, churn=64000)
    elif a.phase == "faults":
        for fault in ("duplicate", "gap", "clock", "truncated"):
            live(fault, a.sources[0], 100000, 1, frames=8, warmup=0, fault=fault)
        live("subscriber-stall", a.sources[0], 1000000, 3, warmup=0, stall=2000)
        live("reconnect", a.sources[0], 1000000, 2, warmup=0, reconnect=16)
    elif a.phase == "geometry":
        if not a.baseline_dir:
            raise ValueError("matched geometry requires frozen baseline")
        for scene, rate, churn in [
            (0, 2000000, 64000),
            (0, 3000000, 0),
            (1, 3000000, 0),
            (2, 2500000, 0),
        ]:
            for name, binary, strategy in [
                ("frozen", a.baseline_dir / "synrex-sustain", "direct"),
                ("direct", a.native_dir / "synrex-sustain", "direct"),
                ("radix1", a.native_dir / "synrex-sustain", "radix"),
                ("radix2", a.native_dir / "synrex-sustain", "radix"),
            ]:
                r = run_trial(
                    binary,
                    a.model,
                    a.sources[scene],
                    a.out / f"scene{scene}-{rate}-{churn}-{name}.json",
                    rate=rate,
                    mode="points-map",
                    seconds=60,
                    points=16384,
                    geometry=strategy,
                    churn=churn,
                    map_cells=65536,
                    verify_map=True,
                    timeline=True,
                )
                results.append(r)
                (a.out / "progress.json").write_text(json.dumps(results, indent=2) + "\n")
                if r["status"] != "PASS":
                    break
    (a.out / "receipt.json").write_text(
        json.dumps(
            {
                "schema": "synrex.streaming_r1.campaign.v1",
                "trials": results,
                "scope": "Controller completion does not override individual native FAILs",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
