# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
"""Execute local qualification and emit receipts. Physical GPU tests require explicit opt-in."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def qualify(build: Path, receipt: Path, gpu_allocation: str | None) -> int:
    build = build.resolve()
    receipt.parent.mkdir(parents=True, exist_ok=True)
    logs = build / "qualification"
    logs.mkdir(exist_ok=True)
    records = []
    env = os.environ.copy()
    env["SYNREX_TEST_NATIVE"] = str(build / "synrex-native")
    env["SYNREX_TEST_WRITER"] = str(build / "synrex-laz-labels")
    env.pop("SYNREX_RUN_GPU", None)
    if gpu_allocation:
        env["SYNREX_RUN_GPU"] = "1"

    def run(name, command):
        completed = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True)
        raw = completed.stdout + completed.stderr
        # Public receipts omit workstation-specific absolute paths.
        output = raw.replace(str(ROOT), "<workspace>")
        log = logs / f"{name}.log"
        log.write_text(output)
        records.append(
            {
                "check": name,
                "status": "PASS" if completed.returncode == 0 else "FAIL",
                "exit_code": completed.returncode,
                "command": [str(p).replace(str(ROOT), "<workspace>") for p in command],
                "log_sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
                "output_tail": output[-3500:],
            }
        )
        return completed

    run("native_build", ["cmake", "--build", str(build), "-j", "4"])
    run("ctest", ["ctest", "--test-dir", str(build), "--output-on-failure"])
    version = run("version", [str(build / "synrex-native"), "version"])
    backend = json.loads(version.stdout)["cuda_tensorrt"] if version.returncode == 0 else False
    junit = logs / "pytest.xml"
    run(
        "integration",
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            f"--basetemp={build / 'qualification-pytest'}",
            f"--junitxml={junit}",
        ],
    )
    cases = list(ET.parse(junit).iter("testcase")) if junit.exists() else []
    counts = {
        "passed": sum(
            not any(c.find(t) is not None for t in ("failure", "error", "skipped")) for c in cases
        ),
        "skipped": sum(c.find("skipped") is not None for c in cases),
        "failed": sum(c.find("failure") is not None or c.find("error") is not None for c in cases),
    }
    gpu_cases = [c for c in cases if c.get("name", "").startswith("test_real_cuda_tensorrt_")]
    gpu_passed = bool(gpu_cases) and all(
        not any(c.find(t) is not None for t in ("failure", "error", "skipped")) for c in gpu_cases
    )
    success = all(r["status"] == "PASS" for r in records)
    gpu_status = (
        "PASS"
        if gpu_allocation and backend and gpu_passed
        else "FAIL"
        if gpu_allocation
        else "NOT_RUN"
    )
    report = {
        "schema": "synrex.byom.qualification.v1",
        "utc": dt.datetime.now(dt.UTC).isoformat(),
        "host": {
            "name": platform.node(),
            "architecture": platform.machine(),
            "kernel": platform.release(),
        },
        "gpu_backend_compiled": backend,
        "gpu_allocation_record": gpu_allocation,
        "gpu_policy": "No remote AGX access or workload/stack changes by this runner",
        "records": records,
        "pytest": counts,
        "NATIVE_BUILD_STATUS": records[0]["status"],
        "CPU_TEST_STATUS": "PASS" if success else "FAIL",
        "GPU_TEST_STATUS": gpu_status,
        "BYOM_E2E_STATUS": gpu_status,
        "PUBLIC_RELEASE_READINESS": "PASS" if success and gpu_status == "PASS" else "FAIL",
        "scope": "Runtime qualification only; license/source assessment and "
        "owner publication approval are separate gates",
    }
    receipt.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k.isupper() or k == "pytest"}, indent=2))
    return 0 if success and (not gpu_allocation or gpu_status == "PASS") else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=Path("build"))
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument(
        "--verified-gpu-allocation",
        help="Operator-confirmed nonconflicting allocation identifier; enables real GPU tests",
    )
    args = parser.parse_args()
    raise SystemExit(qualify(args.build, args.receipt, args.verified_gpu_allocation))
