# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
"""Compare completed reference/direct/flat neural traces and classified metadata exactly."""

import argparse
import filecmp
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", type=Path, required=True)
parser.add_argument("--receipt", type=Path, required=True)
args = parser.parse_args()
if args.receipt.exists():
    parser.error("receipt must be new")
root = args.root
report = {"schema": "synrex.geometry_gpu_exact.v1", "status": "FAIL", "comparisons": []}
try:
    for mode in ("direct", "flat"):
        for model in ("synthetic", "unet"):
            base = root / "geometry-v2-equivalence-reference" / model
            new = root / ("geometry-v2-equivalence-" + mode) / model
            for directory in (base, new):
                receipt = json.loads((directory / "receipt.json").read_text())
                if receipt["status"] != "PASS":
                    raise RuntimeError("neural qualification receipt failed")
            events = sorted(base.glob("*-event.json"))
            if not events or {p.name for p in events} != {p.name for p in new.glob("*-event.json")}:
                raise RuntimeError("missing or extra classified events")
            for event in events:
                x = json.loads(event.read_text())
                y = json.loads((new / event.name).read_text())
                x.pop("timings")
                y.pop("timings")
                if json.dumps(x, sort_keys=True) != json.dumps(y, sort_keys=True):
                    raise RuntimeError("event/class/source metadata differs")
            files = 0
            total = 0
            traces = list(base.glob("*-stream/*.bin"))
            if not traces or {p.relative_to(base) for p in traces} != {
                p.relative_to(new) for p in new.glob("*-stream/*.bin")
            }:
                raise RuntimeError("missing or extra tensor/logit traces")
            for original in traces:
                other = new / original.relative_to(base)
                if not filecmp.cmp(original, other, shallow=False):
                    raise RuntimeError("tensor/logit bytes differ")
                total += original.stat().st_size
                files += 1
            report["comparisons"].append(
                {
                    "mode": mode,
                    "model": model,
                    "status": "PASS",
                    "events": len(events),
                    "trace_files": files,
                    "trace_bytes": total,
                    "reference_receipt_sha256": hashlib.sha256(
                        (base / "receipt.json").read_bytes()
                    ).hexdigest(),
                    "variant_receipt_sha256": hashlib.sha256(
                        (new / "receipt.json").read_bytes()
                    ).hexdigest(),
                }
            )
    report["status"] = "PASS"
finally:
    args.receipt.write_text(json.dumps(report, indent=2) + "\n")
print(report["status"])
