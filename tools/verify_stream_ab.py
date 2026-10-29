# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
"""Exact trace/event/map oracle over completed native RTX qualification artifacts."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(a):
    a.out.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "synrex.streaming_exact_ab.v1",
        "status": "FAIL",
        "scope": (
            "Artifact comparison/map-only replay. Original events and tensors were produced "
            "by actual native neural inference; stress map updates are model-free fixtures."
        ),
        "trace_comparisons": [],
        "event_comparisons": [],
        "map_comparisons": [],
        "reference_map_binary_sha256": sha(a.reference_map),
        "optimized_map_binary_sha256": sha(a.optimized_map),
    }
    try:
        for model in ("synthetic", "unet"):
            original = a.default / model
            pinned = a.pinned / model
            events = sorted(original.glob("*-event.json"), key=lambda p: int(p.stem.split("-")[0]))
            if not events:
                raise RuntimeError("native neural events missing")
            for event in events:
                number = event.name.split("-")[0]
                x, y = json.loads(event.read_text()), json.loads((pinned / event.name).read_text())
                x.pop("timings")
                y.pop("timings")
                if json.dumps(x, sort_keys=True) != json.dumps(y, sort_keys=True):
                    raise RuntimeError("point/class/frame metadata mismatch")
                report["event_comparisons"].append(
                    {
                        "model": model,
                        "frame": int(number),
                        "points": x["frame_points"],
                        "status": "PASS",
                        "canonical_sha256": hashlib.sha256(
                            json.dumps(x, sort_keys=True, separators=(",", ":")).encode()
                        ).hexdigest(),
                    }
                )
                for file in sorted((original / f"{number}-stream").glob("*.bin")):
                    expected = sha(file)
                    others = [
                        original / f"{number}-offline" / file.name,
                        pinned / f"{number}-stream" / file.name,
                        pinned / f"{number}-offline" / file.name,
                    ]
                    if not all(sha(p) == expected for p in others):
                        raise RuntimeError("tensor/logit bytes differ")
                    report["trace_comparisons"].append(
                        {
                            "model": model,
                            "frame": int(number),
                            "file": file.name,
                            "bytes": file.stat().st_size,
                            "sha256": expected,
                            "status": "PASS",
                        }
                    )
            # Separate maps for distinct source events; never merge unknown coordinate frames.
            cases = [events[:4]] + [[e] for e in events[4:6]]
            for case_index, case in enumerate(cases):
                for capacity in (64, 65536):
                    paths = []
                    for name, binary in (
                        ("reference", a.reference_map),
                        ("optimized", a.optimized_map),
                    ):
                        output = a.out / f"{model}-{case_index}-{capacity}-{name}.json"
                        subprocess.run(
                            [str(binary), str(output), str(capacity), *map(str, case)],
                            check=True,
                            capture_output=True,
                            text=True,
                            timeout=180,
                        )
                        paths.append(output)
                    assert sha(paths[0]) == sha(paths[1]), (
                        "full snapshots/deltas/point events differ"
                    )
                    report["map_comparisons"].append(
                        {
                            "model": model,
                            "case": case_index,
                            "source_events": [e.name for e in case],
                            "capacity": capacity,
                            "full_output_sha256": sha(paths[0]),
                            "output_bytes": paths[0].stat().st_size,
                            "status": "PASS",
                            "checks": [
                                "point fields/semantic IDs",
                                "snapshots",
                                "deltas/eviction",
                                "repeated observations",
                                "class updates",
                                "late observations",
                                "aging",
                                "unknown confidence",
                                "coordinate mismatch rejection without mutation",
                            ],
                        }
                    )
        report["status"] = "PASS"
    finally:
        (a.out / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report["status"])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--default", type=Path, required=True)
    p.add_argument("--pinned", type=Path, required=True)
    p.add_argument("--reference-map", type=Path, required=True)
    p.add_argument("--optimized-map", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    verify(p.parse_args())
