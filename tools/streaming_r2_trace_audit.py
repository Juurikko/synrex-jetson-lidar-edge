# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
"""Independently compare actual native offline/stream tensor traces byte for byte."""

import argparse
import filecmp
import hashlib
import json
from pathlib import Path


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--receipt", type=Path, required=True)
    a = p.parse_args()
    rows = []
    for model in ["synthetic", "unet"]:
        root = a.root / model
        receipt = json.loads((root / "receipt.json").read_text())
        assert receipt["status"] == "PASS"
        windows = sorted(root.glob("*-stream"))
        assert windows
        for stream in windows:
            offline = stream.with_name(stream.name.replace("-stream", "-offline"))
            names = {f.name for f in stream.glob("*.bin")}
            assert names and names == {f.name for f in offline.glob("*.bin")}
            assert (
                json.loads((stream / "manifest.json").read_text())["tensor_contract"]
                == json.loads((offline / "manifest.json").read_text())["tensor_contract"]
            )
            for name in sorted(names):
                assert filecmp.cmp(stream / name, offline / name, shallow=False), (
                    stream.name,
                    name,
                )
            rows.append(
                dict(
                    model=model,
                    window=stream.name,
                    status="PASS",
                    exact_tensor_and_logit_bytes=True,
                    traces={name: sha(stream / name) for name in sorted(names)},
                )
            )
    with a.receipt.open("x") as f:
        f.write(
            json.dumps(
                dict(
                    schema="synrex.streaming_r2.trace_audit.v1",
                    status="PASS",
                    windows=len(rows),
                    comparisons=rows,
                    scope="Existing actual native neural traces; no inferred GPU result",
                ),
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
