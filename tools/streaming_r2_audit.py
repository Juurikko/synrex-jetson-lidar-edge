# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
"""Read-only source/preservation/privacy audit; writes one new factual receipt."""

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

from release_checks import FORBIDDEN, findings

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    a = parser.parse_args()
    freeze = json.loads((ROOT / "evidence/streaming_r2_baseline_freeze.json").read_text())
    assert all(sha(ROOT / p) == h for p, h in freeze["protected_files"].items())
    for ref, value in freeze["refs"].items():
        assert (
            subprocess.check_output(["git", "rev-parse", ref], cwd=ROOT, text=True).strip() == value
        )
    names = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=ROOT, text=True
    ).splitlines()
    sensitive = re.compile(
        rb"/home/[a-zA-Z0-9._-]+/|100\.(?:6[4-9]|[7-9][0-9]|1[01][0-9]|12[0-7])\.\d+\.\d+"
        rb"|192\.168\.\d+\.\d+|DESKTOP-[A-Z0-9]{5,}|rtx[0-9]+-wsl"
    )
    legacy = []
    for name in names:
        path = ROOT / name
        assert path.suffix.lower() not in FORBIDDEN and path.stat().st_size < 20 * 1024 * 1024, name
        data = path.read_bytes()
        assert not findings(data), name
        if b"\0" not in data and sensitive.search(data):
            # This device display label was already shipped with R4; preserve exact evidence.
            assert name == "evidence/rtx4090_full_suite.json", name
            original = subprocess.check_output(
                ["git", "show", "byom-unet-fractal-r4-rc2:" + name], cwd=ROOT
            )
            assert original == data
            legacy.append(
                dict(
                    path=name,
                    sha256=sha(path),
                    treatment="Already-public non-connectable device display label; unchanged",
                )
            )
    with a.receipt.open("x") as f:
        f.write(
            json.dumps(
                dict(
                    schema="synrex.streaming_r2.source_audit.v1",
                    status="PASS",
                    files_scanned=len(names),
                    protected_files_unchanged=len(freeze["protected_files"]),
                    accepted_refs_unchanged=True,
                    secrets=0,
                    new_private_paths_or_endpoints=0,
                    forbidden_model_data_binaries=0,
                    legacy_public_device_labels=legacy,
                    scope="Current source; historical object audit recorded separately",
                ),
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
