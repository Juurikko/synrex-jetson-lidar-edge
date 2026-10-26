# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
# ruff: noqa: E402
"""Qualify an already moved installation against both frozen dense model outputs."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from verify_laz import verify
from verify_semantics import verify_semantics


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["prefix", "native-build", "qualified", "data", "synthetic", "out"]:
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--resource-allocation-verified", action="store_true")
    a = p.parse_args()
    if not a.resource_allocation_verified:
        raise ValueError("verified GPU allocation required")
    a.out.mkdir(parents=True, exist_ok=False)
    native = (a.prefix / "bin/synrex-native").resolve()
    before = sha(native)
    env = dict(os.environ, PATH="/nonexistent", PYTHONPATH="", LD_DEBUG="libs")

    def run(args):
        c = subprocess.run(
            [str(native), *map(str, args)], env=env, capture_output=True, text=True, check=True
        )
        assert "libpython" not in c.stderr and "libtorch" not in c.stderr
        return json.loads(c.stdout)

    q = json.loads((a.qualified / "qualification.json").read_text())
    source = a.data / "test" / q["files"][0]["source"]
    output, sidecar = a.out / "unet.laz", a.out / "unet-semantics.json"
    unet = run(
        [
            "classify",
            "--model",
            a.qualified / "unet.srxmodel",
            "--input",
            source,
            "--out",
            output,
            "--semantics",
            sidecar,
        ]
    )
    assert sha(output) == sha(a.qualified / (source.stem + "-classified.laz"))
    validation = verify_semantics(source, output, sidecar, a.qualified / "unet.srxmodel")
    synthetic = run(
        [
            "classify",
            "--model",
            a.synthetic / "engine.srxmodel",
            "--input",
            a.synthetic / "scene/dense-synthetic.laz",
            "--out",
            a.out / "conv3d.laz",
        ]
    )
    synthetic_validation = verify(
        a.synthetic / "scene/dense-synthetic.laz", a.out / "conv3d.laz", a.synthetic / "expected.u8"
    )
    dynamic = subprocess.check_output(["readelf", "-d", str(native)], text=True)
    assert "RPATH" not in dynamic and "RUNPATH" not in dynamic
    examples = a.prefix / "share/synrex/examples/unet"
    for name in ["MODEL_NOTICE.txt", "fractal-classification.qml"]:
        assert sha(examples / name) == sha(ROOT / "examples/unet" / name)
    assert sha(native) == before
    result = dict(
        status="PASS",
        source_commit=subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        installed_binary_sha256=before,
        build_binary_sha256=sha(a.native_build),
        no_rpath_runpath=True,
        native_no_python="PASS: loader inspection and PATH=/nonexistent",
        same_installed_binary_both_models=True,
        notices_and_qgis_style_installed=True,
        unet_native=unet,
        unet_independent=validation,
        conv3d_native=synthetic,
        conv3d_independent=synthetic_validation,
    )
    (a.out / "qualification.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
