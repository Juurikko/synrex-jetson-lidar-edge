# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
"""CPU-only wire, preservation/export transaction and partial product tests."""

import json
import os
import subprocess
from pathlib import Path

import pytest

from synthetic_laz import create

ROOT = Path(__file__).resolve().parents[1]
NATIVE = Path(os.environ.get("SYNREX_TEST_NATIVE_DIR", ROOT / "build")).resolve()


def test_native_wire():
    r = subprocess.run([str(NATIVE / "synrex-wire-tests")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["status"] == "PASS"


def test_atomic_export(tmp_path):
    source, _ = create(tmp_path / "source", 257)
    # Explicitly model-free package fixture; never used to claim neural inference.
    artifact = tmp_path / "fixture.onnx"
    artifact.write_bytes(b"MODEL_FREE_EXPORT_FIXTURE_ONLY")
    model = tmp_path / "fixture.srxmodel"
    subprocess.run(
        [
            str(NATIVE / "synrex-native"),
            "pack",
            "--spec",
            str(ROOT / "examples/unet/dense_voxel_asprs_safe.json"),
            "--onnx",
            str(artifact),
            "--license",
            str(ROOT / "LICENSES/MIT.txt"),
            "--out",
            str(model),
        ],
        check=True,
    )
    work = tmp_path / "products"
    r = subprocess.run(
        [
            str(NATIVE / "synrex-export-tests"),
            str(source),
            str(model),
            str(ROOT / "examples/unet/semantic_classes.json"),
            str(work),
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["status"] == "PASS"
    # Reuse independent R4 record/extra-byte/VLR/class/semantic preservation checks.
    import sys

    sys.path.insert(0, str(ROOT / "tools"))
    from streaming_export import validate

    bundle = work / "complete"
    proof = validate(bundle, source, model, ROOT / "examples/unet/semantic_classes.json")
    assert proof["points"] == 257
    manifest_path = bundle / "bundle.json"
    original_manifest = manifest_path.read_bytes()
    altered = json.loads(original_manifest)
    altered["bindings"]["source_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="source/model"):
        validate(bundle, source, model, ROOT / "examples/unet/semantic_classes.json")
    manifest_path.write_bytes(original_manifest)
    saved = (bundle / "semantics.json").read_bytes()
    (bundle / "semantics.json").write_bytes(saved + b" ")
    with pytest.raises(ValueError, match="digest"):
        validate(bundle, source, model, ROOT / "examples/unet/semantic_classes.json")
    (bundle / "semantics.json").write_bytes(saved)
    (bundle / "bundle.json").unlink()
    with pytest.raises(FileNotFoundError):
        validate(bundle, source, model, ROOT / "examples/unet/semantic_classes.json")


def test_summary_does_not_promote_a_failed_repeat_or_mix_durations(tmp_path):
    """Mutate recorded receipt copies only; this test executes no neural inference."""
    import copy
    import sys

    sys.path.insert(0, str(ROOT / "tools"))
    from summarize_streaming_r1 import summarize

    passing = json.loads((ROOT / "evidence/streaming_r1_transport_sustained.json").read_text())[
        "trials"
    ][0]
    failed = copy.deepcopy(passing)
    failed["status"] = "FAIL"
    failed["sustained_claim"] = False
    path = tmp_path / "mutated-receipts.json"
    path.write_text(json.dumps({"trials": [passing, copy.deepcopy(passing), failed]}))
    summary = summarize([path], tmp_path / "rejected")
    assert summary["failed_trials"] == 1
    assert not summary["repeated_sustained_configurations"]
    long = copy.deepcopy(passing)
    long["seconds"] = 600
    path.write_text(json.dumps({"trials": [passing, long]}))
    assert not summarize([path], tmp_path / "separate")["repeated_sustained_configurations"]
