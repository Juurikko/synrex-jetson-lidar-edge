# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
"""Large original reference fixtures and honest CPU-only preparation receipts."""

import json
import subprocess
import sys

import laspy
import numpy as np
from test_byom import NATIVE, ROOT, SPEC, package, run

from byom_reference import create_dense_scene, expected_labels


def test_dense_50013_source_indices_and_nonconstant_oracle(tmp_path):
    pkg, _, model = package(tmp_path)
    source = create_dense_scene(tmp_path / "scene", 50013)
    labels, oracle = expected_labels(source, SPEC, model)
    assert len(labels) == oracle["points"] == 50013
    assert set(labels) == {2, 5, 6}
    assert sum(oracle["class_histogram"].values()) == 50013
    data = laspy.read(source, laz_backend=laspy.LazBackend.Lazrs)
    assert data.X[1] == data.X[2] and data.gps_time[1] != data.gps_time[2]
    plan = tmp_path / "plan.json"
    run("plan", "--model", pkg, "--input", source, "--out", plan)
    actual = json.loads(plan.read_text())
    assert len(actual["blocks"]) == oracle["blocks"]
    assert actual["memberships"] == oracle["memberships"]
    ids = [p for block in actual["blocks"] for p, _ in block["members"]]
    assert set(ids) == set(range(50013))
    assert np.all(np.isin(labels, SPEC["las_classes"]))


def test_cpu_preparation_never_claims_gpu_qualification(tmp_path):
    out = tmp_path / "prepare"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/qualify_dense_gpu.py"),
            "--native",
            str(NATIVE),
            "--out",
            str(out),
            "--prepare-only",
        ],
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads((out / "receipt.json").read_text())
    assert report["SYNTHETIC_ONNX_EXPORT"] == "PASS"
    assert report["REAL_GPU_INFERENCE"] == "NOT_RUN"
    assert report["TENSORRT_ENGINE_BUILD"] == "NOT_RUN"
    assert report["BYOM_CLASSIFIED_LAZ_E2E"] == "NOT_RUN"
    assert report["INDEPENDENT_LAZ_VALIDATION"] == "NOT_RUN"
    assert not list(out.glob("classified-*.laz"))
