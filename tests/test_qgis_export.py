# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
"""Failure boundaries for the optional viewing exporter; no mocked inference."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import laspy
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("export_qgis", ROOT / "tools/export_qgis.py")
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


def fixture(tmp_path):
    header = laspy.LasHeader(version="1.4", point_format=8)
    cloud = laspy.LasData(header)
    cloud.X = np.array([30, 10, 20], dtype=np.int32)
    cloud.classification = [1, 2, 200]
    classified = tmp_path / "classified.laz"
    cloud.write(classified)
    sidecar = tmp_path / "semantics.json"
    sidecar.write_text(json.dumps(dict(las_classes=[1, 2, 200], semantic_ids=[0, 1, 2])))
    return classified, sidecar


@pytest.mark.parametrize("ids", [[0, 1], [0, 1, 1], [0, 1, 0.5], [0, 1, 256]])
def test_stage_rejects_bad_semantics_even_under_optimization(tmp_path, ids):
    classified, sidecar = fixture(tmp_path)
    info = json.loads(sidecar.read_text())
    info["semantic_ids"] = ids
    sidecar.write_text(json.dumps(info))
    result = subprocess.run(
        [
            sys.executable,
            "-O",
            str(ROOT / "tools/qgis_copc_view.py"),
            "--classified",
            str(classified),
            "--sidecar",
            str(sidecar),
            "--staging",
            str(tmp_path / "stage.las"),
            "--receipt",
            str(tmp_path / "receipt.json"),
        ],
        capture_output=True,
    )
    assert result.returncode != 0
    assert not (tmp_path / "receipt.json").exists()
    assert not (tmp_path / "stage.las").exists()


def test_export_refuses_existing_output(tmp_path):
    with pytest.raises(FileExistsError):
        exporter.export([], tmp_path, "unused", "unused")


def test_export_rejects_mixed_model_identity_before_external_tools(tmp_path):
    packages = []
    for i in range(2):
        directory = tmp_path / str(i)
        (directory / "model/unet.srxmodel").mkdir(parents=True)
        (directory / "model/unet.srxmodel/manifest.json").write_text(json.dumps(dict(model=i)))
        (directory / "selection.json").write_text(json.dumps(dict(files=[])))
        packages.append(directory)
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="mixed model"):
        exporter.export(packages, out, "unused", "unused")
    assert not out.exists()


def test_windows_external_rejects_batch_expansion(tmp_path):
    with pytest.raises(ValueError, match="unsupported shell"):
        exporter.external("tool.cmd", ["%SECRET%"], True, tmp_path)
    assert not (tmp_path / "external-tool.cmd").exists()
