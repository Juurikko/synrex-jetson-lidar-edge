# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
"""Adversarial CPU verification of genuine recorded neural events, without CUDA.

Set SYNREX_R2_ASSETS to the extracted showcase. Rebind output hashes after tampering
so tests exercise independent identity/map/freshness checks, not just checksums.
"""

import copy
import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from streaming_r2_replay import verify  # noqa: E402


@pytest.mark.parametrize("fault", ["semantic", "duplicate", "time", "victim", "missing", "reuse"])
def test_hash_rebound_false_visual_provenance_rejected(tmp_path, fault):
    native_dir = Path(os.environ.get("SYNREX_TEST_NATIVE_DIR", ROOT / "build")).resolve()
    root = Path(os.environ.get("SYNREX_R2_ASSETS", native_dir / "absent-r2-assets"))
    recording = root / "recording"
    if not recording.is_dir():
        pytest.skip("NOT_RUN: genuine R2 release recording required; set SYNREX_R2_ASSETS")
    source = root / "qgis/data/source/TEST-0944_6564-005900112.laz"
    model = root / "models/unet.srxmodel"
    with (recording / "events.jsonl").open() as f:
        frame = json.loads(next(f))
        delta = json.loads(next(f))
    receipt = json.loads((recording / "native-receipt.json").read_text())
    assert frame["schema"] == "synrex.classified_frame.v1"
    assert delta["schema"] == "synrex.semantic_delta.v1"
    frame, delta = copy.deepcopy(frame), copy.deepcopy(delta)
    if fault == "semantic":
        changed = (frame["points"][0]["semantic_id"] + 1) % 7
        frame["points"][0]["semantic_id"] = changed
        frame["points"][0]["model_las_code"] = [1, 2, 200, 6, 9, 17, 201][changed]
    elif fault == "duplicate":
        frame["points"][1]["id"] = frame["points"][0]["id"]
    elif fault == "time":
        frame["points"][0]["timestamp_ns"] += 1
    elif fault == "victim":
        delta["evicted"] = [0, 0, 0]
    elif fault == "reuse":
        receipt["inferred_classes_reused_points"] = 1
    events = json.dumps(frame) + "\n"
    if fault != "missing":
        events += json.dumps(delta) + "\n"
    (tmp_path / "events.jsonl").write_text(events)
    receipt["events_sha256"] = hashlib.sha256(events.encode()).hexdigest()
    (tmp_path / "native-receipt.json").write_text(json.dumps(receipt))
    for name in ["input-provenance.json", "input.jsonl"]:
        os.link(recording / name, tmp_path / name)
    with pytest.raises(AssertionError):
        verify(source, tmp_path, model)
    assert not (tmp_path / "verification.json").exists()
