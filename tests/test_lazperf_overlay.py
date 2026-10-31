# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
"""Exact source admission and atomic build-owned codec generation."""

import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("overlay", ROOT / "tools/prepare_lazperf_overlay.py")
overlay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(overlay)
SOURCE = os.environ.get("SYNREX_TEST_LAZPERF_SOURCE_DIR")


@pytest.fixture
def inputs(tmp_path):
    if not SOURCE:
        pytest.skip("set SYNREX_TEST_LAZPERF_SOURCE_DIR to the exact pinned upstream source")
    source = tmp_path / "upstream"
    manifest = json.loads(overlay.DEFAULT_MANIFEST.read_bytes())
    for name in manifest["files"]:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(SOURCE) / name, path)
    return source, tmp_path / "build/lazperf-overlay"


def test_overlay_exact_and_idempotent_without_upstream_mutation(inputs):
    source, output = inputs
    manifest = json.loads(overlay.DEFAULT_MANIFEST.read_bytes())
    before = overlay.checked_files(source, manifest, "before")
    assert overlay.prepare(source, output) == output
    generated = overlay.checked_files(output, manifest, "after")
    assert len([name for name in before if before[name] != generated[name]]) == 5
    assert (output / "COPYING").read_bytes() == (source / "COPYING").read_bytes()
    times = {name: (output / name).stat().st_mtime_ns for name in generated}
    overlay.prepare(source, output)
    assert times == {name: (output / name).stat().st_mtime_ns for name in generated}
    assert overlay.checked_files(source, manifest, "before") == before
    assert not list(output.parent.glob("*.stage-*"))


@pytest.mark.parametrize("change", ["changed", "extra", "missing", "symlink"])
def test_changed_upstream_rejected_before_publication(inputs, change):
    source, output = inputs
    path = source / "cpp/lazperf/utils.hpp"
    if change == "changed":
        path.write_bytes(path.read_bytes() + b"\n")
    elif change == "extra":
        (source / "cpp/lazperf/extra.cpp").write_text("// unexpected\n")
    elif change == "missing":
        path.unlink()
    else:
        saved = path.with_suffix(".saved")
        path.rename(saved)
        path.symlink_to(saved.name)
    with pytest.raises(ValueError):
        overlay.prepare(source, output)
    assert not output.exists()


def test_existing_overlay_and_upstream_are_revalidated(inputs):
    source, output = inputs
    overlay.prepare(source, output)
    generated = output / "cpp/lazperf/streams.hpp"
    saved = generated.read_bytes()
    generated.write_bytes(saved + b"\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        overlay.prepare(source, output)
    generated.write_bytes(saved)
    external = source / "cpp/lazperf/streams.hpp"
    external.write_bytes(external.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        overlay.prepare(source, output)


def test_failed_patch_leaves_no_partial_overlay(inputs, tmp_path):
    source, output = inputs
    manifest = json.loads(overlay.DEFAULT_MANIFEST.read_bytes())
    patch = (overlay.DEFAULT_MANIFEST.parent / manifest["patch"]).read_bytes()
    # A self-consistent manifest still cannot permit a fuzzy or offset application.
    patch = patch.replace(b"@@ -104,9 +105,9 @@", b"@@ -103,9 +105,9 @@", 1)
    manifest["patch_sha256"] = overlay.digest(patch)
    (tmp_path / manifest["patch"]).write_bytes(patch)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        overlay.prepare(source, output, manifest_path)
    assert not output.exists()
    assert not list(output.parent.glob("*.stage-*"))


def test_patch_hash_and_output_overlap_rejected(inputs, tmp_path):
    source, output = inputs
    manifest = json.loads(overlay.DEFAULT_MANIFEST.read_bytes())
    (tmp_path / manifest["patch"]).write_text("bad patch\n")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="patch hash"):
        overlay.prepare(source, output, manifest_path)
    with pytest.raises(ValueError, match="disjoint"):
        overlay.prepare(source, source / "build")


def test_concurrent_prepare_publishes_one_complete_overlay(inputs):
    source, output = inputs
    command = [
        "python3", str(ROOT / "tools/prepare_lazperf_overlay.py"),
        "--source", str(source), "--output", str(output),
    ]
    processes = [subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                 for _ in range(2)]
    for process in processes:
        stdout, stderr = process.communicate(timeout=15)
        assert process.returncode == 0, stdout + stderr
    manifest = json.loads(overlay.DEFAULT_MANIFEST.read_bytes())
    overlay.checked_files(output, manifest, "after")
    assert not list(output.parent.glob("*.stage-*"))
