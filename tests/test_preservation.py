# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
# Developer and project lead: Tuomas Pietilä.
# AI-assisted software engineering, including OpenAI Codex.
"""Real native LAZ round trips plus independent negative integrity checks."""

import hashlib
import os
import struct
import subprocess
from pathlib import Path

import laspy
import numpy as np
import pytest

from synthetic_laz import create
from verify_laz import metadata_regions, verify

WRITER = Path(os.environ.get("SYNREX_TEST_WRITER", "build/synrex-laz-labels")).resolve()


def run(source, labels, output):
    return subprocess.run(
        [str(WRITER), str(source), str(labels), str(output)], capture_output=True, text=True
    )


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("points", [1, 257, 50013])
def test_native_roundtrip_preserves_every_other_byte(tmp_path, points):
    source, labels = create(tmp_path, points)
    before = {p.name: file_hash(p) for p in [source, labels]}
    output = tmp_path / "classified.laz"
    completed = run(source, labels, output)
    assert completed.returncode == 0, completed.stderr
    result = verify(source, output, labels)
    assert result["points"] == points
    assert result["classification_records_changed"] == points
    assert before == {p.name: file_hash(p) for p in [source, labels]}
    assert metadata_regions(source) == metadata_regions(output)
    with laspy.open(output, laz_backend=laspy.LazBackend.Lazrs) as decoded:
        assert decoded.header.global_encoding.wkt
        assert any(
            v.user_id == "LASF_Projection" and v.record_id == 2112 for v in decoded.header.vlrs
        )
    assert not list(tmp_path.glob(".synrex-laz-review-*"))


@pytest.mark.parametrize("change", ["attributes", "order", "classification"])
def test_independent_verifier_detects_record_tampering(tmp_path, change):
    source, labels = create(tmp_path)
    output = tmp_path / "classified.laz"
    assert run(source, labels, output).returncode == 0
    data = laspy.read(output, laz_backend=laspy.LazBackend.Lazrs)
    if change == "attributes":
        data.intensity[10] ^= 1
    elif change == "order":
        data.points = data.points[::-1].copy()
    else:
        data.classification[10] = 255
    altered = tmp_path / "altered.laz"
    data.write(altered, laz_backend=laspy.LazBackend.Lazrs)
    with pytest.raises(ValueError):
        verify(source, altered, labels)


@pytest.mark.parametrize("region", ["header", "vlr", "padding"])
def test_independent_verifier_detects_metadata_tampering(tmp_path, region):
    source, labels = create(tmp_path)
    output = tmp_path / "classified.laz"
    assert run(source, labels, output).returncode == 0
    raw = bytearray(output.read_bytes())
    if region == "header":
        raw[4] ^= 1
    elif region == "padding":
        raw[struct.unpack_from("<I", raw, 96)[0] - 1] ^= 1
    else:
        # Alter the description byte of the first noncompression VLR.
        cursor = 375
        while raw[cursor + 2 : cursor + 18].rstrip(b"\0 ") == b"laszip encoded":
            cursor += 54 + struct.unpack_from("<H", raw, cursor + 20)[0]
        raw[cursor + 22] ^= 1
    output.write_bytes(raw)
    with pytest.raises(ValueError, match="metadata"):
        verify(source, output, labels)


@pytest.mark.parametrize("delta", [-1, 1])
def test_writer_rejects_wrong_label_count_without_output(tmp_path, delta):
    source, labels = create(tmp_path)
    labels.write_bytes(labels.read_bytes()[:-1] if delta < 0 else labels.read_bytes() + b"\x02")
    before = file_hash(source)
    output = tmp_path / "classified.laz"
    result = run(source, labels, output)
    assert result.returncode != 0 and "label count" in result.stderr
    assert not output.exists() and file_hash(source) == before
    assert not list(tmp_path.glob(".synrex-laz-review-*"))


def test_writer_never_overwrites_existing_output(tmp_path):
    source, labels = create(tmp_path)
    output = tmp_path / "classified.laz"
    output.write_bytes(b"existing-review-artifact")
    assert run(source, labels, output).returncode != 0
    assert output.read_bytes() == b"existing-review-artifact"


@pytest.mark.parametrize("change", ["format", "evlr", "offset", "vlr_count", "truncation"])
def test_writer_fails_closed_on_unsupported_or_malformed_input(tmp_path, change):
    source, labels = create(tmp_path)
    raw = bytearray(source.read_bytes())
    if change == "format":
        raw[104] = 128 | 6
    elif change == "evlr":
        struct.pack_into("<I", raw, 243, 1)
    elif change == "offset":
        struct.pack_into("<I", raw, 96, 374)
    elif change == "vlr_count":
        struct.pack_into("<I", raw, 100, 1000000)
    else:
        raw = raw[:300]
    source.write_bytes(raw)
    output = tmp_path / "classified.laz"
    assert run(source, labels, output).returncode != 0
    assert not output.exists()
    assert not list(tmp_path.glob(".synrex-laz-review-*"))


def test_full_uint8_classification_and_flags_are_distinct(tmp_path):
    source, labels = create(tmp_path)
    labels.write_bytes(np.arange(257, dtype=np.uint16).astype(np.uint8).tobytes())
    output = tmp_path / "classified.laz"
    assert run(source, labels, output).returncode == 0
    assert verify(source, output, labels)["points"] == 257


def test_no_output_when_parent_is_absent(tmp_path):
    source, labels = create(tmp_path)
    output = tmp_path / "missing" / "classified.laz"
    assert run(source, labels, output).returncode != 0
    assert not output.parent.exists()
