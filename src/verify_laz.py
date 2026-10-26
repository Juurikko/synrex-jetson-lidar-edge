# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
# Developer and project lead: Tuomas Pietilä.
# AI-assisted software engineering, including OpenAI Codex.
"""Independent laz-rs integrity verifier for the review utility's narrow LAZ contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import laspy
import numpy as np


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metadata_regions(path: Path) -> dict:
    """Parse raw metadata independently; do not ask the native writer to validate itself."""
    with path.open("rb") as stream:
        header = stream.read(375)
        if len(header) != 375 or header[:4] != b"LASF" or header[24:26] != b"\x01\x04":
            raise ValueError("requires a LAS 1.4 header")
        size, offset, count = struct.unpack_from("<HII", header, 94)
        if size != 375 or not 375 <= offset <= 16 * 1024 * 1024:
            raise ValueError("unsupported metadata prefix")
        format_id = header[104] & 63
        record_size = struct.unpack_from("<H", header, 105)[0]
        if (
            header[104] & 192 != 128
            or format_id not in (6, 7, 8)
            or record_size < {6: 30, 7: 36, 8: 38}[format_id]
        ):
            raise ValueError("requires compressed PDRF 6, 7 or 8 with valid record size")
        if any(header[227:247]):
            raise ValueError("waveform/EVLR fields must be zero")
        prefix = stream.read(offset - 375)
        if len(prefix) != offset - 375 or count > len(prefix) // 54:
            raise ValueError("truncated or invalid VLR region")
    cursor = 0
    retained, compression = [], []
    for _ in range(count):
        if cursor + 54 > len(prefix):
            raise ValueError("truncated VLR")
        user = prefix[cursor + 2 : cursor + 18].rstrip(b"\0 ")
        record_id, length = struct.unpack_from("<HH", prefix, cursor + 18)
        stop = cursor + 54 + length
        if stop > len(prefix) or user == b"copc":
            raise ValueError("truncated VLR or unsupported COPC index")
        vlr = prefix[cursor:stop]
        (compression if (user, record_id) == (b"laszip encoded", 22204) else retained).append(vlr)
        cursor = stop
    if len(compression) != 1:
        raise ValueError("requires one LASzip compression VLR")
    return {"header": header, "vlrs": retained, "padding": prefix[cursor:]}


def verify(source: Path, output: Path, labels: Path, chunk_points: int = 7001) -> dict:
    """Require exact metadata/ordered nonclass bytes and supplied class-byte identity."""
    if chunk_points <= 0:
        raise ValueError("chunk_points must be positive")
    a_meta, b_meta = metadata_regions(source), metadata_regions(output)
    if a_meta != b_meta:
        raise ValueError("public header/non-LASzip metadata/padding changed")
    records_digest = hashlib.sha256()
    classes_digest = hashlib.sha256()
    count = 0
    changed = 0
    with (
        laspy.open(source, laz_backend=laspy.LazBackend.Lazrs) as a,
        laspy.open(output, laz_backend=laspy.LazBackend.Lazrs) as b,
        labels.open("rb") as expected,
    ):
        if (
            a.header.point_count != b.header.point_count
            or labels.stat().st_size != a.header.point_count
        ):
            raise ValueError("point/label count mismatch")
        while count < a.header.point_count:
            left = a.read_points(chunk_points)
            right = b.read_points(chunk_points)
            if len(left) == 0 or len(left) != len(right) or left.array.dtype != right.array.dtype:
                raise ValueError("decoded record count/type mismatch")
            record_size = left.array.dtype.itemsize
            x = left.array.view(np.uint8).reshape(len(left), record_size).copy()
            y = right.array.view(np.uint8).reshape(len(right), record_size).copy()
            wanted = expected.read(len(left))
            actual = y[:, 16].tobytes()
            if actual != wanted:
                raise ValueError("classification does not match supplied source-order labels")
            classes_digest.update(actual)
            changed += int(np.count_nonzero(x[:, 16] != y[:, 16]))
            x[:, 16] = 0
            y[:, 16] = 0
            if not np.array_equal(x, y):
                raise ValueError("source-order nonclassification record bytes changed")
            records_digest.update(x.tobytes())
            count += len(left)
        if len(a.read_points(1)) or len(b.read_points(1)) or expected.read(1):
            raise ValueError("unexpected trailing points or labels")
    labels_hash = classes_digest.hexdigest()
    if labels_hash != digest_file(labels):
        raise ValueError("classification digest mismatch")
    return {
        "status": "PASS",
        "decoder": "laz-rs, explicitly selected through laspy.LazBackend.Lazrs",
        "points": count,
        "classification_records_changed": changed,
        "point_order_and_nonclassification_bytes_exact": True,
        "public_header_non_laszip_vlrs_and_padding_exact": True,
        "source_sha256": digest_file(source),
        "output_sha256": digest_file(output),
        "canonical_nonclassification_sha256": records_digest.hexdigest(),
        "classification_sha256": labels_hash,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("labels", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.source, args.output, args.labels), indent=2))


if __name__ == "__main__":
    main()
