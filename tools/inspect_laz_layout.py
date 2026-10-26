# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
"""Read raw LAS headers/VLRs and independently decode their LAZ point records."""

import argparse
import hashlib
import json
import struct
from pathlib import Path

import laspy


def inspect(path):
    def text(data):
        return data.split(b"\0", 1)[0].decode("utf-8", errors="replace")

    with path.open("rb") as stream:
        header = stream.read(375)
        if header[:4] != b"LASF" or tuple(header[24:26]) != (1, 4):
            raise ValueError("inspection scope is LAS 1.4")
        length = struct.unpack_from("<H", header, 105)[0]
        offset = struct.unpack_from("<I", header, 96)[0]
        vlrs = []
        stream.seek(struct.unpack_from("<H", header, 94)[0])
        for _ in range(struct.unpack_from("<I", header, 100)[0]):
            raw = stream.read(54)
            if len(raw) != 54:
                raise ValueError("truncated VLR")
            user = text(raw[2:18])
            record, size = struct.unpack_from("<HH", raw, 18)
            data = stream.read(size)
            if len(data) != size or stream.tell() > offset:
                raise ValueError("VLR overlaps point records")
            row = dict(
                user=user, record=record, bytes=size, sha256=hashlib.sha256(data).hexdigest()
            )
            if user == "LASF_Spec" and record == 4:
                if size % 192:
                    raise ValueError("invalid Extra Bytes descriptor length")
                descriptors = []
                for start in range(0, size, 192):
                    descriptor = data[start : start + 192]
                    code, options = descriptor[2:4]
                    widths = [1, 1, 2, 2, 4, 4, 8, 8, 4, 8]
                    if not 0 <= code <= 30:
                        raise ValueError("invalid Extra Bytes data type")
                    width = (
                        options if code == 0 else widths[(code - 1) % 10] * ((code - 1) // 10 + 1)
                    )
                    descriptors.append(
                        dict(
                            name=text(descriptor[4:36]),
                            data_type=code,
                            options=options,
                            bytes=width,
                        )
                    )
                row["descriptors"] = descriptors
            if user == "laszip encoded" and record == 22204:
                compressor, coder, major, minor, revision, options, chunk, nevlr, oevlr, n = (
                    struct.unpack_from("<HHBBHIIqqH", data)
                )
                row["laszip"] = dict(
                    compressor=compressor,
                    coder=coder,
                    version=f"{major}.{minor}.{revision}",
                    options=options,
                    chunk_size=chunk,
                    special_evlrs=nevlr,
                    special_evlr_offset=oevlr,
                    items=[struct.unpack_from("<HHH", data, 34 + i * 6) for i in range(n)],
                )
            if record == 2112:
                row["wkt"] = text(data)
            vlrs.append(row)
    cloud = laspy.read(path, laz_backend=laspy.LazBackend.Lazrs)
    base = {6: 30, 7: 36, 8: 38}[header[104] & 63]
    extra = sum(d["bytes"] for v in vlrs for d in v.get("descriptors", []))
    if length != base + extra or cloud.points.array.dtype.itemsize != length:
        raise ValueError("record length/Extra Bytes mismatch")
    return dict(
        filename=path.name,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        version="1.4",
        compressed=bool(header[104] & 128),
        point_format=header[104] & 63,
        record_length=length,
        point_offset=offset,
        points=len(cloud),
        base_bytes=base,
        declared_extra_bytes=extra,
        vlrs=vlrs,
        independent_decoder="laz-rs via laspy",
        decoded_point_bytes=len(cloud) * length,
        decompressed_sha256=hashlib.sha256(cloud.points.array.tobytes()).hexdigest(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", type=Path, nargs="+")
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    result = dict(status="PASS", files=[inspect(p) for p in args.paths])
    with args.receipt.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
