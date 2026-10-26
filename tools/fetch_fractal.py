# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
"""Fetch a fixed, bounded FRACTAL pilot using HTTP ranges, retaining ZIP CRCs.

Real data remain Etalab-2.0, not MIT. No upstream model/weights are downloaded.
"""

import argparse
import concurrent.futures
import hashlib
import io
import json
import struct
import time
import urllib.request
import zipfile
import zlib
from pathlib import Path

REVISION = "36e00245ef92fd180f655f6ee410f2e935693cc2"
BASE = f"https://huggingface.co/datasets/IGNF/FRACTAL/resolve/{REVISION}"


def fetch(url, start, length):
    for attempt in range(5):
        try:
            request = urllib.request.Request(
                url, headers={"Range": f"bytes={start}-{start + length - 1}"}
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                if response.status != 206:
                    raise ValueError("server did not honor bounded range request")
                data = response.read()
                if len(data) != length:
                    raise ValueError("short HTTP range")
                return data
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2**attempt)


class RemoteZip(io.RawIOBase):
    def __init__(self, url, size):
        self.url, self.size, self.position = url, size, 0

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        self.position = offset + (0 if whence == 0 else self.position if whence == 1 else self.size)
        return self.position

    def tell(self):
        return self.position

    def read(self, size=-1):
        size = min(size if size >= 0 else self.size, self.size - self.position)
        if size <= 0:
            return b""
        result = fetch(self.url, self.position, size)
        self.position += size
        return result


def retrieve(url, info, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        content = destination.read_bytes()
    else:
        header = fetch(url, info.header_offset, 30)
        if header[:4] != b"PK\x03\x04":
            raise ValueError("invalid local ZIP header")
        name_length, extra_length = struct.unpack_from("<HH", header, 26)
        payload = fetch(
            url, info.header_offset + 30 + name_length + extra_length, info.compress_size
        )
        content = zlib.decompress(payload, -15) if info.compress_type == 8 else payload
    if len(content) != info.file_size or zlib.crc32(content) != info.CRC:
        raise ValueError(f"ZIP CRC/size mismatch: {info.filename}")
    destination.write_bytes(content)
    return {
        "member": info.filename,
        "bytes": len(content),
        "zip_crc32": f"{info.CRC:08x}",
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train", type=int, default=256)
    parser.add_argument("--val", type=int, default=32)
    parser.add_argument("--test", type=int, default=32)
    args = parser.parse_args()
    receipt = {
        "dataset": "IGNF/FRACTAL",
        "revision": REVISION,
        "license": "etalab-2.0",
        "selection": "first sorted requested LAZ members of split archive 00; pilot only",
        "files": {},
    }
    for split in ("train", "val", "test"):
        count = getattr(args, split)
        if not 1 <= count <= 1000:
            raise ValueError("split count must be 1..1000")
        url = f"{BASE}/data/{split}/{split}-00.zip"
        with urllib.request.urlopen(url, timeout=120) as response:
            # Close immediately: obtain total length without downloading the full archive.
            size = int(response.headers["Content-Length"])
        with zipfile.ZipFile(RemoteZip(url, size)) as archive:
            members = sorted(
                (i for i in archive.infolist() if i.filename.endswith(".laz")),
                key=lambda i: i.filename,
            )[:count]
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            futures = [
                pool.submit(retrieve, url, i, args.out / split / Path(i.filename).name)
                for i in members
            ]
            records = [f.result() for f in futures]
        receipt["files"][split] = {"archive_url": url, "archive_bytes": size, "tiles": records}
        print(f"{split}: {len(records)} verified real LAZ tiles", flush=True)
        (args.out / "dataset.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
