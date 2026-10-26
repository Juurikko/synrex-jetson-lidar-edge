# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
"""Independently verify an extracted release manifest, Git source and model/data bindings."""

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from verify_laz import verify  # noqa: E402
from verify_semantics import verify_semantics  # noqa: E402


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(root):
    manifest = json.loads((root / "publication_candidate.json").read_text())
    listed = set()
    for entry in manifest["files"]:
        path = root / entry["path"]
        if (
            path.is_symlink()
            or not path.resolve().is_relative_to(root.resolve())
            or entry["path"] in listed
        ):
            raise ValueError("unsafe/duplicate archive path")
        if path.stat().st_size != entry["bytes"] or sha(path) != entry["sha256"]:
            raise ValueError("archive artifact mismatch: " + entry["path"])
        listed.add(entry["path"])
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if actual != listed | {"publication_candidate.json"}:
        raise ValueError("unlisted/missing archive files")
    with tempfile.TemporaryDirectory(prefix="synrex-public-reconstruction-") as scratch:
        checkout = Path(scratch) / "source"
        subprocess.run(
            [
                "git",
                "clone",
                "--quiet",
                "--no-checkout",
                str((root / "source.bundle").resolve()),
                str(checkout),
            ],
            check=True,
        )
        commit = manifest["source_commit"]
        tree = subprocess.check_output(["git", "ls-tree", "-r", "-z", commit], cwd=checkout)
        count = 0
        for raw in tree.split(b"\0"):
            if not raw:
                continue
            info, name = raw.split(b"\t", 1)
            mode, kind, oid = info.split()
            if kind != b"blob" or mode not in (b"100644", b"100755"):
                raise ValueError("unsupported source object")
            path = root / "source" / name.decode()
            blob = subprocess.check_output(["git", "cat-file", "blob", oid], cwd=checkout)
            if path.read_bytes() != blob:
                raise ValueError("source export differs from Git blob")
            count += 1
    native = []
    copc = []
    for directory in [root, *sorted((root / "qgis").glob("*"))]:
        package = directory / (
            "models/unet.srxmodel" if directory == root else "model/unet.srxmodel"
        )
        taxonomy = directory / (
            "models/semantic_classes.json" if directory == root else "semantic_classes.json"
        )
        for source in sorted((directory / "data/source").glob("*.laz")):
            stem = source.stem
            classified = directory / "data/classified" / (stem + "-classified.laz")
            sidecar = directory / "data/semantics" / (stem + "-semantics.json")
            binding = directory / "data/semantics" / (stem + "-binding.json")
            native.append(verify_semantics(source, classified, sidecar, package, binding, taxonomy))
            view = directory / "views" / (stem + ".copc.laz")
            if view.exists():
                with tempfile.TemporaryDirectory(prefix="synrex-view-verify-") as scratch:
                    receipt = Path(scratch) / "receipt.json"
                    subprocess.run(
                        [
                            sys.executable,
                            str(ROOT / "tools/qgis_copc_view.py"),
                            "--classified",
                            str(classified),
                            "--sidecar",
                            str(sidecar),
                            "--copc",
                            str(view),
                            "--receipt",
                            str(receipt),
                        ],
                        check=True,
                    )
                    copc.append(json.loads(receipt.read_text()))
    synthetic = root / "data/synthetic"
    reference = verify(
        synthetic / "dense-synthetic.laz",
        synthetic / "synthetic-same-runtime.laz",
        synthetic / "expected.u8",
    )
    model_hashes = {}
    for package in sorted((root / "models").glob("*.srxmodel")):
        package_manifest = json.loads((package / "manifest.json").read_text())
        for artifact in package_manifest["artifacts"].values():
            if sha(package / artifact["path"]) != artifact["sha256"]:
                raise ValueError("model package hash mismatch")
        model_hashes[package.name] = sha(package / "manifest.json")
    return dict(
        status="PASS",
        source_commit=manifest["source_commit"],
        manifest_files=len(listed),
        source_git_blobs=count,
        native_examples=native,
        copc_views=copc,
        synthetic=reference,
        model_manifest_sha256=model_hashes,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.root)
    with args.receipt.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ["status", "source_commit", "manifest_files"]}))


if __name__ == "__main__":
    main()
