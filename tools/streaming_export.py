# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
"""Verify native streaming products and atomically publish an optional R4 QGIS COPC bundle.

Offline consumer only. Uses the unchanged qualified R4 semantic and preservation validators,
COPC source-index mapping and QGIS exporter. Never invokes neural inference.
"""

import argparse
import ctypes
import json
import os
import shutil
import tempfile
from pathlib import Path

from export_qgis import export
from seal_semantics import seal

from verify_semantics import digest, verify_semantics

ROOT = Path(__file__).resolve().parents[1]


def validate(bundle, source, model, taxonomy):
    manifest = json.loads((bundle / "bundle.json").read_text())
    if manifest["schema"] != "synrex.streaming_export_bundle.v1" or manifest["status"] != "PASS":
        raise ValueError("incomplete or unsupported export bundle")
    expected = {"labels.u8", "classified.laz", "semantics.json", "provenance.json"}
    if set(manifest["files"]) != expected or {p.name for p in bundle.iterdir()} != expected | {
        "bundle.json"
    }:
        raise ValueError("partial/unexpected product members")
    for name, value in manifest["files"].items():
        path = bundle / name
        if path.is_symlink() or not path.is_file() or digest(path) != value:
            raise ValueError("export product digest mismatch")
    bindings = manifest["bindings"]
    if bindings["source_sha256"] != digest(source) or bindings["model_manifest_sha256"] != digest(
        model / "manifest.json"
    ):
        raise ValueError("export source/model identity mismatch")
    if bindings["taxonomy_sha256"] != digest(taxonomy):
        raise ValueError("export taxonomy identity mismatch")
    receipt = verify_semantics(source, bundle / "classified.laz", bundle / "semantics.json", model)
    if receipt["points"] != manifest["points"] or manifest["coverage"] != 1:
        raise ValueError("export coverage mismatch")
    semantic_info = json.loads((bundle / "semantics.json").read_text())
    if digest(bundle / "labels.u8") != semantic_info["mapped_labels_sha256"]:
        raise ValueError("export label component mismatch")
    classes = json.loads(taxonomy.read_text())["classes"]
    if [row["semantic_id"] for row in classes] != list(range(len(bindings["las_classes"]))) or [
        row["output_las_code"] for row in classes
    ] != bindings["las_classes"]:
        raise ValueError("export taxonomy mapping mismatch")
    provenance = json.loads((bundle / "provenance.json").read_text())
    if any(provenance.get(key) != value for key, value in bindings.items()):
        raise ValueError("export provenance binding mismatch")
    if (
        len(provenance["source_frame_ids"]) != receipt["points"]
        or len(provenance["measurement_unix_ns"]) != receipt["points"]
    ):
        raise ValueError("export identity/time provenance coverage mismatch")
    return receipt


def publish(bundle, source, model, taxonomy, destination, pdal, qgis_python, windows=False):
    validate(bundle, source, model, taxonomy)
    destination = destination.absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    with tempfile.TemporaryDirectory(
        prefix=".streaming-copc-", dir=destination.parent
    ) as temporary:
        work = Path(temporary)
        package = work / "input"
        stem = source.stem
        for path in (
            package / "data/source",
            package / "data/classified",
            package / "data/semantics",
            package / "model",
        ):
            path.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, package / "data/source" / source.name)
        shutil.copy2(
            bundle / "classified.laz", package / "data/classified" / (stem + "-classified.laz")
        )
        shutil.copy2(
            bundle / "semantics.json", package / "data/semantics" / (stem + "-semantics.json")
        )
        shutil.copytree(model, package / "model/unet.srxmodel")
        seal(
            source,
            bundle / "classified.laz",
            bundle / "semantics.json",
            model,
            taxonomy,
            package / "data/semantics" / (stem + "-binding.json"),
        )
        (package / "selection.json").write_text(json.dumps({"files": [{"source": source.name}]}))
        stage = work / "complete"
        receipt = export([package], stage, pdal, qgis_python, windows)
        shutil.copy2(bundle / "provenance.json", stage / "streaming-provenance.json")
        (stage / "streaming-bundle.json").write_text(
            json.dumps(
                {
                    "schema": "synrex.streaming_copc_bundle.v1",
                    "status": "PASS",
                    "source_bundle_manifest_sha256": digest(bundle / "bundle.json"),
                    "streaming_provenance_sha256": digest(stage / "streaming-provenance.json"),
                    "r4_export_receipt_sha256": digest(stage / "export-receipt.json"),
                    "points": receipt["points"],
                },
                indent=2,
            )
            + "\n"
        )
        for path in stage.rglob("*"):
            if path.is_file():
                with path.open("rb") as f:
                    os.fsync(f.fileno())
        # Persist nested directory entries before the single atomic visibility change.
        directories = [path for path in stage.rglob("*") if path.is_dir()]
        for directory in sorted(directories, key=lambda p: len(p.parts), reverse=True) + [stage]:
            fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        libc = ctypes.CDLL(None, use_errno=True)
        rename = libc.renameat2
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        if rename(-100, os.fsencode(stage), -100, os.fsencode(destination), 1):
            raise OSError(ctypes.get_errno(), "atomic no-replace publication failed")
        fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["bundle", "source", "model", "taxonomy"]:
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--destination", type=Path)
    p.add_argument("--pdal")
    p.add_argument("--qgis-python")
    p.add_argument("--windows-tools", action="store_true")
    a = p.parse_args()
    if a.destination:
        if not a.pdal or not a.qgis_python:
            p.error("COPC export requires existing PDAL and QGIS tools")
        r = publish(
            a.bundle,
            a.source,
            a.model,
            a.taxonomy,
            a.destination,
            a.pdal,
            a.qgis_python,
            a.windows_tools,
        )
    else:
        r = validate(a.bundle, a.source, a.model, a.taxonomy)
    print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
