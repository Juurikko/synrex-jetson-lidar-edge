# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
# ruff: noqa: E402
"""Prepare immutable native examples for the qualified QGIS3.44/38-byte scope.

Display selection only, after freeze. Does not classify, tune or alter any LAZ.
"""

import argparse
import json
import shutil
import struct
import sys
from collections import Counter
from pathlib import Path

import laspy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from verify_semantics import digest, verify_semantics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["data", "qualified", "out"]:
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--copc-views", type=Path)
    a = p.parse_args()
    best = {i: (0, None) for i in range(7)}
    for sidecar in [] if a.copc_views else sorted(a.qualified.glob("*-semantics.json")):
        name = sidecar.name.replace("-semantics.json", ".laz")
        source = a.data / "test" / name
        with source.open("rb") as f:
            header = f.read(107)
        if (header[104] & 63) != 8 or struct.unpack_from("<H", header, 105)[0] != 38:
            continue
        counts = Counter(json.loads(sidecar.read_text())["semantic_ids"])
        for i in best:
            if counts[i] > best[i][0]:
                best[i] = counts[i], name
    if not a.copc_views and any(name is None for _, name in best.values()):
        raise ValueError("Cannot cover all seven classes in qualified viewer scope")
    chosen_names = sorted({value[1] for value in best.values()})
    if a.copc_views:
        chosen_names = sorted(
            p.name.replace(".copc.laz", ".laz") for p in a.copc_views.glob("*.copc.laz")
        )
        if not chosen_names:
            raise ValueError("no independently verified COPC views")
    a.out.mkdir(parents=True, exist_ok=False)
    rows = []
    for name in chosen_names:
        source = a.data / "test" / name
        stem = source.stem
        sidecar = a.qualified / (stem + "-semantics.json")
        output = a.qualified / (stem + "-classified.laz")
        binding = a.qualified / (stem + "-binding.json")
        verify_semantics(
            source,
            output,
            sidecar,
            a.qualified / "unet.srxmodel",
            binding,
            ROOT / "examples/unet/semantic_classes.json",
        )
        for directory, path in [
            ("data/source", source),
            ("data/classified", output),
            ("data/semantics", sidecar),
            ("data/semantics", binding),
        ]:
            destination = a.out / directory
            destination.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination / path.name)
        ids = json.loads(sidecar.read_text())["semantic_ids"]
        counts = Counter(ids)
        row = dict(source=name, points=len(ids), semantic_counts=[counts[i] for i in range(7)])
        if a.copc_views:
            view = a.copc_views / (stem + ".copc.laz")
            verification = a.copc_views / (stem + "-verified.json")
            proof = json.loads(verification.read_text())
            if (
                proof["status"] != "PASS"
                or proof["copc_sha256"] != digest(view)
                or proof["original_classified_sha256"] != digest(output)
                or proof["original_semantic_sidecar_sha256"] != digest(sidecar)
            ):
                raise ValueError("COPC verification binding mismatch")
            views = a.out / "views"
            views.mkdir(exist_ok=True)
            shutil.copy2(view, views / view.name)
            shutil.copy2(verification, views / verification.name)
            with laspy.open(source, laz_backend=laspy.LazBackend.Lazrs) as reader:
                row["original_wkt"] = next(
                    v.string
                    for v in reader.header.vlrs
                    if v.user_id == "LASF_Projection" and v.record_id == 2112
                )
            row["view_filename"] = "views/" + view.name
        rows.append(row)
    shutil.copytree(a.qualified / "unet.srxmodel", a.out / "model/unet.srxmodel")
    for name in [
        "semantic_classes.json",
        "fractal-classification.qml",
        "fractal-classification-r3.qml",
        "MODEL_NOTICE.txt",
    ]:
        shutil.copy2(ROOT / "examples/unet" / name, a.out / name)
    shutil.copy2(ROOT / "tools/qualify_qgis.py", a.out / "qualify_qgis.py")
    shutil.copytree(ROOT / "LICENSES", a.out / "LICENSES")
    selection = (
        "Post-freeze maximum frozen predicted class support among PDRF8/38-byte "
        "files, display only; original48-byte QGIS rejection remains negative evidence"
    )
    if a.copc_views:
        selection = "Original direct-LAZ rejection cases; independently verified COPC views"
    (a.out / "selection.json").write_text(
        json.dumps(dict(selection=selection, files=rows, copc_views=bool(a.copc_views)), indent=2)
        + "\n"
    )
    files = [
        dict(path=p.relative_to(a.out).as_posix(), sha256=digest(p))
        for p in sorted(a.out.rglob("*"))
        if p.is_file()
    ]
    (a.out / "immutable-inputs.json").write_text(json.dumps(files, indent=2) + "\n")


if __name__ == "__main__":
    main()
