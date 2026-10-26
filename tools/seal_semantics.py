# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
# ruff: noqa: E402
"""Independently validate then bind immutable LAZ/sidecar/package/taxonomy hashes.

This post-inference envelope does not alter any native output or semantic ID.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from verify_semantics import digest, verify_semantics


def seal(source, output, sidecar, package, taxonomy, destination):
    receipt = verify_semantics(source, output, sidecar, package)
    classes = json.loads(taxonomy.read_text())["classes"]
    mapping = json.loads((package / "manifest.json").read_text())["contract"]["las_classes"]
    if [row["semantic_id"] for row in classes] != list(range(len(mapping))) or [
        row["output_las_code"] for row in classes
    ] != mapping:
        raise ValueError("taxonomy and package mismatch")
    envelope = dict(
        schema="synrex.semantic_output_binding.v1",
        source_sha256=digest(source),
        output_sha256=digest(output),
        semantic_sidecar_sha256=digest(sidecar),
        model_manifest_sha256=digest(package / "manifest.json"),
        taxonomy_sha256=digest(taxonomy),
        points=receipt["points"],
        index_order="original source point record order",
        independent_validation=receipt,
    )
    with destination.open("x") as f:
        f.write(json.dumps(envelope, indent=2) + "\n")
    return envelope


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["source", "output", "sidecar", "package", "taxonomy", "destination"]:
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    seal(a.source, a.output, a.sidecar, a.package, a.taxonomy, a.destination)


if __name__ == "__main__":
    main()
