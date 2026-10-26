# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
"""Record scoped per-file provenance and excluded private source identities; never copy source."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "147ece0"
OUTPUTS = {"evidence/byom_provenance.json", "evidence/byom_source_assessment.json"}
SOURCE_SUFFIXES = {".cpp", ".cu", ".hpp", ".h", ".py"}


def git(*args, cwd=ROOT):
    return subprocess.check_output(["git", *args], cwd=cwd)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(private_source=None):
    scope = json.loads((ROOT / "LICENSE_SCOPE.json").read_text())
    names = sorted(
        set(
            git("ls-files", "-z", "--cached", "--others", "--exclude-standard").decode().split("\0")
        )
        - {""}
    )
    for name in names:
        path = Path(name)
        source = path.suffix in SOURCE_SUFFIXES or name in {
            "CMakeLists.txt",
            "requirements-byom.lock",
            "requirements-unet.lock",
            "requirements-unet-validation.lock",
            "examples/dense_voxel.json",
            "examples/unet/dense_voxel.json",
        }
        if source and name not in scope["source"]["files"]:
            scope["source"]["files"].append(name)
        if (name.startswith("docs/byom/") or name == "BRING_YOUR_OWN_MODEL.md") and name.endswith(
            ".md"
        ):
            if name not in scope["documentation"]["files"]:
                scope["documentation"]["files"].append(name)
    for section in ("source", "documentation"):
        scope[section]["files"].sort()
    (ROOT / "LICENSE_SCOPE.json").write_text(json.dumps(scope, indent=2, ensure_ascii=False) + "\n")
    baseline = set(git("ls-tree", "-r", "--name-only", BASE).decode().splitlines())
    records = []
    for name in names:
        if name in OUTPUTS:
            continue
        path = ROOT / name
        assert path.is_file() and not path.is_symlink(), name
        category = next(
            (
                s
                for s in (
                    "source",
                    "documentation",
                    "original_procedural_media",
                    "data_derived_media",
                    "data_derived_metadata",
                )
                if s in scope
                if name in scope[s]["files"]
            ),
            None,
        )
        if category:
            license_id = scope[category]["license"]
            origin = (
                "Retained authorized public implementation/expression"
                if name in baseline
                else "New generic BYOM work under Tuomas Pietilä's assignment; "
                "AI-assisted with OpenAI Codex"
            )
            if (
                name in baseline
                and digest(path) != hashlib.sha256(git("show", f"{BASE}:{name}")).hexdigest()
            ):
                origin += "; modified for BYOM R0"
            if category == "data_derived_media":
                origin = "Derived IGN FRACTAL visualization; original presentation Tuomas Pietilä"
            if category == "data_derived_metadata":
                origin = "IGN FRACTAL taxonomy facts and original model/projection metadata layout"
            if path.suffix in SOURCE_SUFFIXES:
                assert "SPDX-License-Identifier: MIT" in path.read_text(), name
        elif name.startswith("LICENSES/"):
            license_id = "Third-party/standard license text; see THIRD_PARTY_NOTICES.md"
            origin = "Separate upstream notice or preserved historical license record"
        elif name.startswith("evidence/"):
            license_id = "Factual evidence; no personal claim over underlying facts/upstream"
            origin = (
                "Historical qualification record"
                if name in baseline
                else "BYOM R0 local qualification record"
            )
        else:
            license_id = "Scope metadata; grants limited to LICENSE_SCOPE.json"
            origin = (
                "Retained public project record" if name in baseline else "BYOM project metadata"
            )
        records.append(
            {"file": name, "sha256": digest(path), "license": license_id, "origin": origin}
        )
    (ROOT / "evidence/byom_provenance.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False) + "\n"
    )
    if private_source:
        assert git("status", "--porcelain", cwd=private_source) == b"", (
            "private reference must remain pristine"
        )
        excluded = []
        for path in sorted((private_source / "native").rglob("*")):
            if path.is_file() and path.suffix in SOURCE_SUFFIXES:
                excluded.append(
                    {
                        "file": str(path.relative_to(private_source)),
                        "sha256": digest(path),
                        "disposition": "EXCLUDED_UNCLEAR_PUBLIC_RELICENSE_RIGHTS",
                        "reason": "No blanket permissive grant established; "
                        "model-specific reference lineage. No source included.",
                    }
                )
        result = {
            "schema": "synrex.byom.source_assessment.v1",
            "private_reference_commit": git("rev-parse", "HEAD", cwd=private_source)
            .decode()
            .strip(),
            "private_reference_worktree": "PRISTINE",
            "private_source_included": False,
            "restricted_model_artifacts_included": False,
            "clean_room_claim": False,
            "reused_public_source": [
                "src/preserve_laz.cpp",
                "src/verify_laz.py",
                "src/synthetic_laz.py",
                "tests/test_preservation.py",
            ],
            "new_source_basis": "Generic explicit adapter contract and public NVIDIA APIs; "
            "docs/byom/PROVENANCE.md",
            "excluded_private_native_files": excluded,
        }
        (ROOT / "evidence/byom_source_assessment.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
    print(
        json.dumps(
            {
                "status": "PASS",
                "files_recorded": len(records),
                "boundary": "scoped source/provenance assessment, not legal opinion",
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-source", type=Path)
    args = parser.parse_args()
    audit(args.private_source)
