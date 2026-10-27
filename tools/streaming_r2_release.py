# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
"""Stage or independently verify a local Streaming R2 RC. Never uploads or publishes.

Large licensed data, models and video are release assets, never normal source history.
Staging requires clean committed source. Native source must remain byte-identical to R1.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

from streaming_r2_replay import verify as verify_recording

ROOT = Path(__file__).resolve().parents[1]
QUALIFIED = "f39fa296332217eb77bd73dd7e1e3515be3f367e"
SEAL = "14bcb4f2c512d735291f796ac905584564e74ddf"
R4 = "3f1f3fabd3d92d69c02394936188502678a6b9ed"


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def save(path, value):
    with path.open("x") as f:
        f.write(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def verify_models(directory):
    identity = json.loads((ROOT / "evidence/unet_r4_release_candidate.json").read_text())
    for name, expected in identity["models_sha256"].items():
        if isinstance(expected, dict):
            for member, value in expected.items():
                assert sha(directory / name / member) == value, (name, member)
        else:
            assert sha(directory / name) == expected, name


def stage(a):
    assert not git("status", "--porcelain"), "Commit qualified source before staging"
    commit = git("rev-parse", "HEAD")
    assert git("rev-parse", "byom-unet-fractal-r4-rc2^{}") == R4
    freeze = json.loads((ROOT / "evidence/streaming_r2_baseline_freeze.json").read_text())
    assert all(sha(ROOT / p) == h for p, h in freeze["protected_files"].items())
    for p in (ROOT / "native").rglob("*"):
        if p.is_file():
            expected = subprocess.check_output(
                ["git", "show", f"{QUALIFIED}:{p.relative_to(ROOT)}"], cwd=ROOT
            )
            assert p.read_bytes() == expected
    a.out.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory() as temporary:
        archive = Path(temporary) / "source.tar"
        with archive.open("wb") as f:
            subprocess.run(["git", "archive", commit], cwd=ROOT, stdout=f, check=True)
        source = a.out / "source"
        source.mkdir()
        with tarfile.open(archive) as t:
            t.extractall(source, filter="data")
    subprocess.run(
        [
            "git",
            "bundle",
            "create",
            str(a.out.resolve() / "source.bundle"),
            "codex/byom-streaming-r0",
            "byom-unet-fractal-r4-rc2",
            "streaming-r0-2m-frozen-95e80bb",
            "streaming-r0-geometry-frozen-958c418",
            "streaming-r0-eviction-2m-20261027",
        ],
        cwd=ROOT,
        check=True,
    )
    shutil.copytree(a.foundation / "models", a.out / "models")
    verify_models(a.out / "models")
    shutil.copytree(
        a.foundation / "qgis/ready",
        a.out / "qgis-r4-foundation",
        ignore=shutil.ignore_patterns("profiles", "*.log", "*.cmd", "*.qgz~"),
    )
    shutil.copytree(
        a.qgis,
        a.out / "qgis",
        ignore=shutil.ignore_patterns(
            "profiles",
            "*.cmd",
            "*-attempt.json",
            "render-started.txt",
            "*.log",
            "streaming_r2_*.py",
        ),
    )
    shutil.copytree(a.recording, a.out / "recording")
    shutil.copytree(a.media, a.out / "media")
    shutil.copytree(a.native_export, a.out / "native-export")
    shutil.copytree(ROOT / "LICENSES", a.out / "LICENSES")
    benchmark = a.out / "data/benchmarks"
    benchmark.mkdir(parents=True)
    allowed = dict(
        zip(
            [
                "TEST-0447_6390-002819414.laz",
                "TEST-0437_6398-002899556.laz",
                "TEST-0944_6564-005900112.laz",
            ],
            [
                "1bbba7aacc64a0c483df6f7fb970f0b40b7115b4a20989628a0ca4677fdfb4c2",
                "a20166855fddc71a7136f9a7eacc0b6f4ac9d101324500331762736b1755b46f",
                "50106aa5d6465707523bf9252ac8ade9e683266b61a94f5eadde6eaeb2276274",
            ],
            strict=True,
        )
    )
    assert {p.name for p in a.corpora.glob("*.laz")} == set(allowed)
    for path in sorted(a.corpora.glob("*.laz")):
        assert sha(path) == allowed[path.name]
        shutil.copy2(path, benchmark / path.name)
    (a.out / "README.md").write_text(
        "# SynRex Streaming R2 RC1 — local owner review\n\n"
        "Native runtime source remains the accepted R1 implementation. "
        "No publication performed.\n\n"
        "- [Release notes](source/docs/streaming/STREAMING_R2_RELEASE.md)\n"
        "- [Reproduction](source/docs/streaming/STREAMING_R2_REPRODUCIBILITY.md)\n"
        "- [Rights and provenance](source/docs/streaming/STREAMING_R2_RIGHTS.md)\n"
        "- [Video](media/synrex-streaming-r2-review.mp4)\n"
        "- [Contact sheet](media/contact-sheet.png)\n\n"
        "Open `qgis/synrex-fractal.qgz` for scene C, or the published R4 foundation project "
        "in `qgis-r4-foundation/`. Models, data, original presentation and third-party tools "
        "have separate licence scopes. This archive is not under a blanket MIT licence.\n"
    )
    files = {
        str(p.relative_to(a.out)): dict(sha256=sha(p), bytes=p.stat().st_size)
        for p in sorted(a.out.rglob("*"))
        if p.is_file()
    }
    save(
        a.out / "RELEASE_MANIFEST.json",
        dict(
            schema="synrex.streaming_r2.release.v1",
            version="streaming-r2-rc1",
            status="LOCAL_REVIEW_CANDIDATE",
            source_commit=commit,
            qualified_native_source=QUALIFIED,
            accepted_r1_evidence_seal=SEAL,
            public_foundation=R4,
            files=files,
            publication="NOT_RUN; explicit owner approval required before any push/upload/post",
            license_scope=(
                "Scoped MIT original source; IGN data Etalab-2.0; "
                "model/SDK and media notices separate"
            ),
            source_bundle_branch="codex/byom-streaming-r0",
        ),
    )
    with tarfile.open(a.out.with_suffix(".tar.gz"), "w:gz") as t:
        t.add(a.out, arcname=a.out.name)
    with tarfile.open(a.out.parent / "streaming-r2-rc1-source.tar.gz", "w:gz") as t:
        t.add(a.out / "source", arcname="streaming-r2-rc1-source")
    print(
        json.dumps(
            dict(
                status="PASS",
                source_commit=commit,
                manifest_sha256=sha(a.out / "RELEASE_MANIFEST.json"),
                archive_sha256=sha(a.out.with_suffix(".tar.gz")),
            ),
            indent=2,
        )
    )


def verify(a):
    manifest = json.loads((a.out / "RELEASE_MANIFEST.json").read_text())
    expected = set(manifest["files"]) | {"RELEASE_MANIFEST.json"}
    actual = {str(p.relative_to(a.out)) for p in a.out.rglob("*") if p.is_file()}
    verify_models(a.out / "models")
    assert expected == actual, "Unexpected/missing release members"
    for name, row in manifest["files"].items():
        p = a.out / name
        assert (
            not p.is_symlink() and p.stat().st_size == row["bytes"] and sha(p) == row["sha256"]
        ), name
    subprocess.run(
        ["git", "bundle", "verify", str(a.out.resolve() / "source.bundle")],
        cwd=ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    with tempfile.TemporaryDirectory() as temporary:
        work = Path(temporary)
        checkout = work / "checkout"
        subprocess.run(
            [
                "git",
                "clone",
                "--quiet",
                "--branch",
                manifest["source_bundle_branch"],
                str(a.out.resolve() / "source.bundle"),
                str(checkout),
            ],
            check=True,
        )
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=checkout, text=True
        ).strip()
        assert head == manifest["source_commit"]
        tracked = subprocess.check_output(["git", "ls-files"], cwd=checkout, text=True).splitlines()
        assert len(tracked) == len([n for n in manifest["files"] if n.startswith("source/")])
        for name in tracked:
            assert (checkout / name).read_bytes() == (a.out / "source" / name).read_bytes(), name
        recording = work / "recording"
        recording.mkdir()
        for name in ["input.jsonl", "events.jsonl", "native-receipt.json", "input-provenance.json"]:
            shutil.copy2(a.out / "recording" / name, recording / name)
        verify_recording(
            a.out / "qgis/data/source/TEST-0944_6564-005900112.laz",
            recording,
            a.out / "models/unet.srxmodel",
        )
        proof = json.loads((recording / "verification.json").read_text())
        old = json.loads((a.out / "recording/verification.json").read_text())
        for key in [
            "source_sha256",
            "engine_sha256",
            "predictions_sha256",
            "coverage",
            "map_deltas",
        ]:
            assert proof[key] == old[key], key
        # NPZ zip timestamps are not semantic content. Compare all arrays exactly.
        import numpy as np

        first = np.load(recording / "snapshots.npz", allow_pickle=False)
        second = np.load(a.out / "recording/snapshots.npz", allow_pickle=False)
        assert first.files == second.files
        assert all(np.array_equal(first[k], second[k]) for k in first.files)
    print(
        json.dumps(
            dict(
                status="PASS",
                files=len(actual),
                source_git_blobs=len(tracked),
                fresh_source_coverage=1.0,
                exact_native_map_snapshots=True,
                release_manifest_sha256=sha(a.out / "RELEASE_MANIFEST.json"),
            ),
            indent=2,
        )
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["stage", "verify"])
    p.add_argument("--out", type=Path, required=True)
    for name in ["foundation", "qgis", "recording", "media", "native-export", "corpora"]:
        p.add_argument("--" + name, type=Path)
    a = p.parse_args()
    if a.command == "stage":
        if any(
            getattr(a, name) is None
            for name in ["foundation", "qgis", "recording", "media", "native_export", "corpora"]
        ):
            p.error("stage requires all artifact paths")
        stage(a)
    else:
        verify(a)


if __name__ == "__main__":
    main()
