# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
"""Prepare a local, hashed, licensed showcase archive; never uploads or publishes."""

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "trained", "qualified", "engine-package", "synthetic", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    qualification = json.loads((args.qualified / "qualification.json").read_text())
    if qualification["status"] != "PASS":
        raise ValueError("require successful genuine qualification before showcase preparation")
    package = args.engine_package
    manifest = json.loads((package / "manifest.json").read_text())
    for artifact in manifest["artifacts"].values():
        assert sha(package / artifact["path"]) == artifact["sha256"]
    assert sha(package / "model.engine") == qualification["engine_sha256"]
    assert sha(args.trained / "unet-fp16.onnx") == qualification["onnx_fp16_sha256"]
    assert sha(args.trained / "unet-fp32.onnx") == qualification["onnx_fp32_sha256"]
    assert "2025-04-05" in (package / "MODEL_LICENSE.txt").read_text()
    args.out.mkdir(parents=True, exist_ok=False)
    shutil.copytree(package, args.out / "models/unet.srxmodel")
    for name in ("best.pt", "unet-fp16.onnx", "unet-fp32.onnx"):
        shutil.copy2(args.trained / name, args.out / "models" / name)
    shutil.copy2(ROOT / "examples/unet/MODEL_NOTICE.txt", args.out / "models/MODEL_NOTICE.txt")
    shutil.copy2(ROOT / "examples/unet/dense_voxel.json", args.out / "models/dense_voxel.json")
    shutil.copytree(args.synthetic / "engine.srxmodel", args.out / "models/synthetic.srxmodel")
    shutil.copy2(args.synthetic / "reference.onnx", args.out / "models/synthetic.onnx")
    synthetic_dir = args.out / "data/synthetic"
    synthetic_dir.mkdir(parents=True)
    for path in (
        args.synthetic / "scene/dense-synthetic.laz",
        args.synthetic / "expected.u8",
        args.qualified / "synthetic-same-runtime.laz",
    ):
        shutil.copy2(path, synthetic_dir / path.name)
    display = json.loads((ROOT / "evidence/unet_showcase_media.json").read_text())
    for row in display["files"]:
        source = args.data / "test" / row["source"]
        stem = source.stem
        assert sha(source) == row["source_sha256"]
        assert sha(args.qualified / (stem + "-classified.laz")) == row["output_sha256"]
        for directory, path in [
            ("data/source", source),
            ("data/classified", args.qualified / (stem + "-classified.laz")),
            ("data/labels", args.qualified / (stem + ".u8")),
        ]:
            destination = args.out / directory
            destination.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination / path.name)
    shutil.copytree(ROOT / "LICENSES", args.out / "LICENSES")
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    (args.out / "source").mkdir()
    archived = subprocess.Popen(["git", "archive", source_commit], cwd=ROOT, stdout=subprocess.PIPE)
    subprocess.run(
        ["tar", "-C", str(args.out / "source"), "-xf", "-"], stdin=archived.stdout, check=True
    )
    archived.stdout.close()
    if archived.wait() != 0:
        raise RuntimeError("source archive failed")
    subprocess.run(
        [
            "git",
            "bundle",
            "create",
            str((args.out / "source.bundle").resolve()),
            "codex/byom-native-mit-r0",
        ],
        cwd=ROOT,
        check=True,
    )
    (args.out / "README.txt").write_text(
        "LOCAL RELEASE CANDIDATE — NOT PUBLISHED; OWNER APPROVAL REQUIRED\n\n"
        "SynRex Native LiDAR BYOM trained dense U-Net pilot.\n"
        "Original source and newly trained model parameters: MIT, with upstream notices.\n"
        "Real source/classified data and labels: IGN FRACTAL, Etalab-2.0.\n"
        "Source last modified 2025-04-05T09:37:48Z, retrieved 2026-10-26.\n"
        "https://huggingface.co/datasets/IGNF/FRACTAL\n"
        "Original documentation/presentation: CC-BY-4.0 Tuomas Pietilä.\n"
        "NVIDIA TensorRT plan/vendor dependencies retain separate NVIDIA terms.\n"
        "The SM89 x86_64 plan is target-specific; rebuild from ONNX elsewhere.\n"
        "No SDK binaries, private source or restricted pretrained assets included.\n\n"
        "See source/docs/byom/UNET_QUALIFICATION.md for metrics and rare-class failures.\n"
        "The source directory is an exact export of the committed public repository.\n"
        "The source bundle also preserves local checkpoints and complete Git history.\n"
        "Restore with: git clone --branch codex/byom-native-mit-r0 source.bundle history\n"
        "Inspect publication_candidate.json and SHA256 identities before use.\n"
    )
    records = []
    for path in sorted(args.out.rglob("*")):
        if path.is_file():
            name = path.relative_to(args.out).as_posix()
            license_id = (
                "MIT original synthetic fixture"
                if name.startswith("data/synthetic/")
                else "Etalab-2.0 IGN FRACTAL"
                if name.startswith("data/")
                else "MIT original parameters; NVIDIA plan terms; MODEL_NOTICE.txt"
                if name.startswith("models/")
                else "Separate upstream licence texts"
                if name.startswith("LICENSES/")
                else "CC-BY-4.0 original presentation; Etalab-2.0 source data"
                if name.endswith(".png")
                else "Mixed scoped repository licences; see source bundle LICENSE_SCOPE.json"
            )
            records.append(
                {
                    "path": name,
                    "bytes": path.stat().st_size,
                    "sha256": sha(path),
                    "license_scope": license_id,
                }
            )
    result = {
        "status": "PASS",
        "publication": "HELD_FOR_EXPLICIT_OWNER_APPROVAL",
        "source_commit": source_commit,
        "qualified_source_commit": qualification["source_commit"],
        "no_upload_or_push": True,
        "files": records,
    }
    (args.out / "publication_candidate.json").write_text(json.dumps(result, indent=2) + "\n")
    archive = args.out.with_suffix(".tar.gz")
    if archive.exists():
        raise FileExistsError(archive)
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(args.out, arcname=args.out.name)
    print(
        json.dumps(
            {
                "status": "PASS",
                "archive": archive.name,
                "bytes": archive.stat().st_size,
                "sha256": sha(archive),
                "source_commit": source_commit,
                "publication": result["publication"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
