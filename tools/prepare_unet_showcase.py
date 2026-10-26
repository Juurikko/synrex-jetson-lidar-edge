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
    parser.add_argument(
        "--media-manifest", type=Path, default=ROOT / "evidence/unet_showcase_media.json"
    )
    parser.add_argument("--qgis-direct", type=Path)
    parser.add_argument("--qgis-copc", type=Path)
    args = parser.parse_args()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise ValueError("showcase requires clean committed source, including all style files")
    qualification = json.loads((args.qualified / "qualification.json").read_text())
    if qualification["status"] != "PASS":
        raise ValueError("require successful genuine qualification before showcase preparation")
    if "frozen_model" in qualification:
        assert sha(args.trained / "best.pt") == qualification["frozen_model"]["weights_sha256"]
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
    (args.out / "models/dense_voxel.json").write_text(
        json.dumps(manifest["contract"], indent=2) + "\n"
    )
    taxonomy = ROOT / "examples/unet/semantic_classes.json"
    if (
        taxonomy.exists()
        and [row["output_las_code"] for row in json.loads(taxonomy.read_text())["classes"]]
        == manifest["contract"]["las_classes"]
    ):
        shutil.copy2(taxonomy, args.out / "models/semantic_classes.json")
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
    display = json.loads(args.media_manifest.read_text())
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
        sidecar = args.qualified / (stem + "-semantics.json")
        if sidecar.exists():
            destination = args.out / "data/semantics"
            destination.mkdir(parents=True, exist_ok=True)
            shutil.copy2(sidecar, destination / sidecar.name)
    if (args.qualified / (Path(display["files"][0]["source"]).stem + "-binding.json")).exists():
        for row in display["files"]:
            binding = args.qualified / (Path(row["source"]).stem + "-binding.json")
            envelope = json.loads(binding.read_text())
            stem = Path(row["source"]).stem
            assert envelope["output_sha256"] == sha(args.qualified / (stem + "-classified.laz"))
            assert envelope["semantic_sidecar_sha256"] == sha(
                args.qualified / (stem + "-semantics.json")
            )
            assert envelope["model_manifest_sha256"] == sha(package / "manifest.json")
            assert envelope["taxonomy_sha256"] == sha(taxonomy)
            shutil.copy2(binding, args.out / "data/semantics" / binding.name)
    shutil.copy2(ROOT / "examples/unet/fractal-classification.qml", args.out / "models")
    for kind, directory in [("direct", args.qgis_direct), ("copc", args.qgis_copc)]:
        if directory is None:
            continue
        receipt = json.loads((directory / "qgis-qualification.json").read_text())
        assert receipt["status"] == "PASS"
        assert receipt["harness_sha256"] == sha(ROOT / "tools/qualify_qgis.py")
        assert receipt["project_sha256"] == sha(directory / "synrex-fractal-r2.qgz")
        destination = args.out / "qgis" / kind
        destination.mkdir(parents=True)
        for name in ["data", "model", "views"]:
            if (directory / name).exists():
                shutil.copytree(
                    directory / name, destination / name, ignore=shutil.ignore_patterns("ept_*")
                )
        for name in [
            "selection.json",
            "fractal-classification-r3.qml",
            "semantic_classes.json",
            "MODEL_NOTICE.txt",
            "qgis-qualification.json",
            "synrex-fractal-r2.qgz",
        ]:
            shutil.copy2(directory / name, destination / name)
        shutil.copy2(ROOT / "tools/qualify_qgis.py", destination / "qualify_qgis.py")
        for row in receipt["files"]:
            screenshot = directory / row["screenshot"]
            assert sha(screenshot) == row["screenshot_sha256"]
            shutil.copy2(screenshot, destination / screenshot.name)
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
        "See source/docs/byom/UNET_R2_RELEASE.md for frozen metrics and limitations.\n"
        "R3 actual QGIS projects/captures, where included: qgis/direct and qgis/copc.\n"
        "See source/docs/byom/QGIS_R3.md for tested scope and direct48-byte workaround.\n"
        "Physical AGX BYOM execution remains NOT_RUN while Spatial Lab owns its locks.\n"
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
                "Etalab-2.0 taxonomy; MIT original metadata layout"
                if name == "models/semantic_classes.json"
                else "MIT original style; IGN semantic names retain Etalab attribution"
                if name.endswith(".qml")
                else "MIT original synthetic fixture"
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
            if name.startswith("qgis/"):
                license_id = (
                    "Etalab-2.0 survey; CC-BY-4.0 original presentation; QGIS/Qt UI upstream rights"
                    if name.endswith(".png")
                    else "MIT original parameters; NVIDIA plan terms; MODEL_NOTICE.txt"
                    if name.endswith("model.engine")
                    else "MIT original helper/style; separate installed QGIS terms"
                    if name.endswith((".py", ".qml"))
                    else "Etalab-2.0 IGN survey/taxonomy; MIT original project/binding layout"
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
