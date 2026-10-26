# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
"""Export bound native FRACTAL examples to independently verified QGIS COPC views.

Runs external PDAL and QGIS Python; never invokes or changes neural inference.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import laspy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from verify_semantics import digest, verify_semantics  # noqa: E402


def windows_path(path):
    return subprocess.check_output(["wslpath", "-w", str(path.resolve())], text=True).strip()


def external(command, arguments, windows, work):
    """Use argv on POSIX; a generated, quoted batch file for WSL Windows tools."""
    if not windows:
        subprocess.run([str(command), *map(str, arguments)], check=True)
        return
    # Batch quoting cannot safely represent these metacharacters. Reject, never interpolate.
    values = [str(command), *map(str, arguments)]
    if any(any(c in v for c in '\r\n"%!&|<>^') for v in values):
        raise ValueError("Windows tool paths/arguments contain unsupported shell characters")
    wrapper = work / "external-tool.cmd"
    environment = ""
    if str(command).lower().endswith("\\pdal.exe"):
        env_file = str(command).rsplit("\\", 1)[0] + "\\o4w_env.bat"
        environment = f'if exist "{env_file}" call "{env_file}"\r\n'
    wrapper.write_bytes(
        (
            "@echo off\r\n"
            + environment
            + "call "
            + " ".join('"' + v + '"' for v in values)
            + "\r\nexit /b %errorlevel%\r\n"
        ).encode("utf-8")
    )
    cmd = (
        shutil.which("cmd.exe")
        or subprocess.check_output(
            ["wslpath", "-u", r"C:\Windows\System32\cmd.exe"], text=True
        ).strip()
    )
    subprocess.run([cmd, "/c", windows_path(wrapper)], check=True)


def export(inputs, out, pdal, qgis_python, windows=False, fallback_crs=None):
    if out.exists():
        raise FileExistsError(out)
    # Validate every source/model/output/semantic binding before creating an export.
    rows = []
    seen = set()
    manifest_hash = None
    for package in inputs:
        selection = json.loads((package / "selection.json").read_text())
        model = package / "model/unet.srxmodel"
        current_hash = digest(model / "manifest.json")
        if manifest_hash is not None and current_hash != manifest_hash:
            raise ValueError("mixed model identities")
        manifest_hash = current_hash
        for row in selection["files"]:
            name = row["source"]
            if Path(name).name != name or name in seen:
                raise ValueError("unsafe or duplicate source name")
            seen.add(name)
            stem = Path(name).stem
            source = package / "data/source" / name
            classified = package / "data/classified" / (stem + "-classified.laz")
            sidecar = package / "data/semantics" / (stem + "-semantics.json")
            binding = package / "data/semantics" / (stem + "-binding.json")
            verification = verify_semantics(
                source,
                classified,
                sidecar,
                model,
                binding,
                ROOT / "examples/unet/semantic_classes.json",
            )
            rows.append((package, row, verification))
    if not rows:
        raise ValueError("no input scenes")
    out.mkdir(parents=True)
    work = out / ".work"
    work.mkdir()
    records = []
    try:
        for package, row, verification in rows:
            stem = Path(row["source"]).stem
            for directory, filename in [
                ("source", row["source"]),
                ("classified", stem + "-classified.laz"),
                ("semantics", stem + "-semantics.json"),
                ("semantics", stem + "-binding.json"),
            ]:
                destination = out / "data" / directory / filename
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(package / "data" / directory / filename, destination)
            classified = out / "data/classified" / (stem + "-classified.laz")
            sidecar = out / "data/semantics" / (stem + "-semantics.json")
            staging = work / (stem + ".las")
            view = out / "views" / (stem + ".copc.laz")
            view.parent.mkdir(exist_ok=True)
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tools/qgis_copc_view.py"),
                    "--classified",
                    str(classified),
                    "--sidecar",
                    str(sidecar),
                    "--staging",
                    str(staging),
                    "--receipt",
                    str(work / (stem + "-stage.json")),
                ],
                check=True,
            )
            tool_path = windows_path if windows else lambda p: str(p.resolve())
            pipeline = work / "pipeline.json"
            pipeline.write_text(
                json.dumps(
                    [
                        {"type": "readers.las", "filename": tool_path(staging)},
                        {
                            "type": "writers.copc",
                            "filename": tool_path(view),
                            "forward": "all",
                            "extra_dims": "all",
                        },
                    ]
                )
            )
            external(pdal, ["pipeline", tool_path(pipeline)], windows, work)
            proof = view.with_name(stem + "-verified.json")
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
                    str(proof),
                ],
                check=True,
            )
            with laspy.open(classified, laz_backend=laspy.LazBackend.Lazrs) as reader:
                original_wkt = next(
                    (
                        v.string
                        for v in reader.header.vlrs
                        if v.user_id == "LASF_Projection" and v.record_id == 2112
                    ),
                    None,
                )
            records.append(
                dict(
                    source=row["source"],
                    points=verification["points"],
                    view_filename="views/" + view.name,
                    original_wkt=original_wkt,
                    native_verification=verification,
                    copc_verification=json.loads(proof.read_text()),
                )
            )
            staging.unlink()
        shutil.copytree(inputs[0] / "model", out / "model")
        shutil.copytree(ROOT / "LICENSES", out / "LICENSES")
        for name in ["semantic_classes.json", "fractal-classification-r3.qml", "MODEL_NOTICE.txt"]:
            shutil.copy2(ROOT / "examples/unet" / name, out / name)
        selection = dict(
            schema="synrex.qgis_export.v1",
            files=records,
            model_manifest_sha256=manifest_hash,
            fallback_crs=fallback_crs,
        )
        (out / "export-inputs.json").write_text(json.dumps(selection, indent=2) + "\n")
        worker = out / "qgis_export_project.py"
        shutil.copy2(ROOT / "tools/qgis_export_project.py", worker)
        external(qgis_python, [tool_path(worker)], windows, work)
        project_receipt = json.loads((out / "qgis-project.json").read_text())
        if project_receipt["status"] != "PASS":
            raise ValueError("QGIS project validation failed")
        files = [
            {"path": p.relative_to(out).as_posix(), "sha256": digest(p)}
            for p in sorted(out.rglob("*"))
            if p.is_file() and work not in p.parents
        ]
        receipt = dict(
            schema="synrex.qgis_export_receipt.v1",
            status="PASS",
            points=sum(r["points"] for r in records),
            scenes=len(records),
            files=files,
            project=project_receipt,
            inference="Existing frozen native predictions; no inference rerun",
            metadata_scope="Source/native files unchanged; COPC is an indexed derivative",
        )
        (out / "export-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        shutil.rmtree(work)
        return receipt
    except Exception as error:
        (out / "EXPORT_FAILED.json").write_text(json.dumps(dict(status="FAIL", error=str(error))))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pdal", required=True, help="PDAL executable or environment wrapper")
    parser.add_argument(
        "--qgis-python", required=True, help="QGIS-enabled Python executable/wrapper"
    )
    parser.add_argument(
        "--windows-tools", action="store_true", help="WSL to installed Windows tools"
    )
    parser.add_argument("--fallback-crs", help="Explicit viewing-only assignment for missing CRS")
    args = parser.parse_args()
    result = export(
        args.input, args.out, args.pdal, args.qgis_python, args.windows_tools, args.fallback_crs
    )
    print(
        json.dumps(dict(status=result["status"], scenes=result["scenes"], points=result["points"]))
    )


if __name__ == "__main__":
    main()
