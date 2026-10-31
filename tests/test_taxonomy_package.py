# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
"""CPU package envelopes only; synthetic artifacts establish no inference evidence."""

import copy
import hashlib
import json
import os
import subprocess

import pytest

from test_byom import NATIVE, ROOT, SPEC, package


def run(*arguments, ok=True):
    result = subprocess.run(
        [str(NATIVE), *map(str, arguments)], capture_output=True, text=True, timeout=15
    )
    assert (result.returncode == 0) == ok, result.stdout + result.stderr
    return result


def digest(data):
    return hashlib.sha256(data).hexdigest()


def taxonomy_file(tmp_path, mapping, name="taxonomy.json"):
    value = {
        "schema": "synrex.taxonomy.v1",
        "taxonomy_id": "example.cpu.surfaces.v1",
        "namespace": "example.cpu",
        "classes": [
            {"semantic_id": i, "key": f"surface_{i}", "name": f"Surface {i}", "las_code": code}
            for i, code in enumerate(mapping)
        ],
        "provenance": {"source": "CPU fixture", "author": "SynRex tests", "license": "MIT"},
        "unknown_semantic_id": 255 if len(mapping) == 256 else None,
    }
    path = tmp_path / name
    # Deliberate whitespace/newline identity must survive package creation exactly.
    path.write_bytes((json.dumps(value, indent=2, ensure_ascii=False) + "\n\n").encode())
    return path


def cpu_package(tmp_path, spec, name):
    """Package-validation placeholder, like the existing model-free export fixture."""
    spec_path = tmp_path / f"{name}.json"
    spec_path.write_text(json.dumps(spec))
    artifact = tmp_path / f"{name}.onnx"
    artifact.write_bytes(b"CPU_PACKAGE_ENVELOPE_FIXTURE_ONLY_NO_INFERENCE")
    output = tmp_path / f"{name}.srxmodel"
    run(
        "pack", "--spec", spec_path, "--onnx", artifact,
        "--license", ROOT / "LICENSES/MIT.txt", "--out", output,
    )
    return output, spec_path, artifact


def v3_package(tmp_path, mapping=None):
    mapping = [0, 2, 5, 6] if mapping is None else mapping
    spec = copy.deepcopy(SPEC)
    spec["las_classes"] = mapping
    spec["output"]["shape"][1] = len(mapping)
    old, spec_path, model = cpu_package(tmp_path, spec, "legacy")
    taxonomy = taxonomy_file(tmp_path, mapping)
    output = tmp_path / "explicit.srxmodel"
    run(
        "pack", "--spec", spec_path, "--onnx", model, "--license", ROOT / "LICENSES/MIT.txt",
        "--taxonomy", taxonomy, "--out", output,
    )
    return output, taxonomy, old


def package_hashes(root):
    return {path.name: digest(path.read_bytes()) for path in root.iterdir()}


@pytest.mark.parametrize("mapping", [[255, 255], [0, 2, 5, 6], [42] * 7, list(range(256))])
def test_explicit_v3_preserves_exact_taxonomy_and_generic_dense_contract(tmp_path, mapping):
    model, taxonomy, old = v3_package(tmp_path, mapping)
    manifest = json.loads((model / "manifest.json").read_text())
    assert set(manifest) == {"schema", "contract", "artifacts", "target"}
    assert manifest["schema"] == "synrex.srxmodel.v3"
    assert manifest["contract"]["adapter"] == "DENSE_VOXEL_SEGMENTATION_V1"
    assert manifest["contract"]["las_classes"] == mapping
    assert manifest["contract"] == json.loads((old / "manifest.json").read_text())["contract"]
    assert set(manifest["artifacts"]) == {"license", "onnx", "taxonomy"}
    assert manifest["artifacts"]["taxonomy"] == {
        "path": "taxonomy.json", "sha256": digest(taxonomy.read_bytes())
    }
    assert (model / "taxonomy.json").read_bytes() == taxonomy.read_bytes()
    assert json.loads(run("inspect", "--model", model).stdout) == manifest
    assert manifest["target"] is None


def test_v3_pack_never_replaces_existing_package(tmp_path):
    model, taxonomy, _ = v3_package(tmp_path)
    before = package_hashes(model)
    result = run(
        "pack", "--spec", tmp_path / "legacy.json", "--onnx", tmp_path / "legacy.onnx",
        "--license", ROOT / "LICENSES/MIT.txt", "--taxonomy", taxonomy, "--out", model,
        ok=False,
    )
    assert "refusing package replacement" in result.stderr
    assert package_hashes(model) == before
    assert not list(tmp_path.glob(".srxmodel-*"))


def test_v2_generic_file_package_keeps_its_envelope_and_forbids_extra_taxonomy(tmp_path):
    spec = copy.deepcopy(SPEC)
    spec["las_classes"] = [0, 2, 5, 6]
    spec["output"]["shape"][1] = 4
    model, _, _ = cpu_package(tmp_path, spec, "legacy")
    manifest_path = model / "manifest.json"
    original = manifest_path.read_bytes()
    manifest = json.loads(original)
    assert manifest["schema"] == "synrex.srxmodel.v2"
    assert set(manifest["artifacts"]) == {"license", "onnx"}
    assert set(package_hashes(model)) == {"manifest.json", "MODEL_LICENSE.txt", "model.onnx"}
    run("inspect", "--model", model)
    assert manifest_path.read_bytes() == original
    taxonomy = taxonomy_file(tmp_path, [0, 2, 5, 6])
    (model / "taxonomy.json").write_bytes(taxonomy.read_bytes())
    run("inspect", "--model", model, ok=False)
    manifest["artifacts"]["taxonomy"] = {
        "path": "taxonomy.json", "sha256": digest(taxonomy.read_bytes())
    }
    manifest_path.write_text(json.dumps(manifest))
    result = run("inspect", "--model", model, ok=False)
    assert "unexpected/missing package artifacts" in result.stderr


@pytest.mark.parametrize("kind", ["onnx", "engine"])
def test_explicit_repack_preserves_source_snapshot_and_engine_target_without_gpu(tmp_path, kind):
    source, _, _ = package(tmp_path)
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if kind == "engine":
        # Structurally valid CPU envelope; payload is never deserialized or executed.
        (source / "model.onnx").unlink()
        engine = b"CPU_ENGINE_ENVELOPE_FIXTURE_ONLY"
        (source / "model.engine").write_bytes(engine)
        del manifest["artifacts"]["onnx"]
        manifest["artifacts"]["engine"] = {"path": "model.engine", "sha256": digest(engine)}
        manifest["target"] = {
            "architecture": "cpu-envelope-fixture",
            "gpu_name": "No physical GPU execution",
            "sm": 89,
            "tensorrt": "fixture-version",
            "cuda_runtime": 13020,
        }
        manifest_path.write_text(json.dumps(manifest))
    run("inspect", "--model", source)
    before = package_hashes(source)
    taxonomy = taxonomy_file(tmp_path, manifest["contract"]["las_classes"])
    output = tmp_path / "repacked.srxmodel"
    run("repack", "--model", source, "--taxonomy", taxonomy, "--out", output)
    actual = json.loads(run("inspect", "--model", output).stdout)
    assert actual["schema"] == "synrex.srxmodel.v3"
    assert actual["target"] == manifest["target"]
    assert actual["contract"] == manifest["contract"]
    assert actual["artifacts"][kind] == manifest["artifacts"][kind]
    assert actual["artifacts"]["license"] == manifest["artifacts"]["license"]
    assert (output / f"model.{kind}").read_bytes() == (source / f"model.{kind}").read_bytes()
    assert (output / "taxonomy.json").read_bytes() == taxonomy.read_bytes()
    assert package_hashes(source) == before
    assert not list(tmp_path.glob(".srxmodel-*"))


def test_repack_changed_meaning_at_same_id_k_projection_creates_new_identity(tmp_path):
    source, taxonomy, _ = v3_package(tmp_path, [42] * 7)
    before = package_hashes(source)
    changed = json.loads(taxonomy.read_bytes())
    changed["classes"][0]["name"] = "Changed meaning"
    taxonomy.write_text(json.dumps(changed))
    output = tmp_path / "changed.srxmodel"
    run("repack", "--model", source, "--taxonomy", taxonomy, "--out", output)
    first = json.loads((source / "manifest.json").read_bytes())
    second = json.loads((output / "manifest.json").read_bytes())
    assert first["contract"] == second["contract"]
    assert first["artifacts"]["taxonomy"]["sha256"] != second["artifacts"]["taxonomy"]["sha256"]
    assert package_hashes(source) == before


@pytest.mark.parametrize("destination", ["existing", "inside", "aliased_inside"])
def test_repack_never_replaces_outputs_or_writes_inside_source(tmp_path, destination):
    source, _, _ = package(tmp_path)
    before = package_hashes(source)
    mapping = json.loads((source / "manifest.json").read_text())["contract"]["las_classes"]
    taxonomy = taxonomy_file(tmp_path, mapping)
    output = tmp_path / "existing"
    if destination == "existing":
        output.mkdir()
        (output / "keep").write_bytes(b"keep")
    elif destination == "inside":
        output = source / "child.srxmodel"
    else:
        alias = tmp_path / "alias"
        alias.symlink_to(source, target_is_directory=True)
        output = alias / "child.srxmodel"
    result = run("repack", "--model", source, "--taxonomy", taxonomy, "--out", output, ok=False)
    assert "CUDA/TensorRT" not in result.stderr
    assert package_hashes(source) == before
    if destination == "existing":
        assert (output / "keep").read_bytes() == b"keep"
    else:
        assert "outside source package" in result.stderr
        assert not output.exists()


@pytest.mark.parametrize(
    "fault",
    ["schema", "channels", "projection", "hash", "symlink", "fifo", "size", "duplicate",
     "utf8", "path", "missing", "package_version"],
)
@pytest.mark.parametrize("command", ["inspect", "classify"])
def test_invalid_v3_taxonomy_fails_before_gpu_or_output(tmp_path, fault, command):
    model, _, _ = v3_package(tmp_path)
    manifest_path = model / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    artifact = model / "taxonomy.json"
    value = json.loads(artifact.read_bytes())
    if fault == "schema":
        value["schema"] = "synrex.taxonomy.v2"
    elif fault == "channels":
        value["classes"].pop()
    elif fault == "projection":
        value["classes"][0]["las_code"] = 255
    if fault in {"schema", "channels", "projection"}:
        artifact.write_text(json.dumps(value))
    elif fault == "hash":
        artifact.write_bytes(artifact.read_bytes() + b"\n")
    elif fault == "symlink":
        external = tmp_path / "external.json"
        artifact.rename(external)
        artifact.symlink_to(external)
    elif fault == "fifo":
        artifact.unlink()
        os.mkfifo(artifact)
    elif fault == "size":
        artifact.write_bytes(b" " * 65537)
    elif fault == "duplicate":
        artifact.write_bytes(artifact.read_bytes().replace(b'"schema":', b'"schema":0,"schema":'))
    elif fault == "utf8":
        artifact.write_bytes(artifact.read_bytes().replace(b"Surface 0", b"\xc3\x28"))
    elif fault == "path":
        manifest["artifacts"]["taxonomy"]["path"] = "../taxonomy.json"
    elif fault == "missing":
        del manifest["artifacts"]["taxonomy"]
    elif fault == "package_version":
        manifest["schema"] = "synrex.srxmodel.v4"
    if fault not in {"hash", "symlink", "fifo", "path", "missing", "package_version"}:
        manifest["artifacts"]["taxonomy"]["sha256"] = digest(artifact.read_bytes())
    manifest_path.write_text(json.dumps(manifest))
    output = tmp_path / "must-not-exist.laz"
    arguments = [command, "--model", model]
    if command == "classify":
        arguments += ["--input", tmp_path / "absent.laz", "--out", output]
    result = run(*arguments, ok=False)
    assert result.returncode == 1
    assert json.loads(result.stderr)["status"] == "FAIL"
    expected = {
        "schema": "unsupported taxonomy schema",
        "channels": "taxonomy/output channel mismatch",
        "projection": "taxonomy LAS projection mismatch",
        "hash": "taxonomy exact artifact digest mismatch",
        "symlink": "taxonomy cannot open regular artifact",
        "fifo": "taxonomy artifact must be regular",
        "size": "taxonomy artifact size",
        "duplicate": "taxonomy duplicate JSON key",
        "utf8": "UTF-8",
        "path": "taxonomy fixed package basename required",
        "missing": "unexpected/missing package artifacts",
        "package_version": "unsupported model package version",
    }[fault]
    assert expected in result.stderr, result.stderr
    assert "CUDA/TensorRT" not in result.stderr
    assert "absent.laz" not in result.stderr
    assert not output.exists()
    assert not list(tmp_path.glob(".srxmodel-*"))


@pytest.mark.parametrize("fault", ["symlink", "fifo", "size", "duplicate", "depth"])
def test_manifest_admission_is_bounded_regular_and_strict(tmp_path, fault):
    model, _, _ = package(tmp_path)
    manifest = model / "manifest.json"
    if fault == "symlink":
        external = tmp_path / "external-manifest.json"
        manifest.rename(external)
        manifest.symlink_to(external)
    elif fault == "fifo":
        manifest.unlink()
        os.mkfifo(manifest)
    elif fault == "size":
        manifest.write_bytes(b" " * (1024 * 1024 + 1))
    elif fault == "duplicate":
        manifest.write_text(manifest.read_text().replace('"schema":', '"schema":0,"schema":'))
    else:
        manifest.write_text("[" * 17 + "0" + "]" * 17)
    result = run("inspect", "--model", model, ok=False)
    assert result.returncode == 1
    expected = {
        "symlink": "cannot open regular non-symlink package manifest",
        "fifo": "package manifest must be regular",
        "size": "package manifest size",
        "duplicate": "duplicate JSON key",
        "depth": "package manifest nesting limit",
    }[fault]
    assert expected in result.stderr, result.stderr
    assert "CUDA/TensorRT" not in result.stderr


def test_exact_manifest_byte_limit_preserves_parsed_envelope(tmp_path):
    model, _, _ = package(tmp_path)
    manifest = model / "manifest.json"
    raw = manifest.read_bytes()
    expected = json.loads(raw)
    manifest.write_bytes(raw + b" " * (1024 * 1024 - len(raw)))
    assert json.loads(run("inspect", "--model", model).stdout) == expected


def test_engine_pack_invalid_taxonomy_rejects_before_target_query(tmp_path):
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps(SPEC))
    taxonomy = taxonomy_file(tmp_path, [42, 42])  # Wrong K for the existing dense fixture.
    output = tmp_path / "must-not-exist.srxmodel"
    result = run(
        "pack", "--spec", spec, "--engine", tmp_path / "absent.engine",
        "--license", tmp_path / "absent-license", "--taxonomy", taxonomy, "--out", output,
        ok=False,
    )
    assert "taxonomy/output channel mismatch" in result.stderr
    assert "CUDA/TensorRT" not in result.stderr
    assert not output.exists()
    assert not list(tmp_path.glob(".srxmodel-*"))


@pytest.mark.parametrize("kind", ["onnx", "engine"])
@pytest.mark.parametrize("role", ["model", "license"])
@pytest.mark.parametrize("fault", ["symlink", "fifo", "size", "empty"])
def test_new_pack_bounds_source_snapshots_before_any_engine_target_query(
    tmp_path, kind, role, fault
):
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps(SPEC))
    taxonomy = taxonomy_file(tmp_path, SPEC["las_classes"])
    model = tmp_path / f"model.{kind}"
    license_path = tmp_path / "LICENSE.txt"
    model.write_bytes(b"CPU_PACKAGE_SNAPSHOT_FIXTURE_ONLY_NO_INFERENCE")
    license_path.write_bytes(b"CPU fixture license")
    victim = model if role == "model" else license_path
    if fault == "symlink":
        external = tmp_path / "external"
        victim.rename(external)
        victim.symlink_to(external)
    elif fault == "fifo":
        victim.unlink()
        os.mkfifo(victim)
    elif fault == "size":
        # Sparse metadata-only fixture: no 2 GiB allocation or data generation.
        with victim.open("wb") as out:
            out.truncate(2147483649)
    else:
        victim.write_bytes(b"")
    output = tmp_path / "must-not-exist.srxmodel"
    result = run(
        "pack", "--spec", spec, f"--{kind}", model, "--license", license_path,
        "--taxonomy", taxonomy, "--out", output, ok=False,
    )
    expected = {
        "symlink": "cannot open regular non-symlink source artifact",
        "fifo": "source artifact must be regular",
        "size": "artifact size outside supported range",
        "empty": "artifact size outside supported range",
    }[fault]
    assert expected in result.stderr, result.stderr
    assert "CUDA/TensorRT" not in result.stderr
    assert not output.exists()
    assert not list(tmp_path.glob(".srxmodel-*"))
