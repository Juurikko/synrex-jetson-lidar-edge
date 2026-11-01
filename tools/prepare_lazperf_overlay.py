# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
"""Prepare the checked, build-owned laz-perf decoder overlay without editing upstream."""

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

DEFAULT_MANIFEST = (
    Path(__file__).resolve().parents[1] / "patches/lazperf-b7bbe261/manifest.json"
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def checked_files(root, manifest, phase):
    """Check every copied byte, and reject extra codec files (including symlinks)."""
    expected = set(manifest["files"])
    actual = {
        p.relative_to(root).as_posix()
        for p in (root / "cpp/lazperf").rglob("*")
        if not p.is_dir() or p.is_symlink()
    } | {"COPYING"}
    if actual != expected:
        raise ValueError("laz-perf file set differs from pinned manifest")
    result = {}
    for name, hashes in manifest["files"].items():
        path = root / name
        if path.is_symlink() or any(p.is_symlink() for p in path.parents if p != root.parent):
            raise ValueError(f"symlink in pinned input: {path}")
        data = path.read_bytes()
        if digest(data) != hashes[phase]:
            raise ValueError(f"laz-perf {phase} hash mismatch: {name}")
        result[name] = data
    return result


def apply_patch(root, patch):
    """Apply unified hunks at their exact original line; no offset or fuzzy matching."""
    lines = patch.decode("utf-8").splitlines(keepends=True)
    index = 0
    while index < len(lines):
        if not lines[index].startswith("--- a/"):
            raise ValueError("invalid patch file header")
        name = lines[index][6:].rstrip("\n")
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("unsafe patch path")
        index += 1
        if lines[index] != "+++ b/" + name + "\n":
            raise ValueError("patch must modify the same existing file")
        index += 1
        path = root / name
        original = path.read_text().splitlines(keepends=True)
        output, consumed = [], 0
        while index < len(lines) and lines[index].startswith("@@ "):
            hunk = re.fullmatch(r"@@ -(\d+),(\d+) \+(\d+),(\d+) @@\n", lines[index])
            if hunk is None:
                raise ValueError("invalid patch hunk")
            old_start, old_count, new_start, new_count = map(int, hunk.groups())
            if old_start - 1 < consumed:
                raise ValueError("overlapping patch hunks")
            output.extend(original[consumed:old_start - 1])
            consumed = old_start - 1
            if len(output) != new_start - 1:
                raise ValueError("patch new-line offset mismatch")
            removed = added = 0
            index += 1
            while index < len(lines) and lines[index][:1] in (" ", "+", "-"):
                if lines[index].startswith("--- a/"):
                    break
                kind, content = lines[index][0], lines[index][1:]
                if kind in (" ", "-"):
                    if consumed >= len(original) or original[consumed] != content:
                        raise ValueError(f"patch context mismatch: {name}:{consumed + 1}")
                    consumed += 1
                    removed += 1
                if kind in (" ", "+"):
                    output.append(content)
                    added += 1
                index += 1
            if (removed, added) != (old_count, new_count):
                raise ValueError("patch hunk count mismatch")
        output.extend(original[consumed:])
        path.write_text("".join(output))


def prepare(source, output, manifest_path=DEFAULT_MANIFEST):
    source, output = source.resolve(), output.parent.resolve() / output.name
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("overlay and upstream source must be disjoint")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    patch = (manifest_path.parent / manifest["patch"]).read_bytes()
    if digest(patch) != manifest["patch_sha256"]:
        raise ValueError("overlay patch hash mismatch")
    if (source / ".git").exists():
        commit = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip()
        if commit != manifest["upstream_commit"]:
            raise ValueError("laz-perf upstream commit mismatch")
    # Validate the current external input even when an overlay already exists.
    inputs = checked_files(source, manifest, "before")
    identity = {
        "schema": manifest["schema"], "upstream_commit": manifest["upstream_commit"],
        "manifest_sha256": digest(manifest_bytes), "patch_sha256": digest(patch),
        "helper_sha256": digest(Path(__file__).read_bytes()),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with (output.parent / (output.name + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if output.exists() or output.is_symlink():
            if output.is_symlink():
                raise ValueError("overlay output is a symlink")
            if json.loads((output / "overlay-identity.json").read_bytes()) != identity:
                raise ValueError("existing overlay identity differs; use a fresh binary directory")
            checked_files(output, manifest, "after")
            return output
        staging = Path(tempfile.mkdtemp(prefix=output.name + ".stage-", dir=output.parent))
        try:
            for name, data in inputs.items():
                path = staging / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            apply_patch(staging, patch)
            checked_files(staging, manifest, "after")
            # Retain upstream license/comments and mark the five modified files.
            changed = [n for n, h in manifest["files"].items() if h["before"] != h["after"]]
            (staging / "SYNREX_OVERLAY_NOTICE.txt").write_text(
                "laz-perf: Apache-2.0; upstream copyright notices remain in each file.\n"
                "SynRex decoder arithmetic and memory-stream correction, 2026-10-31.\n"
                "Modified files (see source-controlled patch):\n" + "\n".join(changed) + "\n"
            )
            (staging / "overlay-identity.json").write_text(json.dumps(identity, indent=2) + "\n")
            os.rename(staging, output)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(prepare(args.source, args.output))
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"laz-perf overlay rejected: {error}\n")


if __name__ == "__main__":
    main()
