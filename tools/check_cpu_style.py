# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
"""Check changed maintained Python files and changed C++ lines without rewriting history."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CPP_SUFFIXES = {".cpp", ".hpp", ".h", ".cu"}
MAINTAINED = {"tools", "src", "native", "tests", "examples"}


def git(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=ROOT, text=True)


def changed_lines(base: str, path: str) -> set[int]:
    diff = git("diff", "--no-ext-diff", "--unified=0", base, "--", path)
    lines = set()
    pattern = r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@"
    for match in re.finditer(pattern, diff, re.MULTILINE):
        first = int(match[1])
        count = int(match[2]) if match[2] is not None else 1
        lines.update(range(first, first + count))
    return lines


def check_cpp(path: str, lines: set[int]) -> list[int]:
    if not lines:
        return []
    result = subprocess.check_output(
        [
            "clang-format", "--output-replacements-xml",
            *[f"--lines={n}:{n}" for n in sorted(lines)], path,
        ],
        cwd=ROOT,
    )
    source = (ROOT / path).read_bytes()
    failures = []
    for replacement in ET.fromstring(result).findall("replacement"):
        offset = int(replacement.attrib["offset"])
        end = offset + int(replacement.attrib["length"])
        first_line = source[:offset].count(b"\n") + 1
        last_line = source[:end].count(b"\n") + 1
        if lines.intersection(range(first_line, last_line + 1)):
            failures.append(first_line)
    return failures


def check(base: str) -> dict:
    # Resolve before using a caller-supplied revision in further Git commands.
    base = git("rev-parse", "--verify", "--end-of-options", f"{base}^{{commit}}").strip()
    names = set(git("diff", "--name-only", "-z", "--diff-filter=ACMR", base, "--").split("\0"))
    untracked = set(git("ls-files", "--others", "--exclude-standard", "-z").split("\0"))
    names.update(untracked)
    selected = sorted(
        name
        for name in names
        if name and Path(name).parts[0] in MAINTAINED and (ROOT / name).is_file()
    )
    python = [name for name in selected if Path(name).suffix == ".py"]
    if python:
        subprocess.run([sys.executable, "-m", "ruff", "check", *python], cwd=ROOT, check=True)
    cpp = [name for name in selected if Path(name).suffix in CPP_SUFFIXES]
    failures = {}
    for path in cpp:
        lines = (
            set(range(1, len((ROOT / path).read_bytes().splitlines()) + 1))
            if path in untracked
            else changed_lines(base, path)
        )
        affected = check_cpp(path, lines)
        if affected:
            failures[path] = affected
    if failures:
        raise RuntimeError(f"clang-format changes needed on modified lines: {failures}")
    return {"status": "PASS", "base": base, "python_files": python, "cpp_files": cpp}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="HEAD", help="Commit before the maintained changes")
    arguments = parser.parse_args()
    print(json.dumps(check(arguments.base), indent=2))
