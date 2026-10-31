# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tuomas Pietilä
# Developer and project lead: Tuomas Pietilä.
# AI-assisted software engineering, including OpenAI Codex.
"""Local review checks: file/URL hygiene, SPDX inventory, benchmarks, media and Git objects.

This is a scoped pattern/inventory check, not legal clearance or an exhaustive secret detector.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from importlib import metadata
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
IGNORE = {".git", ".deps", ".venv", ".cache", ".pytest_cache", ".ruff_cache", "__pycache__"}
GENERATED = {"evidence/release_checks.json", "evidence/provenance.json"}
SECRET_PATTERNS = {
    "token": re.compile(r"gh[pousr]_[A-Za-z0-9_]{30,}"),
    "github_pat": re.compile(r"github_pat_[A-Za-z0-9_]{30,}"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
    "aws_key": re.compile(r"AKIA[A-Z0-9]{16}"),
    "absolute_user_or_mount_path": re.compile(r"/(?:home|mnt|root)/[A-Za-z0-9_.-]+"),
    "credential_assignment": re.compile(
        r"(?i)(?:password|api_key|access_token)\s*[=:]\s*['\"][A-Za-z0-9+/_.-]{12,}['\"]"
    ),
}
FORBIDDEN = {".laz", ".las", ".u8", ".pth", ".pt", ".onnx", ".engine", ".srxmodel"}


def run(*args: str) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def files() -> list[Path]:
    # Git inventory bounds traversal even when external build trees/caches are present.
    # Include new candidates during local review, while honoring repository ignores.
    names = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT
    ).decode().split("\0")
    return sorted(
        ROOT / name
        for name in set(names)
        if name
        and (ROOT / name).is_file()
        and not (ROOT / name).is_symlink()
        and not any(
            part in IGNORE or part.startswith("build") or part == "archive"
            for part in Path(name).parts
        )
    )


def findings(data: bytes) -> list[str]:
    if b"\0" in data:
        return []
    text = data.decode("utf-8", errors="replace")
    return [name for name, regex in SECRET_PATTERNS.items() if regex.search(text)]


def origin(path: str) -> tuple[str, str]:
    scope = json.loads((ROOT / "LICENSE_SCOPE.json").read_text())
    for category in ["source", "documentation", "original_procedural_media"]:
        if path in scope[category]["files"]:
            return (
                "Tuomas Pietilä-led original work with OpenAI Codex; see LICENSE_SCOPE.json"
                if category != "original_procedural_media"
                else "Accepted original R1 procedural media; GIF is a documented excerpt",
                scope[category]["license"] + "; original expression only; upstream/facts excluded",
            )
    if path in {"LICENSES/MIT.txt", "LICENSES/CC-BY-4.0.txt"}:
        return (
            "Standard license text; MIT notice personalized on owner confirmation",
            "active for explicit LICENSE_SCOPE.json scopes; no personal claim over license text",
        )
    if path == "LICENSES/LicenseRef-Review-Only.txt":
        return (
            "Preserved historical staging notice",
            "superseded only for explicitly licensed original scopes",
        )
    if path.startswith("LICENSES/"):
        return (
            "Unmodified upstream permission/license notice; see THIRD_PARTY_NOTICES.md",
            "upstream",
        )
    if path.startswith("evidence/benchmark"):
        return (
            "Sanitized factual transcription of accepted historical device evidence",
            "factual measurements; no personal copyright claimed",
        )
    if path.startswith(("evidence/model_lineage", "evidence/numerical_lineage")):
        return (
            "Sanitized factual provenance/numerical summaries; no model artifacts",
            "factual records; no personal copyright claimed",
        )
    if path.startswith("evidence/source_review"):
        return (
            "Historical source audit; original staging state superseded by license_activation.json",
            "factual identities; no personal copyright claimed; excluded source not licensed",
        )
    return (
        "Tuomas Pietilä-led showcase work with OpenAI Codex; see LICENSE_SCOPE.json",
        "scope metadata/factual audit records; grants restricted to LICENSE_SCOPE.json originals",
    )


def check(
    history: bool, check_only: bool, dependency_locks: list[Path] | None = None
) -> dict:
    selected = files()
    hits = {str(p.relative_to(ROOT)): findings(p.read_bytes()) for p in selected}
    hits = {name: rules for name, rules in hits.items() if rules}
    assert not hits, hits
    for p in selected:
        assert p.suffix.lower() not in FORBIDDEN, p.relative_to(ROOT)
        assert p.stat().st_size < 20 * 1024 * 1024, p.name
    source_files = [
        p
        for p in selected
        if p.suffix in {".py", ".cpp", ".cu", ".h", ".hpp"} or p.name == "CMakeLists.txt"
    ]
    for p in source_files:
        assert "SPDX-License-Identifier: MIT" in p.read_text(), p.name
        assert "Copyright (c) 2026 Tuomas Pietilä" in p.read_text(), p.name
    broken = []
    for p in selected:
        if p.suffix != ".md":
            continue
        links = re.findall(r"\]\(([^\s)]+)(?:\s+[^)]*)?\)|src=\"([^\"]+)\"", p.read_text())
        for pair in links:
            target = next(x for x in pair if x)
            if urlsplit(target).scheme or target.startswith("#"):
                continue
            relative = unquote(target.split("#")[0])
            if not (p.parent / relative).exists() and not (
                not check_only and relative in GENERATED
            ):
                broken.append({"file": p.name, "target": target})
    assert not broken, broken
    benchmark = json.loads((ROOT / "evidence/benchmark.json").read_text())
    assert [s["points"] for s in benchmark["scenes"]] == [468416, 886928, 1308277, 2126455]
    assert [round(s["wall_seconds"], 6) for s in benchmark["scenes"]] == [
        6.644378,
        6.237315,
        7.280629,
        10.110652,
    ]
    assert sum(s["points"] for s in benchmark["scenes"]) == 4790076
    assert round(sum(s["wall_seconds"] for s in benchmark["scenes"]), 6) == 30.272974
    assert round(benchmark["aggregate_points_per_second"]) == 158229
    assert round(benchmark["scenes"][-1]["points_per_second"]) == 210318
    for name, record in json.loads((ROOT / "evidence/media_manifest.json").read_text()).items():
        assert hashlib.sha256((ROOT / "media" / name).read_bytes()).hexdigest() == record["sha256"]
    dependency_licenses = {}
    locks = dependency_locks or [ROOT / "requirements-dev.lock", ROOT / "requirements-byom.lock"]
    for line in [line for lock in locks for line in lock.read_text().splitlines()]:
        line = line.partition("#")[0].strip()
        if not line:
            continue
        name, version = line.split("==")
        dist = metadata.distribution(name)
        assert dist.version == version
        expression = dist.metadata.get("License-Expression") or dist.metadata.get("License") or ""
        notice_files = [str(p) for p in dist.files or [] if "license" in str(p).lower()]
        assert expression or notice_files, name
        dependency_licenses[name] = {
            "version": version,
            "declared_license": expression[:180],
            "installed_license_notice_present": bool(notice_files),
        }
    report = {
        "status": "PASS",
        "secret_path_scan": {
            "status": "PASS",
            "pattern_findings": hits,
            "scope": "review tree; pattern-based, not exhaustive",
        },
        "relative_markdown_links": {"status": "PASS", "broken": broken},
        "license_inventory": {
            "status": "PASS_OWNER_CONFIRMED_ORIGINAL_SCOPES_UPSTREAM_EXCLUDED",
            "source_files_with_mit_spdx": len(source_files),
            "personal_licenses": "MIT original source / CC BY 4.0 original documentation/media",
            "dependencies": dependency_licenses,
        },
        "benchmark_and_media_manifest_checks": "PASS",
        "forbidden_data_model_extensions": "NONE",
        "history": "Use --history --check-only after the single initial commit; no public remote",
        "tracked_candidate_files": len(selected),
        "inventory_scope": "maintained Git candidates; historical archive excluded",
        "dependency_locks": [lock.name for lock in locks],
    }
    scope = json.loads((ROOT / "LICENSE_SCOPE.json").read_text())
    assert scope["developer"] == scope["project_lead"] == "Tuomas Pietilä"
    assert scope["ownership_confirmation"] == "CONFIRMED_BY_TUOMAS_PIETILA"
    assert scope["active_personal_licenses"] is True
    assert scope["source"]["license"] == "MIT"
    assert scope["documentation"]["license"] == "CC-BY-4.0"
    assert scope["original_procedural_media"]["license"] == "CC-BY-4.0"
    licensed = set()
    for category in ["source", "documentation", "original_procedural_media"]:
        assert scope[category]["state"] == "ACTIVE_OWNER_AUTHORIZED"
        assert all((ROOT / name).is_file() for name in scope[category]["files"])
        assert not licensed.intersection(scope[category]["files"])
        licensed.update(scope[category]["files"])
    assert all(str(p.relative_to(ROOT)) in scope["source"]["files"] for p in source_files)
    for name in scope["documentation"]["files"]:
        assert "SPDX-License-Identifier: CC-BY-4.0" in (ROOT / name).read_text(), name
    assert "Copyright (c) 2026 Tuomas Pietilä" in (ROOT / "LICENSES/MIT.txt").read_text()
    activation = json.loads((ROOT / scope["confirmation_record"]).read_text())
    assert activation["developer"] == "Tuomas Pietilä"
    assert activation["original_source_tests_configuration"] == "MIT"
    assert activation["original_documentation_diagrams_video"] == "CC-BY-4.0"
    report["personal_attribution"] = {
        "status": "PASS",
        "developer_and_project_lead": "Tuomas Pietilä",
        "ai_assistance": "OpenAI Codex",
        "license_scope_confirmation": scope["ownership_confirmation"],
    }
    if history:
        assert run("git", "rev-list", "--count", "--all") == "1"
        assert run("git", "remote") == ""
        assert run("git", "status", "--porcelain") == ""
        fsck = subprocess.run(
            ["git", "fsck", "--full", "--strict", "--no-reflogs"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        assert fsck.returncode == 0 and not fsck.stderr and not fsck.stdout
        objects = run(
            "git",
            "cat-file",
            "--batch-all-objects",
            "--batch-check=%(objectname) %(objecttype) %(objectsize)",
        ).splitlines()
        object_findings = {}
        for record in objects:
            oid, kind, _ = record.split()
            content = subprocess.check_output(["git", "cat-file", kind, oid], cwd=ROOT)
            rules = findings(content)
            if rules:
                object_findings[oid] = rules
        assert not object_findings, object_findings
        identities = run("git", "show", "-s", "--format=%an%n%ae%n%cn%n%ce", "HEAD").splitlines()
        expected_email = json.loads((ROOT / "evidence/attribution_update.json").read_text())[
            "git_identity"
        ]["email"]
        assert identities == ["Tuomas Pietilä", expected_email, "Tuomas Pietilä", expected_email]
        report["history"] = {
            "status": "PASS",
            "commits": 1,
            "author_and_committer": "Tuomas Pietilä",
            "existing_github_noreply_identity": True,
            "remotes": [],
            "clean_tree": True,
            "all_git_objects_scanned": len(objects),
            "git_fsck": "PASS",
            "head": run("git", "rev-parse", "HEAD"),
        }
    if not check_only:
        provenance = []
        for p in selected:
            name = str(p.relative_to(ROOT))
            if name in GENERATED:
                continue
            source, license_state = origin(name)
            provenance.append(
                {
                    "file": name,
                    "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                    "origin": source,
                    "license_state": license_state,
                }
            )
        (ROOT / "evidence/provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        (ROOT / "evidence/release_checks.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument(
        "--dependency-lock",
        action="append",
        type=Path,
        help="Verify this lock (repeatable); defaults to historical development/BYOM locks",
    )
    args = parser.parse_args()
    print(json.dumps(check(args.history, args.check_only, args.dependency_lock), indent=2))
