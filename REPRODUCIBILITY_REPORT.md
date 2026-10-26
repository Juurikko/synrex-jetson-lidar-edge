# Reproducibility report

<!-- Copyright (c) 2026 Tuomas Pietilä. SPDX-License-Identifier: CC-BY-4.0. Original expression only; upstream material and facts excluded. -->

**Developer and project lead: Tuomas Pietilä ([@Juurikko](https://github.com/Juurikko)).**

> **Superseding publication status — 2026-10-26:** [Juurikko/synrex-jetson-lidar-edge](https://github.com/Juurikko/synrex-jetson-lidar-edge) is published and its public visibility is verified. **PUBLIC_GITHUB_RELEASE=PASS · PUBLIC_VISIBILITY=VERIFIED.** Initial public commit: [`4405474`](https://github.com/Juurikko/synrex-jetson-lidar-edge/commit/4405474c73c25b8c19a5c3bf1b2d70fe30ed6d73). The original staging report below is retained as history; its statements about no creation/push, no remote, pending publication or `NOT_RUN` describe that earlier stage and are superseded for this repository's release. Licensing, personal/upstream attribution, noncommercial-reference restrictions and the limited model-free release scope remain unchanged. Evidence files and manifests retain their initial-commit contents.

> **Post-publication check scope:** `tools/release_checks.py --check-only` still checks documentation links, license inventory, benchmark arithmetic, media hashes and scoped hygiene. Its unchanged `--history` option enforces the historical single-commit/no-remote staging state and is not a post-publication history check. Verify the original commit as an unchanged ancestor, documentation-only descendants and normal remote readback separately. The provenance manifest records the initial public commit, rather than hashes of later documentation revisions.

## Executed local checks

The runnable public-focused preservation utility was built on a Linux x86-64 CPU using GCC 13.3.0, CMake 3.28.3 and Python 3.12. No Jetson, GPU workload, pretrained model, ECLAIR file or paid service was used. The historical hardware benchmark was not rerun.

| Check | Executed outcome |
|---|---|
| C++17 Release build | PASS; project code compiled with `-Wall -Wextra -Wpedantic -Werror` |
| Release synthetic/integrity suite | **19 passed** |
| AddressSanitizer + UndefinedBehaviorSanitizer build and same suite | **19 passed**, leak detection enabled |
| Standalone synthetic demo | PASS; **50,013 points**, native writer and independent laz-rs verification |
| Python Ruff lint/format | PASS |
| C++ clang-format check | PASS |
| Dependency/license inventory and SPDX source headers | PASS; MIT source headers, active owner-confirmed MIT / CC BY 4.0 scopes, unchanged upstream notices; publication withheld |
| Relative Markdown/media links, benchmark arithmetic and media hashes | PASS |
| Pattern-based secret/private-path and forbidden artifact checks | PASS; no restricted data/model extensions or oversized files |
| Clean local Git history/object checks | One initial commit, no remotes; strict fsck and text-pattern scan of Git objects performed after commit |
| R0/R1 accepted publication inputs | SHA256 baseline unchanged across **210 files** |

Machine-readable local evidence is in [synthetic result](evidence/synthetic_demo.json), [source review](evidence/source_review.json), [media manifest](evidence/media_manifest.json), [release checks](evidence/release_checks.json) and [provenance](evidence/provenance.json). Generated reports exclude their own recursive hashes. Post-commit history checks are rerunnable with `tools/release_checks.py --history --check-only`; the exact local commit ID is reported in the untracked external staging receipt.

## Dependencies and deterministic inputs

The codec checkout is laz-perf **3.4.0**, commit `b7bbe26109dc986f42d4fc80b8de3d2b6ca634ce`. The CMake file builds only codec sources; it does not configure upstream tests or download their datasets. The dependency remains external and retains its original license. Two upstream notice files retain their original trailing whitespace; narrowly scoped `.gitattributes` exceptions preserve those notice bytes while keeping project-source whitespace checks enabled. Upstream headers are treated as system headers so their platform-specific pragmas do not weaken warning-as-error checks on the project source.

Python dependencies, development tools and transitive packages are pinned in `requirements-dev.lock`. The verifier selects laz-rs explicitly; it cannot silently switch to the native writer's codec. Synthetic generation uses seed **20261026**, a fixed creation date, invented coordinate integers and record attributes, and a height-rule label fixture. It has no restricted survey/model input.

Exact compressed output bytes can depend on codec/library versions and input header formatting. Reproduction compares semantic labels, ordered nonclassification bytes and metadata, rather than assuming an arbitrary codec build reproduces a historical model-run whole-file hash. The demo's output/classification/canonical digests are recorded for the tested environment.

## Commands

The [README quickstart](README.md#run-the-synthetic-integrity-demonstration) builds and runs the Release demo/tests. Run sanitizer and formatting checks as follows after installing the pinned development environment:

```bash
cmake -S . -B build-sanitize -DLAZPERF_SOURCE_DIR="$PWD/.deps/laz-perf" \
  -DCMAKE_BUILD_TYPE=Debug -DSYNREX_SANITIZE=ON
cmake --build build-sanitize --target synrex-laz-labels -j 4
SYNREX_TEST_WRITER=build-sanitize/synrex-laz-labels ASAN_OPTIONS=detect_leaks=1 \
  .venv/bin/python -m pytest -q --basetemp=build-sanitize/pytest
.venv/bin/ruff check src tests tools
.venv/bin/ruff format --check src tests tools
.venv/bin/clang-format --dry-run --Werror src/preserve_laz.cpp
.venv/bin/python tools/release_checks.py --check-only
```

The layout above assumes dependencies installed inside this repository; local preparation used an equivalent environment outside the tracked tree. Build/test/generated synthetic files are ignored. The demo refuses overwrite; specify a new `--out` directory for another run.

To rebuild the optional GIF preview from the unchanged accepted MP4, run `.venv/bin/python tools/media_preview.py`. It creates a 9-second, 360 × 450, 6 fps excerpt and updates the GIF record in the media manifest. This is media tooling, not inference. The full R1 MP4 and both accepted diagrams remain unchanged.

## Limits of these checks

The tests establish the staged utility's behavior on controlled synthetic fixtures and selected malformed inputs. They do not prove broad LAZ compatibility, hostile-input parser security, power-loss durability or thread/process locking of mutable inputs. The output commit operation requires same-filesystem hard-link support. No general metadata rewriting, EVLR, waveform or COPC support is claimed.

Local tests do not reproduce the neural classifier, license restricted model/data, recalibrate its accuracy or remeasure Jetson throughput. Secret scans are pattern-based and do not replace human review. License/header checks document provenance and notices; they are not legal clearance. The repository is deliberately held for owner review before any public creation or push.
