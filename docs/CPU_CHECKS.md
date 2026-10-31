<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Tuomas Pietilä -->

# Maintained CPU verification

[CPU contracts](../.github/workflows/cpu.yml) runs on disposable GitHub-hosted
Ubuntu 24.04 workers with GCC 13, CPython 3.11.12 and two build workers. Both
Release and Debug ASan/UBSan configurations build with CUDA disabled. No job
uses the persistent RTX runner, repository secrets, dependency caches or a GPU.
The workflow pins action revisions, Python dependencies and native source commits.
Debug uses `-O1 -g` with both sanitizers, preserving the full snapshot and map
fixtures within the same test deadlines. Run steps use explicit Bash with
`-eo pipefail`, so logging through `tee` cannot convert a failed check into success.
The 262,144-cell historical oversize reproduction and full 65,536-cell admitted
snapshot run in separate required processes, each capped at 120 seconds. Their
combined allowance is therefore 240 seconds. Isolation avoids retaining one
fixture's allocations or sanitizer quarantine in the next; earlier combined-test
timeouts remain failed attempts. This is a test execution allowance, separate
from production latency, capacity and numerical qualification gates.
That process split alone did not resolve the timeout: the admitted fixture still
timed out in attempt four. A focused ASan/UBSan diagnostic and independent review
identified the pinned JSON library's repeated parent-array scan in its DOM
callback parser. Wire JSON now uses strict SAX duplicate/depth admission followed
by ordinary DOM parsing of the same bounded bytes. Regressions retain the legacy
event-depth boundary, decoded-key duplicate rejection and parsed-value parity.
The complete snapshot transport/reconstruction remains required under sanitizers.

| Check | Release | Debug ASan/UBSan |
| --- | --- | --- |
| Clean CMake configure/build, compiler warnings as errors | Required | Required |
| All registered CPU CTests, 180-second limit per test | Required | Required |
| Complete pytest discovery, including export and preservation executables | Required | Required |
| Installed CPU executable/link contract exercised by pytest | Required | Required |
| Maintained release inventory, license notices, relative links and media hashes | Required | — |
| Ruff on changed maintained Python files; clang-format on changed C++ lines | Required | — |

CTest covers native package, geometry, stream and semantic-map/transport contracts.
`checked_process_memory` also checks resident-page accounting at 4 KiB and 64 KiB,
invalid page-size queries and multiplication overflow. The live and sustained
receivers share this conversion; Linux `ru_maxrss` remains a separate KiB quantity.
Some native tests require generated fixtures: `tests/test_streaming_r1.py` invokes
`synrex-export-tests`, and `tests/test_preservation.py` invokes `synrex-laz-labels`.
Running CTest alone therefore does not cover the CPU suite. Performance/profile
executables and opt-in physical GPU tests are outside this CPU qualification.

The complete pytest invocation retains explicit `NOT_RUN` reasons for unallocated
GPU tests and absent genuine R2 recordings. Logs and JUnit XML from both configurations
are uploaded even after failure; skips do not establish GPU or recording qualification.
Each matrix job has a 30-minute limit, serial CTest and bounded BLAS/OpenMP workers.

Use `requirements-cpu-ci.lock` for these checks. Its exact package versions were
selected independently of the preserved historical release locks. Install into an
isolated environment locally, or use the disposable workflow interpreter. Native
sources are laz-perf commit `b7bbe26109dc986f42d4fc80b8de3d2b6ca634ce` and
nlohmann/json commit `9cca280a4d0ccf0c08f47a99aa71d1b0e52f8d03`.

After configuring the CPU build as in the workflow, run from the repository root:

```bash
export SYNREX_TEST_NATIVE_DIR="$PWD/build-cpu-Release"
cmake --build "$SYNREX_TEST_NATIVE_DIR" --parallel 2
ctest --test-dir "$SYNREX_TEST_NATIVE_DIR" --output-on-failure --timeout 180
python -m pytest -q -ra --basetemp="$SYNREX_TEST_NATIVE_DIR/pytest" \
  --junitxml="$SYNREX_TEST_NATIVE_DIR/pytest.xml"
python tools/release_checks.py --check-only --dependency-lock requirements-cpu-ci.lock
python tools/check_cpu_style.py --base HEAD
```

`SYNREX_TEST_NATIVE_DIR` selects all native test binaries; the historical explicit
`SYNREX_TEST_NATIVE` and `SYNREX_TEST_WRITER` binary overrides retain precedence.
`check_cpu_style.py --base COMMIT` compares the working tree to that commit, including
new files, and excludes historical archives. It checks complete changed Python files
and only changed C++ lines. Native compilation supplies the corresponding static
warning checks. It never rewrites source files.

Release checks use the bounded Git candidate inventory, excluding ignored output,
external dependencies and the historical archive. Archived benchmark/media facts
remain verified through their explicit manifests. `--check-only` writes no receipts
into that archive. The preserved `--history` option asserts the original single-commit
staging history and is unsuitable for this maintained branch.
