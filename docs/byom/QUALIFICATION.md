# Native BYOM R0 qualification and reproducibility

<!-- Copyright (c) 2026 Tuomas Pietilä. SPDX-License-Identifier: CC-BY-4.0. Original expression only; upstream material and facts excluded. -->

Date: 2026-10-26. Developer/project lead: Tuomas Pietilä; AI-assisted with OpenAI Codex.
**Later physical milestone:** [RTX4090 qualification](RTX4090_QUALIFICATION.md) now passes the
complete 50,013-point native GPU pipeline and full target suite. The CPU-checkpoint report below
is retained as the `eaf3ce2` milestone; its NOT_RUN GPU rows describe that earlier checkpoint.
Physical AGX qualification still remains NOT_RUN. Nothing is pushed or published.

**Second genuine model:** [trained U-Net qualification](UNET_QUALIFICATION.md)
passes on 2,586,115 held-out real points, through the same binary as the original
Conv3D. It adds FP16 engine inspection, numerical/quality metrics, real PDRF8
preservation and sampled memory. The current full GPU suite passes 62 tests with
one CPU-only skip. These new receipts remain separate from the earlier checkpoints.

This report concerns the new generic BYOM implementation. Historical SegFormer benchmarks,
class IDs, model conversion comparisons and AGX qualification receipts are unchanged and are
not used as evidence of new BYOM GPU execution.

## Executed results at the CPU checkpoint

| Gate | Status | Evidence / boundary |
|---|---|---|
| MIT licensing scope | PASS | Retained public MIT writer/verifier; new generic source; excluded private file identities; separate upstream notices. This is a scoped provenance assessment, not legal adjudication of excluded source. |
| Native CPU build | PASS | GCC 13, C++17, Release on i5radeon x86_64 |
| Native CUDA/TensorRT build | PASS | Full executable compiled/linked, CUDA 13.2.86, SM87 + PTX, TensorRT 10.16 headers/libs; no device execution |
| CTest contract checks | PASS | CPU, ASan/UBSan and GPU-linked builds; feature reduction, normalization, geometry, shapes and source identities |
| CPU integration suite | PASS | 50 passed, 4 physical GPU cases skipped |
| ASan/UBSan integration suite | PASS | 50 passed, 4 physical GPU cases skipped; no reported sanitizer errors |
| GPU-linked CPU integration suite | PASS | 49 passed, 5 skipped (four physical GPU cases plus CPU-only-backend rejection test) |
| Model export/package creation | PASS | Original deterministic Conv3D ONNX, checked graph, SHA256 verification, actual ONNX Runtime CPU nonconstant inference |
| LAZ writer / independent integrity | PASS | Original 19 laz-rs preservation/integrity tests, including 50,013-point chunk boundary |
| CPU and GPU-linked relocation | PASS | Installed executable moved; package/geometry executed; installed ELF has no workspace RUNPATH. NVIDIA libraries supplied externally for GPU-linked CPU checks. |
| TensorRT engine build/initialization | NOT_RUN | No local NVIDIA device; no verified AGX allocation |
| CUDA tensor/inference/postprocessing execution | NOT_RUN | Kernel compilation is not a physical GPU test |
| Native BYOM LAZ-to-classified-LAZ E2E | NOT_RUN | Requires the physical TensorRT and CUDA stages above |
| GPU repeatability / GPU relocation | NOT_RUN | Implemented opt-in integration tests, not executed passes |
| Jetson aarch64 build/ABI | NOT_RUN | Local build is x86_64 with SM87 device code; not an aarch64 product binary |
| Public release readiness | FAIL | Physical GPU/E2E gates incomplete; local candidate only; no owner publication approval requested or used |

Receipts are [CPU qualification](../../evidence/byom_cpu_qualification.json),
[sanitizers](../../evidence/byom_sanitizer_qualification.json) and
[GPU build with CPU tests](../../evidence/byom_gpu_build_cpu_qualification.json).
They record commands, exit codes, sanitized log tails, log hashes and JUnit-derived counts.
Full local logs/JUnit files reside under each ignored build directory's `qualification/`.
[Dependency identities](../../evidence/byom_dependencies.json) include compiler/SDK wheel hashes,
upstream revisions and per-file external source identities. They do not redistribute vendor SDKs.
[Source assessment](../../evidence/byom_source_assessment.json) lists excluded private native
files by hash and records its pristine Git state. [Per-file provenance](../../evidence/byom_provenance.json)
covers the public candidate; its own generated file and the source-assessment output are excluded
from recursive self-hashing. Original benchmark/media artifacts remain untouched.

## Reproduce the executed checks

Follow the [quickstart](QUICKSTART.md) to fetch pinned external sources, configure `build`,
and install both development lock files. Then:

```bash
.venv/bin/python tools/qualify_byom.py --build build \
  --receipt build/qualification-rerun.json
cmake -S . -B build-sanitize -G Ninja -DCMAKE_BUILD_TYPE=Debug \
  -DSYNREX_SANITIZE=ON -DLAZPERF_SOURCE_DIR="$PWD/.deps/laz-perf" \
  -DNLOHMANN_JSON_INCLUDE_DIR="$PWD/.deps/json/single_include"
.venv/bin/python tools/qualify_byom.py --build build-sanitize \
  --receipt build-sanitize/qualification-rerun.json
.venv/bin/ruff check src tests tools examples
.venv/bin/python tools/audit_byom.py
.venv/bin/python tools/release_checks.py --check-only
```

The historical `release_checks.py --history` mode intentionally still describes the original
single-commit/no-remote staging state. Do not use it as a BYOM history assertion. The new
assessment records individual files and local Git checkpoints without changing remote history.

For reproducible GPU *compilation* on i5radeon without system installation, the local check used
NVIDIA CUDA pip wheels extracted under `.deps/`, the public TensorRT v10.16 headers and its
ONNX parser submodule, and the official `tensorrt-cu13-libs==10.16.1.11` distribution. The compiler
components are pinned in the dependency receipt. CMake received the resulting local
`CMAKE_CUDA_COMPILER`, `CUDAToolkit_ROOT`, `TENSORRT_ROOT` and `CMAKE_CUDA_ARCHITECTURES=87`.
This is compilation/link validation, not Jetson-stack or inference qualification. The general
supported build procedure uses the target's existing SDKs; no Jetson packages are installed.

## Physical qualification still required

After an operator verifies a nonconflicting GPU allocation, build on that allocated target
and run the opt-in suite. The runner records the supplied allocation identifier and never
contacts a remote AGX or modifies workloads:

```bash
.venv/bin/python tools/qualify_byom.py --build build-gpu \
  --verified-gpu-allocation OPERATOR_CONFIRMED_ALLOCATION_ID \
  --receipt build-gpu/physical-qualification.json
```

The four cases build and initialize actual TensorRT plans, run actual CUDA preprocessing,
TensorRT inference, CUDA argmax/mapped source-point scatter and classified LAZ output. They compare
against an independent brute-force geometry/ONNX Runtime CPU oracle, then independent laz-rs
decoding. Context-sensitive features expose overlap-policy errors; intensity exercises voxel
means; a different grid and FP16 I/O exercise replacement without recompilation. Repeated outputs
must be byte-identical. Wrong tensor names and target identities must fail without output; installed
and moved executables must reproduce the same classified LAZ. These tests remain `NOT_RUN` here.

The supported-input limitations in the [adapter](ADAPTER.md) and quickstart are release contract
limits. No performance, scientific classification accuracy, arbitrary ONNX, streaming or
autopilot-coexistence claim is made. Completing physical qualification may reveal issues that
compile/CPU checks cannot detect. The candidate is not a completed GPU-qualified release.
