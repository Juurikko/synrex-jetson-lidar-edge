# SynRex · Native LiDAR classification on Jetson

<!-- Copyright (c) 2026 Tuomas Pietilä. SPDX-License-Identifier: CC-BY-4.0. Original expression only; upstream material and facts excluded. -->

**Developer and project lead: Tuomas Pietilä ([@Juurikko](https://github.com/Juurikko)).**

**Already-georeferenced LAZ → native C++ / CUDA / TensorRT → classified LAZ.**

On a physical **NVIDIA Jetson AGX Orin 64GB**, the demonstrated pipeline processed **2,126,455 points in 10.110652 seconds: 210,318 points/s** on the largest tested file. Across four files, it processed **4,790,076 points in 30.272974 seconds: 158,229 points/s aggregate**. These are one-shot full product executions, including initialization and classified-file output.

[![Watch the accepted 30-second engineering showcase](media/preview.gif)](media/synrex-jetson-lidar-r1.mp4)

**Engineering: Tuomas Pietilä · AI-assisted with OpenAI Codex.** [Presentation credits and accepted-video attribution](media/PERSONAL_CREDITS.md).

[Watch/download the 30-second MP4](media/synrex-jetson-lidar-r1.mp4). 1080 × 1350, 30 fps, silent. The survey geometry is original illustration, **not genuine model predictions**. Green sections describe verified engineering; amber drone sections are future concepts. Some GitHub views link rather than autoplay relative MP4 assets; the GIF supplies an embedded preview. It is a 9-second excerpt at 360px width/6 fps; the full accepted MP4 remains unchanged.

## What has been demonstrated

The native product accepts a compressed, already-georeferenced point-cloud file, builds the model's geometric/voxel representation, performs GPU inference, scatters class IDs back to source points and writes a classified LAZ. It preserves source point order and all other point-record data within the qualified file contract. It does not perform sensor georeferencing.

The deployed product requires **no Python or PyTorch runtime**. It still depends on CUDA, TensorRT, glibc and its bundled native libraries. Its existing trained reference model is **TreeAIBox-derived and noncommercial**; this project does not introduce a new neural architecture or establish commercial rights to those weights.

| Measured boundary | Result |
|---|---:|
| Largest tested file | 2,126,455 points / 10.110652 s |
| Largest-file throughput | 210,318 points/s |
| Four-file total | 4,790,076 points / 30.272974 s |
| Four-file aggregate throughput | 158,229 points/s |

**Scope:** MODE_50W, CUDA 13.2, TensorRT 10.16, FP16; one process per file, no warmups/repetitions, normal OS caching. This is not sustained streaming, an in-flight demonstration, a flight-to-map measurement, a state-of-the-art claim or a world record. See [benchmark protocol and all four results](BENCHMARK.md).

## What this repository contains

This is an engineering writeup **with runnable model-free source**, not a complete downloadable classifier. It contains:

- A substantive C++ LAZ preservation utility that applies supplied uint8 labels while retaining the source header, non-LASzip metadata, padding, point order and all other record bytes.
- An independent **laz-rs** verifier, deterministic synthetic data generator and 19 runnable tests, including negative integrity tests.
- Architecture, numerical-validation findings, qualified benchmark receipts and the accepted R1 media.

The public-focused utility was written for this staging release after reviewing the preservation design. **It is not the frozen writer binary measured in the benchmark.** No neural classifier, CUDA kernels, model-dependent preprocessing implementation, weights, ONNX model, TensorRT engine or restricted dataset is included. Historical hardware throughput cannot be reproduced from this repository alone. [Release boundaries and code provenance](PUBLICATION_RIGHTS.md) explain why.

**Publication status — 2026-10-26:** this repository is public at [Juurikko/synrex-jetson-lidar-edge](https://github.com/Juurikko/synrex-jetson-lidar-edge). **PUBLIC_GITHUB_RELEASE=PASS · PUBLIC_VISIBILITY=VERIFIED.** Initial public commit: [`4405474`](https://github.com/Juurikko/synrex-jetson-lidar-edge/commit/4405474c73c25b8c19a5c3bf1b2d70fe30ed6d73). This dated status supersedes earlier staging-only statements; historical audit reports and evidence are retained. Personal developer/project-lead attribution is Tuomas Pietilä. **MIT** applies to the confirmed personally controlled original source/tests; **CC BY 4.0** applies to the original documentation, diagrams and video. Copyright (c) 2026 Tuomas Pietilä applies only to protectable original expression within that scope; factual records and upstream material are excluded. [License scopes](LICENSE_SCOPE.json). See [personal attribution and licensing](PERSONAL_ATTRIBUTION_AND_LICENSING.md). Upstream dependencies retain their own licenses. See the [historical publication-gate report and superseding status](PUBLICATION_BLOCKERS.md).

## Bring your own model: write verified LAZ outputs

**Already have a classifier?** If your model or algorithm can produce one **ASPRS/LAS-compatible classification byte per original source point**, the runnable, model-independent part of this repository can apply those labels to a supported compressed LAZ file and independently verify its preservation. You do **not** need the private SegFormer engine or a Jetson to run this utility.

**[Bring your own model — exact inputs, commands, integration requirements and limitations](BRING_YOUR_OWN_MODEL.md)**

> Example: a survey or forestry developer produces source-ordered, explicitly mapped ground/vegetation/building classification codes, then calls `synrex-laz-labels input.laz labels.u8 output.laz`. The public C++ tool changes the classification field, not the other decoded point-record bytes within its supported LAS contract. **It does not perform neural inference or convert arbitrary internal class IDs into standard LAS semantics.**

**Public geospatial-data question:** Finland's National Land Survey already offers classified LAZ point clouds. Could national datasets additionally offer **versioned, quality-assessed semantic layers and lightweight derived map products**? See [national semantic point-cloud services: what exists, and a realistic extension](PUBLIC_SEMANTIC_POINT_CLOUDS.md). This is a proposal, not an existing capability demonstrated by this software.

## Run the synthetic integrity demonstration

Prerequisites: Linux, Python 3.12 with venv support, CMake ≥ 3.20, a C++17 compiler and Git. No Jetson, GPU, Python inference framework, pretrained model or real dataset is needed. Dependency downloads are public and free; existing system tools are used without system installation.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.lock
mkdir -p .deps
git clone --branch 3.4.0 --depth 1 https://github.com/hobuinc/laz-perf.git .deps/laz-perf
git -C .deps/laz-perf rev-parse HEAD
# Expected: b7bbe26109dc986f42d4fc80b8de3d2b6ca634ce
cmake -S . -B build -DLAZPERF_SOURCE_DIR="$PWD/.deps/laz-perf" -DCMAKE_BUILD_TYPE=Release
cmake --build build --target synrex-laz-labels -j 4
.venv/bin/python tools/synthetic_demo.py --points 50013
.venv/bin/python -m pytest -q --basetemp=build/pytest
```

The demo creates invented records, assigns labels using a simple height rule, calls the native C++ writer, then independently decodes both files with laz-rs. **The height rule is a test fixture, not a segmentation model.** The demo refuses to overwrite its files: select a new `--out` directory for another run.

For trusted files satisfying the utility contract:

```bash
build/synrex-laz-labels INPUT.laz LABELS.u8 OUTPUT.laz
.venv/bin/python src/verify_laz.py INPUT.laz OUTPUT.laz LABELS.u8
```

`LABELS.u8` must contain exactly one byte per point, in original order. The native utility supports the review contract **compressed LAS 1.4 / PDRF 7 / 42-byte records / no EVLRs or waveform data / no COPC index**. It refuses output replacement. This narrow scope is intentional; it is not a general-purpose LAZ editor. [Architecture and failure handling](ARCHITECTURE.md) and the [reproduction report](REPRODUCIBILITY_REPORT.md) give the practical limits.

## Why data integrity matters

A classification result is more useful if it remains compatible with downstream geospatial tooling. The qualified production outputs were independently decoded using laz-rs rather than relying on the native laz-perf writer to verify itself. The checks compare source-order decompressed records after masking only the classification byte, and separately compare qualified metadata and digests. [Validation](VALIDATION.md) also records known ONNX/FP16 numerical differences; preservation exactness does not imply perfect model accuracy or PyTorch bit parity.

## Applications and what comes next

File-based classification could support survey review, forestry inventories, infrastructure inspection and semantic layers for autonomous-system development. The measured result establishes a file-processing boundary, not qualification of every application or a specific deployment.

The longer-term proposal is onboard UAV classification: transmit compact semantic maps, identify missing or uncertain coverage and selectively revisit targets before landing. **That workflow has not been demonstrated.** It requires streaming integration, georeferencing, bounded latency, confidence calibration, power/thermal qualification and telemetry experiments. [Future UAV work](FUTURE_UAV.md) separates proposals from required evidence.

## Explore the engineering

| Document | Focus |
|---|---|
| [Architecture](ARCHITECTURE.md) | Decode, geometry, occupancy, inference, reconstruction, file output |
| [Benchmark](BENCHMARK.md) | All measured scenes, timing boundary and limitations |
| [Validation](VALIDATION.md) | Independent record/metadata checks and model-conversion deltas |
| [Contributions and method](CONTRIBUTIONS_AND_METHOD.md) | Integration decisions, AI assistance and attribution |
| [Public release audit](PUBLIC_RELEASE_AUDIT.md) | Included/excluded material and quality gates |
| [Repository contents](REPOSITORY_CONTENTS.md) | File-by-file purpose and provenance |

<picture><img src="media/pipeline-portrait.png" alt="Four-stage native LAZ-to-classified-LAZ pipeline on Jetson" width="420"></picture>

Developer and project lead: **Tuomas Pietilä ([@Juurikko](https://github.com/Juurikko))**. Developed using AI-assisted software engineering, including **OpenAI Codex**; human-led architecture, specifications, decisions and verification review are detailed in [Contributions and method](CONTRIBUTIONS_AND_METHOD.md). Existing model research belongs to its respective authors; see [third-party notices](THIRD_PARTY_NOTICES.md). Public repository: [Juurikko/synrex-jetson-lidar-edge](https://github.com/Juurikko/synrex-jetson-lidar-edge).
