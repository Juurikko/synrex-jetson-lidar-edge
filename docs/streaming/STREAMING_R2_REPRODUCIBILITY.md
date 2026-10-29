# Reproduce Streaming R2 RC1

<!-- Copyright (c) 2026 Tuomas Pietilä. SPDX-License-Identifier: CC-BY-4.0. -->

Use the SHA256 from the independent candidate receipt to verify the archive before
extracting into an empty directory. Set `SYNREX_RELEASE` to its absolute extracted root.
No private checkout or internal endpoint is required. Large assets remain outside Git.

## Fresh CPU source and public dependencies

Prerequisites: Linux, Git, Python 3.12/venv, CMake ≥3.20, Ninja, C++17 compiler and
OpenSSL 3 development files. These public pins were cloned afresh during R2 reconstruction.

```bash
export SYNREX_RELEASE="$PWD/streaming-r2-rc1"
git clone --branch codex/byom-streaming-r0 "$SYNREX_RELEASE/source.bundle" streaming-source
cd streaming-source
git checkout --detach "$(python3 -c 'import json,os; print(json.load(open(os.environ["SYNREX_RELEASE"]+"/RELEASE_MANIFEST.json"))["source_commit"])')"
git clone --depth 1 --branch 3.4.0 https://github.com/hobuinc/laz-perf.git .deps/laz-perf
git clone --depth 1 --branch v3.11.3 https://github.com/nlohmann/json.git .deps/json
git -C .deps/laz-perf rev-parse HEAD
# b7bbe26109dc986f42d4fc80b8de3d2b6ca634ce
git -C .deps/json rev-parse HEAD
# 9cca280a4d0ccf0c08f47a99aa71d1b0e52f8d03
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.lock -r requirements-byom.lock
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DLAZPERF_SOURCE_DIR="$PWD/.deps/laz-perf" \
  -DNLOHMANN_JSON_INCLUDE_DIR="$PWD/.deps/json/single_include"
cmake --build build -j4
ctest --test-dir build --output-on-failure
SYNREX_R2_ASSETS="$SYNREX_RELEASE" .venv/bin/python -m pytest -q --basetemp=build/pytest
.venv/bin/ruff check .
.venv/bin/python tools/release_checks.py --check-only
.venv/bin/python tools/streaming_r2_release.py verify --out "$SYNREX_RELEASE"
```

CPU builds never simulate neural inference. The original 90 CPU regressions pass; five
GPU tests skip without a verified allocation. Six additional adversarial visual-provenance
tests run with the genuine package supplied, even without CUDA. They rebind hashes after
mutating semantic labels, source IDs, timestamps, eviction victims, missing deltas or
semantic-reuse counters; independent validation must reject each.

```bash
cmake -S . -B build-sanitize -G Ninja -DCMAKE_BUILD_TYPE=Debug -DSYNREX_SANITIZE=ON \
  -DLAZPERF_SOURCE_DIR="$PWD/.deps/laz-perf" \
  -DNLOHMANN_JSON_INCLUDE_DIR="$PWD/.deps/json/single_include"
cmake --build build-sanitize -j4
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
  ctest --test-dir build-sanitize --output-on-failure
```

## Native GPU correctness and transport

First establish an available physical allocation; do not stop unrelated jobs. R2 reused
the installed RTX 4090 / SM89 environment: driver 610.88, CUDA 13.2.86 / runtime 13020,
TensorRT 10.16.1, Linux/WSL2, Ryzen 7700X. No SDK is installed by this guide.
Set `CUDA_ROOT` and `TENSORRT_ROOT` to your externally installed compatible SDKs.

```bash
cmake -S . -B build-gpu -DCMAKE_BUILD_TYPE=Release -DSYNREX_ENABLE_CUDA=ON \
  -DLAZPERF_SOURCE_DIR="$PWD/.deps/laz-perf" \
  -DNLOHMANN_JSON_INCLUDE_DIR="$PWD/.deps/json/single_include" \
  -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_CUDA_COMPILER="$CUDA_ROOT/bin/nvcc" \
  -DTENSORRT_ROOT="$TENSORRT_ROOT"
cmake --build build-gpu -j4
export LD_LIBRARY_PATH="$CUDA_ROOT/lib:$TENSORRT_ROOT/lib:${LD_LIBRARY_PATH:-}"
ctest --test-dir build-gpu --output-on-failure
python3 tools/qualify_streaming.py --native-dir build-gpu \
  --synthetic-model "$SYNREX_RELEASE/models/synthetic.srxmodel" \
  --unet-model "$SYNREX_RELEASE/models/unet.srxmodel" \
  --licensed-input "$SYNREX_RELEASE/data/benchmarks/TEST-0447_6390-002819414.laz" \
  --licensed-input "$SYNREX_RELEASE/data/benchmarks/TEST-0437_6398-002899556.laz" \
  --licensed-input "$SYNREX_RELEASE/data/benchmarks/TEST-0944_6564-005900112.laz" \
  --source-commit "$(git rev-parse HEAD)" --out build/new-gpu-qualification
.venv/bin/python tools/streaming_r2_trace_audit.py \
  --root build/new-gpu-qualification/equivalence --receipt build/new-trace-audit.json
SYNREX_RUN_GPU=1 SYNREX_TEST_NATIVE="$PWD/build-gpu/synrex-native" \
  .venv/bin/python -m pytest -q tests/test_byom.py \
  -k 'real_cuda_tensorrt_to_classified_laz or semantic_sidecar_retains_channels'
```

CUDA library-directory layout varies; use the actual installed `lib`/`lib64` directory.
The strict workload guard is conservative, including Windows compute reporting under
WSL. Do not bypass its rejection. Engine bytes are target/SDK-specific. To rebuild from
frozen ONNX on a different allocated target, follow the existing
[model-package and engine commands](../byom/RELEASE_R4.md); that new engine needs its own
numerical qualification and cannot inherit RTX results. CPU inspection is available with
`build/synrex-native inspect --model ...`.

Run [receiver, subscriber, then sender](STREAMING_R1_RUNTIME.md) in separate terminals.
Use new output receipt paths; keep all queue, memory, coverage and latency gates intact.
`streaming_r1_campaign.py --phase smoke` proves both models through external processes;
`--phase faults` preserves native FAIL for malformed/late/gap/duplicate/stalled input.
Sustained phases are optional future reproduction of accepted results, not required to
view the package. Never claim a passing sustained rate from the smoke or media runs.

For optional LAZ, add `--export-bundle NEW_DIRECTORY --taxonomy examples/unet/semantic_classes.json`
to the source-bound subscriber. Verify with `tools/streaming_export.py`; optional PDAL/QGIS
conversion uses [the unchanged verified COPC workflow](../byom/QGIS_R4.md). Native Linux
atomic no-replace publication is required; the known unsupported DrvFS case fails closed.

## Verify and regenerate media

The package verifier hashes every file, clones the bundle, compares all exported Git
blobs, recomputes every recorded delta against an independent oracle, and compares all
map snapshot arrays. NPZ archive timestamps may differ; semantic arrays must be exact.
Use an isolated environment for offline media tools:

```bash
python3 -m venv build/media-venv
build/media-venv/bin/pip install -r requirements-streaming-media.lock
build/media-venv/bin/python tools/streaming_r2_media.py \
  --source "$SYNREX_RELEASE/qgis/data/source/TEST-0944_6564-005900112.laz" \
  --recording "$SYNREX_RELEASE/recording" \
  --qgis-image "$SYNREX_RELEASE/qgis/TEST-0944_6564-005900112-qgis-map.png" \
  --second-scene "$SYNREX_RELEASE/qgis-r4-foundation" \
  --font /usr/share/fonts/truetype/dejavu/DejaVuSans.ttf --out build/new-review-media
```

FFmpeg is an external development tool from the pinned imageio-ffmpeg wheel, not shipped
in release assets. Rendering is deterministic at the point/semantic/camera level; encoded
video bytes can vary across FFmpeg/platform versions. All 1,296 frames must decode.
The main views show every third original point for viewing only; every source point was
freshly classified. No model or source label is fabricated.

To make a new genuine recording, `streaming_r2_replay.py prepare` converts the bound scene
into 16,384-point frames and 4,096-point chunks, preserving XYZ/intensity/returns/IDs.
Replay with native `synrex-stream replay --map-cells 65536 --queue 2`, then run
`streaming_r2_replay.py verify` into new paths. Its emulated timestamps are not LAS GPS.
This diagnostic JSON/file path is not the qualified low-latency TCP benchmark.

For fresh QGIS map export, run `tools/streaming_r2_qgis_render.py` with externally installed
QGIS Python and `SYNREX_QGIS_ROOT` set to a disposable copy of `qgis/`. It independently
reads every COPC source index/semantic ID and renders with the real provider. The optional
desktop worker is `streaming_r2_qgis.py`, launched through the installed QGIS environment
with an isolated local profile and `--code`; R2's two stalled startup attempts remain FAIL.
Do not import the problematic direct 48-byte LAZ files for the QGIS demonstration.
