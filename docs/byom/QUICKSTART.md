# Native BYOM R0 quickstart

<!-- Copyright (c) 2026 Tuomas Pietilä. SPDX-License-Identifier: CC-BY-4.0. Original expression only; upstream material and facts excluded. -->

Developer and project lead: Tuomas Pietilä. AI-assisted engineering with OpenAI Codex.

The runtime implements `DENSE_VOXEL_SEGMENTATION_V1`. It is model-neutral within that
explicit adapter, with no Python or PyTorch in native classification. CPU builds inspect,
package and plan; classification requires the CUDA/TensorRT build. See the
[adapter specification](ADAPTER.md), [provenance assessment](PROVENANCE.md) and
[qualification report](QUALIFICATION.md) before using it as a release candidate.

The [physical RTX4090 milestone](RTX4090_QUALIFICATION.md) qualifies the original 50,013-point
fixture and GPU suite. AGX execution remains separate. To reproduce the large test on an idle,
allocated NVIDIA workstation after building:

```bash
.venv/bin/python tools/qualify_dense_gpu.py --native build-gpu/synrex-native \
  --out build-gpu/dense-qualification --points 50013 \
  --verified-gpu-allocation VERIFIED_ALLOCATION_ID
```

This exports the model, creates invented LAZ/expected labels, builds the TensorRT engine/package,
runs native inference twice, independently verifies preservation and checks negative cases. The
desktop runner requires usable `nvidia-smi` telemetry and an allocation; it does not allocate an
AGX or stop other work. Its `--prepare-only` mode exports CPU fixtures while keeping GPU gates
explicitly NOT_RUN. Each native classification receipt includes separate monotonic wall-clock
and CUDA-event stage timings; see the report for measurement boundaries.

## Build on a CPU workstation

Prerequisites: Linux/glibc, CMake >=3.20, Ninja or Make, C++17 compiler, OpenSSL 3 development
headers/library, Git. Tested with GCC 13 on x86_64. No network fetches occur during CMake.
Keep third-party sources external to version control:

```bash
mkdir -p .deps
git clone --branch 3.4.0 --depth 1 https://github.com/hobuinc/laz-perf.git .deps/laz-perf
git clone --branch v3.11.3 --depth 1 https://github.com/nlohmann/json.git .deps/json
git -C .deps/laz-perf rev-parse HEAD
# b7bbe26109dc986f42d4fc80b8de3d2b6ca634ce
git -C .deps/json rev-parse HEAD
# 9cca280a4d0ccf0c08f47a99aa71d1b0e52f8d03
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DLAZPERF_SOURCE_DIR="$PWD/.deps/laz-perf" \
  -DNLOHMANN_JSON_INCLUDE_DIR="$PWD/.deps/json/single_include"
cmake --build build -j 4
ctest --test-dir build --output-on-failure
```

To build only the existing LAZ-label writer with its original dependency set, set
`-DSYNREX_BUILD_BYOM=OFF`. The original CLI and preservation suite are retained.

Development/export/verification dependencies are separate from runtime dependencies:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.lock -r requirements-byom.lock
.venv/bin/python -m pytest -q --basetemp=build/pytest
```

## Create an original model package

This exports an actual 1x1x1 Conv3D ONNX graph using original analytic constants, with no
training data, restricted weights or PyTorch. It produces low/mid/high feature-dependent
predictions; it is an engineering fixture, not a trained semantic survey model.

```bash
mkdir -p build/demo
.venv/bin/python examples/export_reference.py --out build/demo/reference.onnx
build/synrex-native pack --spec examples/dense_voxel.json \
  --onnx build/demo/reference.onnx --license LICENSES/MIT.txt \
  --out build/demo/reference-onnx.srxmodel
build/synrex-native inspect --model build/demo/reference-onnx.srxmodel
```

R0 uses a **directory** named `*.srxmodel`, schema `synrex.srxmodel.v2`. Its fixed members
are `manifest.json`, `MODEL_LICENSE.txt` and either `model.onnx` or `model.engine`.
All artifacts have SHA256 entries; the model/license files are verified before use.
This is derived from the prior package's contract/licensing/hash/target design, with a
new versioned format. It does not read the private v1 tar package or reuse its model semantics.
The manifest is not signed: hashes detect corruption relative to a trusted manifest,
not an adversary replacing an entire package. Use trusted local model sources.

## Build for an allocated NVIDIA target

Use a compatible NVIDIA GPU, Linux x86_64 or aarch64, CUDA development toolkit and TensorRT
10.x development libraries/headers including `nvonnxparser`. TensorRT 8/11, dynamic shapes,
INT8 I/O, plugins and sparse adapters are outside this R0 contract. CUDA/TensorRT stay under
NVIDIA terms; the install does not bundle or modify those system libraries.

**AGX:** perform CPU work on i5radeon. Run the following GPU commands only after an operator
has verified a nonconflicting allocation. An idle GPU reading alone does not establish that
the Spatial Edge Lab autopilot can be displaced. This work does not stop processes, change
power modes/clocks, install Jetson system packages or reserve its GPU automatically.

```bash
# Example architecture: 87 for Orin. Set the correct architecture for your allocated GPU.
cmake -S . -B build-gpu -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSYNREX_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87 \
  -DLAZPERF_SOURCE_DIR="$PWD/.deps/laz-perf" \
  -DNLOHMANN_JSON_INCLUDE_DIR="$PWD/.deps/json/single_include"
# If SDKs are outside standard paths, also set CUDAToolkit_ROOT and TENSORRT_ROOT.
cmake --build build-gpu -j 4
build-gpu/synrex-native target
build-gpu/synrex-native build-engine --model build/demo/reference-onnx.srxmodel \
  --out build/demo/reference.engine
build-gpu/synrex-native pack --spec examples/dense_voxel.json \
  --engine build/demo/reference.engine --license LICENSES/MIT.txt \
  --out build/demo/reference-engine.srxmodel
```

ONNX import and TensorRT compilation occur explicitly, once on the target, outside the
classification command. `build-engine` deserializes and validates the built engine's actual
tensor names, modes, shapes, dtypes and device/linear layout before saving it. Engine packages
record architecture, exact GPU name, compute capability, TensorRT version integer and CUDA
runtime version. Classification compares these before deserialization. Rebuild for a new target.
Supply engines built for the recorded target; repackaging does not make an incompatible plan
portable. Stock TensorRT's engine validation remains an additional compatibility check.

## Classify LAZ and independently verify

Supported input: nonempty compressed LAS 1.4, standard 375-byte public header, PDRF 6/7/8,
at least 30/36/38 record bytes respectively, one LASzip VLR, no EVLR/waveform/COPC;
metadata prefix <=16 MiB.
Coordinates must be finite with positive scales and within the adapter's resource limits.
Extra Bytes are retained within the declared record length. No coordinate reprojection,
georeferencing, arbitrary point formats or general LAZ editing is provided.

```bash
build-gpu/synrex-native classify --model build/demo/reference-engine.srxmodel \
  --input INPUT.laz --out CLASSIFIED.laz
```

The single native process decodes LAZ, creates block/voxel memberships, transfers points to
CUDA, builds features on CUDA, runs a persistent TensorRT context per block, performs CUDA
argmax and original-index scatter, then links the audited writer to preserve records and
metadata. An invocation-local label file connects inference to that writer; it needs no Python
or child executable. All output commands refuse replacement. A completed LAZ is published
only after successful inference and writer close. This is not a power-loss durability promise.

Your model's class indices are explicitly mapped through `las_classes`; no mapping is inferred.
Synthetic mapping `[2,5,6]` is a test convention. Choose an appropriate ASPRS/LAS mapping for
your model's trained semantics and retain its license/notice. LAS byte acceptance is not proof
of semantic accuracy. Replace compatible packages with `--model` without rebuilding the runtime.

For labels from an independent oracle, one mapped uint8 byte per original point:

```bash
.venv/bin/python src/verify_laz.py INPUT.laz CLASSIFIED.laz EXPECTED.u8
```

Run the physical GPU integration suite only under the verified allocation:

```bash
SYNREX_RUN_GPU=1 SYNREX_TEST_NATIVE="$PWD/build-gpu/synrex-native" \
  SYNREX_TEST_WRITER="$PWD/build-gpu/synrex-laz-labels" \
  .venv/bin/python -m pytest -q --basetemp=build-gpu/pytest \
  --junitxml=build-gpu/pytest.xml
```

These tests export/compile the real Conv3D model and compare native classified LAZ against
ONNX Runtime CPU predictions plus independent laz-rs record/metadata verification. They exercise
both overlap policies, FP32/FP16 tensor I/O, repeatability, bad tensor names/targets and relocated
inference. Without the opt-in they report `SKIPPED / NOT_RUN`; they do not substitute CPU inference.

## Install and relocate

```bash
cmake --install build-gpu --prefix "$PWD/build/install"
mv build/install build/relocated
build/relocated/bin/synrex-native version
```

The writer and generic core are statically linked from project/laz-perf sources. OpenSSL,
glibc/libstdc++, CUDA and TensorRT remain external system dependencies. The installed executable
has no build-tree RUNPATH; a correctly configured target must provide vendor libraries through
its normal loader configuration. The package may live anywhere. No Python paths or historical
reference build paths are embedded in the product workflow. CPU relocation is tested locally;
physical GPU relocation is qualified on RTX4090 and remains separate for AGX.
