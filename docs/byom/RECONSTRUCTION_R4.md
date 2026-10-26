# Reconstruct the R4 public release

<!-- Copyright (c) 2026 Tuomas Pietilä. SPDX-License-Identifier: CC-BY-4.0. -->

Use a trusted archive SHA256 from `evidence/unet_r4_release_candidate.json`. Extract it
into an empty directory and set `RELEASE` to the absolute extracted showcase directory.
All large model/data artifacts are in that package; they are absent from normal Git.
The original R3 archive/tag remain unchanged. No private checkout is required.

Clone this candidate's bundled public source history explicitly by branch, then detach
at its recorded source identity. The bundle does not advertise a default HEAD:

```bash
export RELEASE="$PWD/showcase-r4-rc2"  # extracted archive directory
git clone --branch codex/byom-native-mit-r0 "$RELEASE/source.bundle" public-source
cd public-source
SOURCE_COMMIT=$(python3 -c 'import json,os; print(json.load(open(os.environ["RELEASE"]+"/publication_candidate.json"))["source_commit"])')
git checkout --detach "$SOURCE_COMMIT"
mkdir -p .deps
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
cmake --build build -j 4
ctest --test-dir build --output-on-failure
.venv/bin/python -m pytest -q --basetemp=build/pytest
.venv/bin/ruff check .
.venv/bin/python tools/release_checks.py --check-only
.venv/bin/python tools/verify_showcase.py --root "$RELEASE" --receipt build/archive-verification.json
.venv/bin/python examples/export_reference.py --out build/reconstructed.onnx
sha256sum build/reconstructed.onnx
# e9a55b97e1c38e9be057f7f23f395be00418c80d6e160736af53e7a64d43067c
build/synrex-native inspect --model "$RELEASE/models/synthetic.srxmodel"
build/synrex-native inspect --model "$RELEASE/models/unet.srxmodel"
cmake --install build --prefix ../install-stage
mv ../install-stage ../relocated
../relocated/bin/synrex-native version
git status --porcelain
```

Export `RELEASE` into the environment before the commands above. Prerequisites are Linux,
Git, Python/venv, CMake ≥3.20, Ninja, a C++17 compiler and OpenSSL 3 development files.
The CPU build packages/inspects/verifies models and rejects classification without CUDA;
it does not substitute CPU neural inference. `verify_showcase.py` verifies every registered
archive file, compares all exported source bytes with bundled Git blobs, checks model
artifact hashes, independently decodes native pairs and restores source indices in COPC.
The fresh reconstruction receipt records 76 CPU PASS and five GPU tests NOT_RUN.

For NVIDIA classification, configure a separate `build-gpu` using the
[explicit CUDA/TensorRT build](QUICKSTART.md), set the actual allocated target architecture
(89 for the qualified RTX4090), and point CMake to the existing matching SDKs as needed.
Build without changing the target's system stack. Run no GPU job until a nonconflicting
allocation is verified. Then use the [two-model commands](RELEASE_R4.md) or original
[Conv3D qualification runner](QUICKSTART.md). The RTX engines are target-specific; Orin
needs a separate native engine and is NOT_RUN in this release. Training/export pins,
official dataset selections and the frozen evaluation protocol remain in the source
evidence and [training quickstart](UNET_R2_QUICKSTART.md).

To open existing predictions no GPU is needed: open `qgis/ready/synrex-fractal.qgz` in
QGIS. To reconstruct that project and its COPCs from source/native outputs, follow the
[one-command optional export](QGIS_R4.md), which needs external PDAL and QGIS-enabled Python.
Both models, all source/native data pairs, semantic sidecars/bindings, licences and genuine
screenshots stay in the release archive. No compiler, SDK, QGIS binary, dataset bulk archive
or private engineering repository is bundled.
