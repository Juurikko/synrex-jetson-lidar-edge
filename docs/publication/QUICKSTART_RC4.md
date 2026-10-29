# Streaming R2 RC4 public source reconstruction

<!-- Copyright (c) 2026 Tuomas Pietilä. SPDX-License-Identifier: CC-BY-4.0. Original expression only; upstream material and facts excluded. -->

This source release adds native streaming to the separately preserved
[public R4 foundation](https://github.com/Juurikko/synrex-jetson-lidar-edge/releases/tag/byom-unet-fractal-r4-rc2).
The source and 37-second video are the public RC4 package. No private Git bundle,
review archive, external SDK, dataset or model binary is needed for the CPU checks.

Prerequisites: Linux, Git, Python 3.12/venv, CMake 3.20+, Ninja, a C++17 compiler
and OpenSSL 3 development files. Use a new directory:

```bash
git clone --single-branch --branch releases/streaming-r2-rc4 https://github.com/Juurikko/synrex-jetson-lidar-edge.git streaming-r2-source
cd streaming-r2-source
git checkout --detach streaming-r2-rc4
git rev-parse HEAD
git clone --depth 1 --branch 3.4.0 https://github.com/hobuinc/laz-perf.git .deps/laz-perf
git clone --depth 1 --branch v3.11.3 https://github.com/nlohmann/json.git .deps/json
git -C .deps/laz-perf rev-parse HEAD
# b7bbe26109dc986f42d4fc80b8de3d2b6ca634ce
git -C .deps/json rev-parse HEAD
# 9cca280a4d0ccf0c08f47a99aa71d1b0e52f8d03
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.lock -r requirements-byom.lock
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSYNREX_ENABLE_CUDA=OFF   -DLAZPERF_SOURCE_DIR="$PWD/.deps/laz-perf"   -DNLOHMANN_JSON_INCLUDE_DIR="$PWD/.deps/json/single_include"
cmake --build build -j2
ctest --test-dir build --output-on-failure
.venv/bin/python -m pytest -q --basetemp=build/pytest
.venv/bin/python tools/release_checks.py --check-only
```

An extracted source tar has no Git metadata. To run the qualification-preparation
test and historical tools that record `git rev-parse HEAD`, initialize a local
repository and commit that extracted source first, using your own Git identity.
The public tag/asset checksum remains the external source identity; a locally
created reconstruction commit does not replace it.

```bash
cmake -S . -B build-sanitize -G Ninja -DCMAKE_BUILD_TYPE=Debug   -DSYNREX_ENABLE_CUDA=OFF -DSYNREX_SANITIZE=ON   -DLAZPERF_SOURCE_DIR="$PWD/.deps/laz-perf"   -DNLOHMANN_JSON_INCLUDE_DIR="$PWD/.deps/json/single_include"
cmake --build build-sanitize -j2
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1   ctest --test-dir build-sanitize --output-on-failure
```

CPU execution validates contracts, transport, geometry, mapping and preservation;
it does not simulate neural inference. Five GPU tests require an independently
allocated GPU, and six recorded-event adversarial tests require a genuine R2
recording; they skip when those inputs are absent. A skipped test is NOT_RUN.

For independently published weights, ONNX and ready-to-view COPC examples, download
`showcase-r4-rc2.tar.gz` from the R4 release above and verify SHA256
`a2b6fa566b8982859e2e7b7f1a986291d0f056bb2460ccc8199a7f540dd0d82c`
before extracting. Its checkpoint SHA256 is
`788b65c42bb5d50d9509a56d38368f51890c4a93300d53578e290d193456c7c4`.
Retain its data/model notices. R4 remains a separate release with its original
source and benchmark scope; its binaries and TensorRT plans are target specific.

[R1 runtime](../streaming/STREAMING_R1_RUNTIME.md) describes receiver/subscriber/sender
commands. New neural runs need the separately installed CUDA/TensorRT SDK and
appropriate licensed model/input assets; use new output locations. The optional
[LAZ/COPC workflow](../byom/QGIS_R4.md) is outside headline streaming throughput.

Earlier R2/RC2/RC3/RC4 package instructions and scripts are preserved historical
source. They can refer to local tags, review inputs and bundle-based workflows
that are deliberately absent here. Do not run those owner-review packagers as a
public publication procedure. They are not prerequisites for building this source.
The original private review archive and its history are not public assets.

The released 37-second video is the approved unchanged visual illustration, not
a timing source. Its historical on-screen “local candidate” label is retained.
No physical sensor, flight, pose, Orin or Thor qualification is claimed.
