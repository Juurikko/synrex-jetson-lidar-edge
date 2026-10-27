# Track B streaming developer quickstart

<!-- Copyright (c) 2026 Tuomas Pietilä. SPDX-License-Identifier: CC-BY-4.0. -->

Use an isolated branch descended from final R4 audit `c3451b6`; do not alter accepted tags,
weights or packages. New original source/tests are MIT; original documentation is CC BY 4.0.
Third-party SDK/model/data terms remain separate. No private engineering source was imported.
Dependencies and pinned revisions are the public R4 [reconstruction](../byom/RECONSTRUCTION_R4.md)
and [CUDA build](../byom/QUICKSTART.md). No new runtime dependencies are introduced beyond C++
threads. Native production inference never starts Python, PyTorch or ONNX Runtime.

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DLAZPERF_SOURCE_DIR="$PWD/.deps/laz-perf" \
  -DNLOHMANN_JSON_INCLUDE_DIR="$PWD/.deps/json/single_include"
cmake --build build -j 4
ctest --test-dir build --output-on-failure
build/synrex-stream validate --input examples/streaming/deterministic.jsonl
build/synrex-stream fixture --out build/replay.jsonl --frames 4 --points 4096 \
  --chunk 1024 --seed 7 --taxonomy synrex.fractal7.v1
```

No GPU SDK is required for parser, assembly, queue, map or orchestration tests. CPU inference
commands fail closed. For sanitizer checks, use a separate build with `-DSYNREX_SANITIZE=ON`.
The physical GPU qualifier is built but deliberately excluded from ordinary CTest execution.

On an available RTX allocation, configure `build-gpu` with the R4 documented SDK arguments and
`-DSYNREX_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89`. Recheck GPU compute processes,
utilization and free memory, including Windows-native reporting on WSL, before GPU execution.
Leave all other workloads and system settings alone.

```bash
build-gpu/synrex-stream replay --input build/replay.jsonl \
  --model "$RELEASE/models/unet.srxmodel" --out build/events.jsonl \
  --receipt build/replay-receipt.json --queue 2 --policy block --map-cells 65536
build-gpu/synrex-stream-gpu-tests "$RELEASE/models/synthetic.srxmodel" \
  "$RELEASE/models/unet.srxmodel" build/equivalence "$LICENSED_LAZ"
```

All outputs must be new. CLI options are strict key/value pairs. Remove only your disposable
build outputs before repeating commands, or use a new destination. Partial event files after
FAIL are not complete deliverables. The final receipt is required. `--policy reject-new` is
available for explicit loss testing; any unclassified or unemitted input makes coverage FAIL.
SIGINT/SIGTERM request cancellation at the next producer boundary; the C++ Service API can
cancel directly from another thread. No mid-kernel teardown occurs.

[Session API](../../native/include/synrex/session.hpp) verifies a package once, deserializes one
engine, creates one context/stream and preallocates all device adapter buffers for its lifetime.
All windows share those resources. The 512 MiB adapter cap excludes TensorRT engine/context
allocations and the CUDA driver context; tracked allocation peak includes requested vendor
allocator calls. A minimum 64 MiB free-device reserve is checked before adapter allocation.
Host storage is bounded by one assembling frame, at most eight input and eight output frames,
one active inference and one writer frame, plus geometry (model's membership/block limits),
map capacity and a maximum 1 MiB parser line. Default queues use considerably less than these
maxima. Geometry remains R4 CPU geometry; CUDA feature preparation/scatter remain unchanged.
GPU voxel/index optimization requires measured evidence and a subsequent parity gate.

The qualifier uses the same executable for both frozen models without rebuilding. It compares
actual CUDA-prepared tensors byte for byte, source-order classes/projections exactly, and
finite TensorRT logits with `abs_error <= .001 + .001 * abs(reference)`. Reference is native
R4 `infer()` with identical frame boundaries, geometry, context/overlap and zero temporal halo.
Trace I/O is diagnostic overhead, so qualifier timings are not neural performance claims.
Optional licensed LAZ input is split into contiguous windows of at most 65,536 source records;
IDs equal original record indices. No LiDAR or model binaries enter Git.

Use `--rate-pps 100000` on replay for optional point-count-paced packet delivery. Maximum
schedule lag is reported explicitly; this short experiment is not sustained-rate qualification.
Pacing uses monotonic host time and preserves every historical source timestamp unchanged.

Replay reports incoming points/s, freshly classified points/s, map updates/s, drops, partials,
late points and p50/p95/p99 acceptance-to-output latency separately. Both initialization-inclusive wall rates and replay-only rates are reported. Unpaced or short
paced runs do not establish sustainable timestamped sensor rates.
Source-to-wall age is NOT_RUN for historical timestamps in this replay CLI.
The separate sustained benchmark supplies deterministic emulated acquisition times and the
longer density, fault, motion, occlusion and resource matrix; see the
[bounded RTX report](SUSTAINED_RTX_REPORT.md) and [sustained protocol](SUSTAINED_PROTOCOL.md). RTX results never
qualify physical Orin/Thor. ARM builds require a separate target engine, exact SDK identity and
an available device; the frozen RTX engine must never be copied as a portable target plan.

Native target cache configurations are prepared for later Orin/Thor work:
`cmake -C cmake/targets/orin.cmake ...` and `cmake -C cmake/targets/thor.cmake ...`.
Both reject non-ARM64 hosts and incorrect SM selections. Supply target-native dependency
paths, build ONNX into a new target engine, package it with the actual target identity, then
run this same equivalence suite physically. TensorRT 10 API support, CUDA toolkit SM support,
models, numeric budgets, resource coexistence and sustained performance need separate target
qualification. Orin/Thor builds, inference and performance remain NOT_RUN. No AGX/Thor
workload is accessed. LitePT-S/TorchSparse++ stays outside this dense-adapter source line.

For the restartable campaign, run this development orchestrator on the idle RTX host after
building the exact source commit. It guards resource use, executes only native inference,
checks both package hashes before/after, runs the same binary without Python on PATH, exercises
two densities at three short paced rates, and requires explicit FAIL for loss/reordered input:

```bash
python3 tools/qualify_streaming.py --native-dir "$PWD/build-gpu" \
  --synthetic-model "$RELEASE/models/synthetic.srxmodel" \
  --unet-model "$RELEASE/models/unet.srxmodel" \
  --licensed-input "$LICENSED_LAZ" --source-commit "$(git rev-parse HEAD)" \
  --out "$PWD/build/streaming-campaign"
```

Repeated `--licensed-input` accepts further licensed source scenes. LAS fixture timestamps
are explicitly emulated UNIX frame timestamps; the qualifier does not claim original GPS
clock or return-attribute decoding. Stream metadata preserves the attributes actually supplied.
Raw generated events, point fixtures, tensors and logs remain outside Git. Device memory,
utilization and power are sampled at 200 ms intervals where available; peaks may be missed.
This campaign remains a short correctness probe. Sustained results come from `synrex-sustain`
and its separate 30–60-second campaigns, never from these short paced probes.

For a bounded 2M point-plus-map confirmation with the frozen trained model, use an idle RTX
allocation and a new output directory:

```bash
python3 tools/confirm_streaming.py --native "$BUILD/synrex-sustain" \
  --model "$UNET" --sources "$SCENE_A" "$SCENE_C" \
  --source-commit "$(git rev-parse HEAD)" --out "$NEW_CONFIRMATION"
```

This selects 16,384-point windows, two frames per queue, a 65,536-cell map, five seconds of
warmup, and three 60-second target trials plus independent point-only/output regressions.
Window sizes are explicit context choices; compare each with its own identical-boundary
offline reference. Pinned staging remains off by default.
For relocated binaries, export the actual installed CUDA and TensorRT runtime library
directories. The qualified CUDA SDK uses `$CUDA_ROOT/lib`; do not assume `lib64` exists.
