# Physical AGX R3 admission and exact continuation

<!-- Copyright (c) 2026 Tuomas Pietilä. SPDX-License-Identifier: CC-BY-4.0. -->

Physical BYOM qualification is **NOT_RUN**. The read-only
[admission receipt](../../evidence/unet_r3_agx_admission.json) observes the actual
`synrex-spatial-autopilot-r0.service` active with PID1249253, a live
`point-lio:validate-benchmark` worker, no owner PAUSE, and **both** supervisor/GPU
flocks held by that supervisor. The older service names being inactive do not
establish availability. No native build, source/model transfer, GPU context,
engine build, benchmark or stack/power/service modification was performed.
An idle utilization sample cannot override those locks or reserve future work.

The observed storage guard reports an existing45.101GiB programme allocation;
about87GiB NVMe and8.2GiB root were available, with hard50GiB/5GiB reserves.
That free space is not a BYOM allocation and must not be consumed speculatively.
Existing CUDA13.2 nvcc and TensorRT development headers are present; their presence
is not a new ARM64 build/compatibility PASS.

After the supervisor releases its allocation naturally, obtain a documented BYOM
window and a separate bounded scratch allowance outside the programme's remaining
budget. Do not stop/pause the autopilot to create the window. Follow the existing
[full AGX plan](AGX_QUALIFICATION_PLAN.md), with these exact R2 inputs:

| Input | SHA256 |
|---|---|
| Frozen U-Net FP16 ONNX | `4a043dfacefa9e45f2e80e604b565f7ae90fb35ff81831af8f26cfd3d59ae047` |
| Frozen U-Net weights (reference only) | `788b65c42bb5d50d9509a56d38368f51890c4a93300d53578e290d193456c7c4` |
| Frozen U-Net FP32 ONNX | `316dfd710b455f2324a86e18f39c7df960525fe94fe3702579030f014887fb44` |
| Original analytic Conv3D ONNX | `e9a55b97e1c38e9be057f7f23f395be00418c80d6e160736af53e7a64d43067c` |

Use the immutable RC2 source tag `byom-unet-fractal-r2-rc2` at
`f8d3ab81804e0c427684ac08f1b055e75206b861` or its reviewed R3 descendant.
Retain audit commit `e3fcb82`. Transfer a verified local Git bundle and only the
bounded public model/example assets with notices. Do not transfer/reuse the SM89
plan, private source, full dataset, or allocate inside Spatial Lab's worktree.

Before each admission, record actual unit/worker state, nonblocking availability
of the existing supervisor and GPU locks, accepted boot/mount/error guards,
current free memory/storage and thermal/power state. An allocated runner must
hold **both existing locks for the entire build/engine/inference window**, refuse
if either is busy, and release them without editing/deleting their files. No
changes to supervisor state or policy. Enforce the original reserve/budget plus
the separately documented BYOM allowance throughout. If entry fails, stop before
transfer or execution and keep physical results NOT_RUN.

Within that allocation, use the existing SDK and qualified system toolchain:

```bash
cmake -S . -B build-agx -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSYNREX_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87 \
  -DLAZPERF_SOURCE_DIR="$PWD/.deps/laz-perf" \
  -DNLOHMANN_JSON_INCLUDE_DIR="$PWD/.deps/json/single_include"
cmake --build build-agx -j 2
ctest --test-dir build-agx --output-on-failure
build-agx/synrex-native target
build-agx/synrex-native pack --spec examples/unet/dense_voxel_asprs_safe.json \
  --onnx assets/unet-fp16.onnx --license examples/unet/MODEL_NOTICE.txt \
  --out assets/unet-onnx.srxmodel
build-agx/synrex-native build-engine --model assets/unet-onnx.srxmodel \
  --precision fp16 --out assets/unet-sm87.engine
build-agx/synrex-native pack --spec examples/unet/dense_voxel_asprs_safe.json \
  --engine assets/unet-sm87.engine --license examples/unet/MODEL_NOTICE.txt \
  --out assets/unet-sm87.srxmodel
```

Build a separate target-native original Conv3D engine from its exact ONNX/contract.
Use the **same executable hash** for both packages. Native target checks must
match ARM64/SM87 and the actual TensorRT version; never override them. Missing
prerequisites are blockers, not authorization to change the qualified stack.
The desktop `nvidia-smi` guard may be unsupported on Jetson (the read-only query
returned N/A); prepare Jetson-specific process/lock/telemetry admission rather
than disabling the guard.

Run the50,013-point synthetic oracle and at least the four real benchmark scenes
plus a rare-class scene. Compare actual native CUDA inputs/logits against the
frozen independent FP32 reference and record per-class point metrics/differences.
Keep the existing FP16 gates (RMSE<.03, point agreement≥.995), disclose measured
maximum error and rare-class changes, and never retrain against this consumed test.
Independently verify with laz-rs all source-order IDs, class mappings, nonclass
record bytes and qualified metadata; seal output/source/model/taxonomy hashes.
Repeat invocations and persistent cycles. Test corrupt hashes, incompatible
tensors/targets, malformed inputs and overwrite refusal. Relocate the installation
and repeat both models with no Python/Torch loaded in classification.

Capture launch-to-exit, native stage times, persistent cycles, complete throughput,
requested allocation peaks and separately sampled device residency. Use read-only
`tegrastats` and current nvpmodel/thermal telemetry; no mode, clock, cooling or
stack change. Record ambient/thermal/throttling context, power sampling scope,
software/compiler/source/model/engine hashes, allocation ID, and before/after
storage/worker state. Desktop inference numbers cannot qualify AGX performance.
