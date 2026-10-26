# RTX4090 native BYOM qualification

<!-- Copyright (c) 2026 Tuomas Pietilä. SPDX-License-Identifier: CC-BY-4.0. Original expression only; upstream material and facts excluded. -->

**Result: PASS for the original synthetic dense-voxel pipeline on RTX4090-Wsl.**
Qualified source: `5a40d81b54c266579f150002d0269ac8eeb61d62`, a descendant of the preserved
CPU checkpoint `eaf3ce21cf7fd1a2787f1a1b4599e7ab920612a5`. Later report-only commits do not
change those executed source identities. No GitHub push, publication or visibility change occurred.

## Environment, transfer and resource boundary

SSH access initially required the owner's Tailscale authentication check; the owner completed it.
The older forwarded SSH ports reset connections, so the verified, already-known direct endpoint
was used. No host-key verification was disabled. Git bundles transferred exact local commits;
no private repository, model or restricted dataset was transferred.

The target is Linux x86_64 / Ubuntu 24.04.3 under WSL2, NVIDIA GeForce RTX 4090 / SM89,
24,564 MiB GPU memory, Windows KMD 610.88 / WSL driver 610.57.01. Preflight showed 0% GPU
utilization, 24,009 MiB free GPU memory, about 85 GiB available RAM and 366 GiB free in the WSL
filesystem. The Windows backing drive had about 26 GiB free, so only the required SDK subset
was transferred, not the entire multi-architecture TensorRT wheel.

WSL process reporting has [documented limitations](https://docs.nvidia.com/cuda/archive/13.2.1/wsl-user-guide/index.html).
Both Linux and Windows-native `nvidia-smi` were inspected. Windows showed normal idle desktop
applications, no model training or other compute workload; Linux reported no compute processes.
The qualification repeatedly checked memory/processes/utilization before launching GPU commands.
The guard waits briefly for trailing utilization samples from preceding completed invocations;
it refuses reported competing compute processes or insufficient memory. No workload was stopped,
suspended or reconfigured. No AGX command was executed.

The target's system CUDA was 12.1; TensorRT and OpenSSL development headers were not installed
system-wide. Isolated user-workspace dependencies supplied CUDA 13.2.86 tools, the 13.2 runtime,
TensorRT 10.16.1.11 libraries with v10.16 public headers and matching OpenSSL 3.0.13 headers.
The existing system OpenSSL shared library remained in use. laz-perf 3.4.0 and json 3.11.3 stayed
external pinned dependencies. No driver, system toolkit, Jetson stack, clock or power setting changed.

## Executed pipeline and correctness

`tools/qualify_dense_gpu.py` exported the original 1x1x1 Conv3D model to ONNX, built and initialized
a genuine TensorRT engine on the target, created SHA256-verified ONNX/engine packages, and executed:

```text
50,013 invented source points in LAZ
 -> native laz-perf decode and generic block geometry
 -> CUDA feature reduction / NCZYX tensor preparation
 -> TensorRT enqueueV3, 18 completed inference blocks
 -> CUDA argmax, explicit LAS mapping and original-index scatter
 -> retained native metadata-preserving LAZ writer
```

The model has input `features [1,2,4,6,8]`, output `scores [1,3,4,6,8]`, FP32 I/O, occupancy
and normalized mean-intensity features, and explicit class mapping `[2,5,6]`. ONNX source/weights
are original analytic constants, not pretrained research assets. The runtime does not import Python
or PyTorch; export and the independent development oracle run separately.

All 50,013 labels exactly match independent brute-force geometry plus ONNX Runtime CPU inference.
The nonconstant LAS histogram is **2: 14,339; 5: 17,867; 6: 17,807**. Two fresh native processes
produce identical labels and identical classified LAZ. The second has `PATH=/nonexistent`; loader
diagnostics and native dependency inspection show no libpython/libtorch. Positive CUDA-event
intervals and completed `enqueueV3` block counts accompany the actual classified output.

Independent laz-rs verification passed both on RTX4090-Wsl and, after download, on i5radeon:
source point order, every nonclassification record byte, public header, non-LASzip VLRs, Extra Bytes
descriptors and padding are exact. Coincident points retain distinct original record identities.
The file crosses the writer's 50,000-point chunk boundary.

| Artifact | SHA256 |
|---|---|
| Original ONNX | `e9a55b97e1c38e9be057f7f23f395be00418c80d6e160736af53e7a64d43067c` |
| Target TensorRT engine | `4dd03d4c2c6291e98145c64ec2fb7e174ba69d7b7e1728e84cf7357e51772f59` |
| Classified LAZ, both repeats | `2230895f034708d78ef52a23533c42ae29be8f9a7f7f7aa82262ce6a48d81cb0` |
| Source-ordered labels | `7794ebb0fba2d8d5091ebcf41547879b5f7c5709c4ce1f9f5a1963d86db3acbb` |

Bad tensor name/shape, invalid class mapping, wrong target, corrupted SHA256, invalid engine bytes,
unsupported point format and output overwrite all reject without publishing or modifying output.
The complete target suite passes **55 tests, 1 skipped**; that skip is the CPU-only-build rejection
test. All four actual GPU cases pass, including overlap policies, intensity reduction, FP32/FP16,
alternate grid, repeated output and relocated inference.

The first run of the formerly unexecuted GPU tests exposed an overstrict fixture assertion:
`last_wins` constrains local-X features to the final tile stride and cannot reach that model's
highest feature band. The assertion was corrected to require nonconstant valid classes and exact
oracle agreement; all three classes remain mandatory in the intensity qualification. The initial
failure and a resource-guard pause are retained in private local logs, not relabeled as passes.

## Separate timing measurements

These are two complete file invocations, each with native engine/context initialization, no forced
clocks and ordinary OS caching, following engine compilation. They are a synthetic engineering
measurement, not a trained-model throughput or accuracy benchmark. No historical SegFormer
measurement is reused. The second invocation additionally enables loader diagnostics.

| Phase | First run, ms | Repeat, ms |
|---|---:|---:|
| CPU LAZ decode | 40.431 | 39.480 |
| CPU geometry | 8.745 | 8.909 |
| CPU preprocessing total | 49.177 | 48.389 |
| CUDA tensor preparation | 0.388 | 0.387 |
| TensorRT inference, 18 blocks | 0.083 | 0.088 |
| CUDA postprocessing | 0.085 | 0.082 |
| Native LAZ encoding/preservation | 68.159 | 65.546 |
| GPU initialization and initial upload | 234.330 | 237.351 |
| Complete native end to end | **357.186** | **357.806** |

CUDA timings use [CUDA events](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__EVENT.html)
on the actual inference stream and are accumulated across blocks after stream completion. They
are stream stage intervals, not a promise of pure kernel-only latency. Decode, geometry, writer
and end-to-end measurements use a monotonic host clock. End-to-end also includes package hashes,
allocation, transfer/orchestration, temporary label I/O and output hashing; rows are not additive
because CPU preprocessing total already includes decode/geometry and some overhead is separate.
The CUDA preparation interval brackets the occupied-voxel feature kernel. Input zeroing and
per-block membership transfers precede that interval and remain included in end-to-end time;
they are not represented as measured kernel-only preprocessing time.

An optional Nsight Systems 2023.1 profile also ran classification successfully and reproduced the
same output. Its importer reported `Unknown driver API function index: 719` with the newer CUDA
driver. Full profiler analysis is therefore not claimed as PASS; a CUDA-13-compatible Nsight version
is needed for a complete API trace. The required native CUDA-event timing and correctness evidence
are independent of that optional profiler. No profiler/system installation was changed.

## Evidence and next boundary

[Dense GPU receipt](../../evidence/rtx4090_dense_gpu.json),
[full target suite](../../evidence/rtx4090_full_suite.json),
[downloaded independent verification](../../evidence/rtx4090_download_validation.json) and
[requested status gates](../../evidence/rtx4090_status.json) record the executed outcomes.
[Target source/transfer identities](../../evidence/rtx4090_target_snapshot.json) verify the exact
checkout, clean target worktree and dependency/bundle bytes. The [private-transfer record](../../evidence/rtx4090_transfer.json)
records the isolated dependency boundary; [optional profiler diagnostics](../../evidence/rtx4090_profiler.json)
preserve the trace-import limitation separately from required GPU qualification.
Generated LAZ/models, raw SSH logs and profiler artifacts stay in ignored qualification workspaces.
[Source provenance review](PROVENANCE_REVIEW_R1.md) records a second-pass technical inspection,
not a clean-room or external legal certification.

The separate [physical AGX plan](AGX_QUALIFICATION_PLAN.md) is prepared. **Physical AGX build,
inference, performance and autopilot coexistence remain NOT_RUN.** Its nonconflicting allocation
and Jetson-specific resource checks are entry prerequisites. RTX4090 success does not qualify
an aarch64 binary or make its TensorRT engine portable to Orin. Nothing is published.
