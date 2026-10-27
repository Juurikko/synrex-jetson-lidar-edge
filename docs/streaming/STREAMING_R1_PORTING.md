# Independent ARM64 continuation gates

<!-- Copyright (c) 2026 Tuomas Pietilä. SPDX-License-Identifier: CC-BY-4.0. -->

R1 source/configuration preparation does not constitute a target build or performance result.
AGX/Orin and Thor were not accessed. The existing resource allocation to other projects is
unchanged. Target work starts only on an explicitly available physical allocation.

| Gate | Orin | Thor |
| --- | --- | --- |
| Native CPU/OS | ARM64 Linux | ARM64 Linux |
| CUDA architecture | SM87 | SM110 |
| Configuration | `cmake/targets/orin.cmake` | `cmake/targets/thor.cmake` |
| SDK | Record actual JetPack/CUDA/TensorRT and support for native adapter API | Record actual Thor SDK/CUDA/TensorRT and SM110 compiler support |
| Engine | Build a new SM87 engine from the approved frozen ONNX | Build a new SM110 engine from the approved frozen ONNX |
| Physical build/inference/throughput | NOT_RUN | NOT_RUN |

The RTX serialized engine is not portable qualification. Preserve its package, weights, source
ONNX/export identity and license notices. Store each new engine in a new target-bound package;
never overwrite frozen RTX artifacts. Record every dependency and binary hash. Source-only MIT
rights do not override model, SDK or data licenses.

First run the CPU wire/geometry/map/export tests and ASan/UBSan natively. The wire layout is
explicit little endian with binary64 doubles and fixed-width integer fields; no structure
padding or pointer values cross the network. Map pointers remain process-local. The Linux
socket and `renameat2(RENAME_NOREPLACE)` export paths require native syscall support. Verify
atomic no-replace publication on the actual target filesystem; fail closed if unsupported.
RSS sampling must use the target's actual page size, rather than assuming desktop 4 KiB pages.

Then run both frozen Conv3D and trained FRACTAL models through the same native binary and
adapter. Compare exact membership, source/voxel order and prepared tensors; require existing
finite-logit numeric budgets and exact reconstructed semantic IDs/LAS projection with original
IDs and measurement times. Reject class differences rather than adjusting preprocessing,
weights, class taxonomy or hidden overlap. Check repeated context reuse and cleanup.

Only after numerical parity run external producer/receiver/subscriber qualification. Record
source rate, freshly classified rate, map observation/update/insertion/eviction rates, complete
coverage, losses, queue peaks, replay scheduling age and receive-to-validated-output latency.
Keep existing correctness/latency/queue gates. Shared-memory target memory accounting and target
thermal/power sampling need explicit platform receipts; RTX NVML observations cannot substitute
for them. Do not change power modes or other projects' allocations without later authorization.

Repeat stable A/B, dense C and full spatial churn at rates the target actually supports. Use
warmup, two 60-second repetitions, bounded adaptive failure stops and a physically measured
10–30-minute stability run where available. Preserve failed trials. Exercise partial TCP reads,
reconnects, missing/duplicate sequences, timestamp discontinuities, cancellation, slow subscribers
and export under load. A passing software loopback replay does not qualify a physical LiDAR,
UDP vendor parser, SLAM pose source or geographic projection.

LitePT-S/TorchSparse++ and larger models are separate future adapters. They are not included in
the R1 dense runtime, its model-neutral interface, its performance figures or either target gate.
Public publishing still requires a later owner decision; all continuation remains local.
