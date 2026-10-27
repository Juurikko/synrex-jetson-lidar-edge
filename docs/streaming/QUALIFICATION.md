# Track B streaming R0 qualification

<!-- Copyright (c) 2026 Tuomas Pietilä. SPDX-License-Identifier: CC-BY-4.0. -->

**PASS for B0 and bounded B1 on physical RTX4090**, with B2 equivalence and B4 event/map
implementations. This is local development, not a streaming release. Full B3/B5 and target
hardware qualifications remain gated. The executed source is
`a20051f2490adb92a6cc36f390906639b875ea20`, descended from final R4 audit `c3451b6`.
The archive SHA256 was identical before/after transfer, and its embedded Git identity was
verified. Later report-only commits do not change this executed runtime.

[Exact gates and source hashes](../../evidence/streaming_r0_qualification.json),
[full sanitized RTX campaign](../../evidence/streaming_r0_rtx_campaign.json),
[baseline identity/resource contract](../../evidence/streaming_r0_baseline.json),
[protocol](PROTOCOL.md) and [restartable build/qualification procedure](QUICKSTART.md)
are committed. Raw tensors, licensed point fixtures, predictions and logs stay in ignored
qualification workspaces. No model/data binaries were added to Git.

| Gate | Result |
| --- | --- |
| B0 strict protocol, IDs/timestamps/frame assembly, bounded budgets, deterministic fixtures | PASS |
| B1 persistent model verification/engine/context/stream/device buffers across different windows | PASS |
| B1 bounded input/output queues, backpressure, cancellation and loss accounting | PASS |
| B1 physical RTX4090 inference, both unchanged frozen models, same executable | PASS |
| B2 identical input tensors, finite logits, exact semantic IDs/LAS mapping/source order | PASS |
| B2 all seven semantic outputs, explicit loss/reorder rejection | PASS |
| B3 stage measurements | PASS |
| B3 new CUDA voxel/index optimization | NOT_RUN |
| B4 complete classified events, bounded semantic map and deltas | PASS |
| B4 optional asynchronous LAZ/COPC exporter | NOT_RUN |
| B5 two densities × three short paced rates and explicit packet faults | PASS for coverage/accounting |
| B5 sustainable rates, long motion/occlusion matrix and genuine sensor clocks/transport | NOT_RUN |
| B6 gated native ARM64/SM87 and SM110 configurations | PASS for source/configuration only |
| Physical Orin/AGX and Thor build/inference/performance | NOT_RUN |
| Public push, PR or streaming release | NOT_RUN |

Tests: two native CTests pass in CPU Release and ASan/UBSan builds; 81 Python tests pass on
the CPU host with five physical GPU cases correctly skipped there. Those same five existing
R4 GPU cases then pass on RTX with the new binary, including FP32/FP16, both overlap policies,
alternate grids, original record preservation, relocated inference and semantic sidecars.
RTX native CTest also passes both tests. CPU orchestration fixtures explicitly use a
MODEL_FREE_TEST_ONLY session; their results are never reported as neural performance.

The RTX preflight found 0% utilization, no Linux compute process, only identified Windows
desktop applications, 24,009 MiB free GPU memory and sufficient isolated storage. No running
workload, driver, dependency installation, clocks or power limits were changed. Existing R4
SDKs and model packages were read from their qualified locations. Both packages and both
streaming executables have identical before/after hashes. AGX/Thor were never contacted.
The native replay also passes with `PATH=/nonexistent`; dynamic dependencies include CUDA,
TensorRT, OpenSSL and standard native libraries, with no libpython/libtorch.

Actual neural equivalence results:

| Frozen model | Freshly classified points | Successive windows | Maximum logit absolute error | Semantic output histogram |
| --- | ---: | ---: | ---: | --- |
| Original synthetic Conv3D | 3,108 | 4 | 0 | `[585,1944,579]` |
| Frozen FRACTAL U-Net | 91,117 | 6 | 0 | `[9798,46028,3917,29370,92,1089,823]` |

U-Net input combines four invented windows and 88,009 genuine licensed source records from
two IGN/FRACTAL test scenes. IDs are original record indices for those genuine fixtures.
Their replay timestamps are explicitly emulated UNIX frame times; no claim is made about
original LAS GPS clocks or return decoding. Every supplied source timestamp/identity survives.
Both paths use identical frame boundaries, zero temporal halo, model spatial context and R4
voxelization/overlap. Actual CUDA inputs are byte-identical, model logits agree within the
predeclared `.001 + .001 * abs(reference)` budget (observed error zero), and all source-order
semantic channels and explicit safe LAS projections match exactly. This is equivalence, not
a new accuracy evaluation or retraining result.

One threaded U-Net replay, including model startup, JSON parsing, inference, classified-point
events, map deltas and output sync, measured (final hashing/receipt I/O and teardown excluded):

| Measurement | Observed |
| --- | ---: |
| Incoming source points / freshly classified / emitted | 91,117 / 91,117 / 91,117 |
| Dropped points / partial frames / missing sequences | 0 / 0 / 0 |
| Incoming points/s, initialization included | 75,390.41 |
| Freshly classified points/s, initialization included | 75,390.41 |
| Map updates/s, initialization included | 11,174.89 |
| Map deltas / resident map cells | 13,506 / 10,942 |
| Session initialization / replay measurement | 220.08 ms / 1,208.60 ms |
| Replay drain time excluding initialization | 988.37 ms |
| Replay-only incoming and freshly classified points/s | 92,189.38 |
| First-chunk acceptance to output p50 / p95 / p99 | 14.33 / 710.85 / 710.85 ms |
| CPU JSON parse + assembly / map update | 634.81 / 142.60 ms |
| Host peak RSS for this replay | 442,806,272 bytes |
| Tracked requested CUDA allocation peak, U-Net session | 8,199,355 bytes |

Latency percentiles use nearest rank over complete frames (latest at most 4,096 frames in an
unbounded run); this sample has six frames. Large-frame age includes assembly and event/map
writes. Queue capacities are two frames each; this run reached one frame high water with no
producer waits. Device-wide campaign samples peaked at 528 MiB used, 8% sampled utilization
and 55.43 W; 200 ms sampling can miss transient peaks. These device-wide figures include
context/desktop/vendor allocations and differ from tracked requested adapter/vendor bytes.

Diagnostic U-Net totals across the six equivalence windows were 11.85 ms CPU geometry,
1.22 ms H2D/zeroing, 1.88 ms CUDA feature preparation, 2.09 ms TensorRT, 0.062 ms CUDA scatter
and 0.95 ms D2H. Diagnostic window wall time was 34.30 ms including tensor trace work. These
are different scopes from complete replay and must not be substituted for all-point E2E
rates. CUDA feature preparation uses CPU-generated memberships; new CUDA index generation
has not been implemented. The measured bottleneck is CPU JSON parsing/assembly, so this slice
makes no neural acceleration claim and retains the existing verified GPU kernels.

Short paced runs requested 100,000, 1,000,000 and 10,000,000 incoming points/s, at 256 and
4,096 points/frame, for twelve frames each. All six emitted every point with no loss. The
actual replay-only incoming/freshly classified rates were approximately 64k–109k points/s.
Higher requests accumulated up to 400.84 ms schedule lag. These results do **not** pass those
requested rates as sustainable targets. Packet loss and reordering tests deliberately produce
FAIL receipts: loss input received 48,896 points, emitted 45,056 and explicitly dropped 3,840;
reordering received 49,152, emitted 45,056 and dropped 4,096. Both record one missing sequence.
Their enclosing rejection gates PASS because incomplete coverage was detected, never hidden.

The source commits are `7c68470` (first B0/B1 implementation), `0130f57` (accounting,
equivalence and qualification hardening) and `a20051f` (portable WSL system-tool discovery).
A final evidence-only commit records these receipts. The accepted R4 tag object and target
commit remain unchanged; frozen packages, weights and accepted R4 predictions are untouched.
