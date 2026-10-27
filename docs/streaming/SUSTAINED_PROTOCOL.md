# Sustained streaming qualification protocol

<!-- Copyright (c) 2026 Tuomas Pietilä. SPDX-License-Identifier: CC-BY-4.0. -->

Continuation baseline: accepted Track B `c5d65c6`. Frozen R4 models, source identities,
predictions, tags and earlier qualification receipts remain unchanged. New measurements use
a separate native `synrex-sustain` executable and isolated output directories.

The primary contract is the versioned typed C++ Frame API followed by complete source-order
classified-point callbacks. It includes source generation/copy, point validation and conversion,
CPU geometry, actual CUDA preparation, TensorRT, CUDA reconstruction, transfers, bounded
service queues, and output consumption. Every emitted ID, coordinate, intensity and timestamp
is checked against the deterministic source. Point/class metadata enters a rolling digest.
No classified points are replaced with map state. Optional `json` mode additionally encodes
classified events in the output thread. JSON input parsing and disk encoding are separate
frontend/export costs, not hidden inside the typed API result.

`points` consumes all classified source events; `points-map` additionally updates the original
bounded semantic map and consumes all deltas. Source points/s and semantic-map deltas/s are
reported separately. `model-only` executes actual repeated TensorRT enqueueV3 on the last
occupied CUDA-prepared block, with 32 enqueues per synchronization. Its blocks/s and block
latency measure neural execution; point-equivalents/s are explicitly an upper bound excluding
geometry, reconstruction, queues and output. No source points are emitted in that mode.

Default gates are declared before measurement: five seconds of warmup, at least 30 seconds
of measured complete frames, every received point emitted exactly once, p99 source-to-output
latency ≤250 ms, source schedule lag ≤50 ms, oldest pending source point ≤500 ms, two input
and two output queue slots, process RSS ≤1 GiB, measured RSS range ≤64 MiB, and device-wide
sampled usage ≤4 GiB. Arrival times never slip to conceal overload. The producer rejects a
full queue visibly and stops that experiment. Source latency includes point acquisition within
a frame, queueing, model execution and output consumption. Histograms count every sample in
fixed storage; percentile bin upper edges have at most 3.125% relative quantization error.
Warmup and measured samples are separated by source frame identity. Drain is included in rates.

Sources are deterministic synthetic data or immutable licensed LAZ records decoded once before
replay. LAZ source IDs are original record indices, including when a frame wraps through the
finite corpus. Each temporal observation has a distinct frame ID. Emulated UNIX acquisition
timestamps project onto the monotonic replay clock; original LAS GPS time is not asserted.
No reprojection, motion correction or model preprocessing changes occur. Replaying the same
records in successive frames still runs fresh neural inference every time.

Initial sweep: 50k, 100k, 200k, 350k, 500k, 750k, 1M incoming points/s, stopping a mode after
its first failed bound. Higher rates require a passing lower rate. Repeat the best supported
rate. The configuration, actual duration, complete coverage, queue history, latency, resource
samples, source/package/binary hashes and explicit failed/skipped trials accompany every result.
Compare identical fixtures, window boundaries, context and output modes when evaluating an
optimization. Preserve the baseline binary; change one bottleneck at a time.

Example on an available RTX allocation with the existing qualified SDKs and packages:

```bash
build-gpu/synrex-sustain --model "$UNET" --source "$LICENSED_LAZ" \
  --rate 100000 --seconds 30 --warmup 5 --points 4096 --mode points-map \
  --receipt build/sustain-100k.json
python3 tools/sustained_campaign.py --native "$PWD/build-gpu/synrex-sustain" \
  --model "$UNET" --source "$LICENSED_LAZ" --source-commit "$(git rev-parse HEAD)" \
  --out build/sustained-baseline
```

No Python/PyTorch executes production inference. Python controls native processes and samples
hardware only. The resource guard checks both Linux and Windows desktop preflight on WSL,
stops its own process on competing reported compute workloads or resource violations, and
never changes drivers, clocks, system dependencies or unrelated jobs. AGX/Thor remain untouched.

The next qualification matrix preserves the two accepted 1.5M point-plus-map trials and the
2M bounded-queue failures. `tools/window_campaign.py` runs unchanged reference and optimized
binaries in separate directories, closes the 600k/700k/750k comparisons, and tests 1.6M,
1.75M, 1.9M and 2M adaptively. It tests 8,192/16,384/32,768 points per frame after the
4,096-point case, repeats successful target rates, and then uses distinct licensed scenes.
A frame size is part of the inference context: predictions at different frame boundaries
are **not** required or claimed to match each other. Every frame-size configuration must
match its own R4 offline reference with identical geometry, boundaries and overlap policy.
The frozen dense tensor shape and weights are unchanged by source-frame batching.

`--pinned 1` selects experimental persistent host staging; default is zero. It does not
change queues, CUDA stream count, tensor features or neural weights. Compare matching point
and map digests and trace bytes before interpreting its timings. The GPU qualification
also destroys each session and asserts zero remaining requested tracked CUDA/pinned bytes;
CUDA context and vendor caches are outside this allocator accounting. Device-wide memory is
sampled separately. No `cudaMalloc` or `cudaMallocHost` occurs in the persistent run loop.

Native counters increment on every successful actual TensorRT enqueue and every successful
freshly reconstructed point window. Sustained acceptance reconciles those counters with the
sum of observed inference blocks, completed service windows and freshly classified points.
A semantic histogram and occupied-voxel counts describe the measured source distribution.
Map-label reuse remains zero; map state never enters the inference result path.

`--timeline 1` records at most 256 sampled measured windows. Host monotonic timestamps cover
source-window acquisition, submission, inference, sink entry, map update and sink completion.
Each block records CUDA-event preparation/inference/scatter/transfer durations and the host
submission/synchronization bracket. This is an instrumented diagnostic: GPU events use their
own clock, so their absolute placement on the host timeline is not asserted. Source generation,
worker execution and map consumption overlap. Report their separate service times and measured
source-to-output latency; never sum concurrent stage durations to claim wall-clock latency.
Scalar semantic-map validation is separately timed as a subset of map processing.

The strict out-of-order fixture remains a negative protocol test. The swapped packet plus
unassemblable tail yield 12 late points and 16 dropped points, with 48 of 64 incoming points
assembled. The original expectation of four late points was wrong; the assembler was not
relaxed. Its original FAIL and corrected PASS are retained in
[the regression receipt](../../evidence/sustained_out_of_order_regression.json).
