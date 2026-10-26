# Separate bounded streaming inference milestone

<!-- Copyright (c) 2026 Tuomas Pietilä. SPDX-License-Identifier: CC-BY-4.0. -->

The qualified persistent benchmark reuses a decoded cloud, geometry, engine,
context and buffers. It is not a streaming classifier. Preserve the frozen U-Net
and Conv3D packages as regressions while implementing streaming separately.

1. Introduce a native resident-session API with explicit memory/queue limits,
   cancellation and backpressure. Retain the static dense adapter contract;
   dynamic batching requires an independently versioned contract and tests.
2. Preserve whole-cloud-minimum grid anchoring exactly. Either obtain trusted
   bounds with validation or perform an initial pass. Finalize count/mean features
   only after the complete voxel/block is available. Spatial watermarks must be
   explicit for out-of-order input; never silently classify incomplete cells.
3. Carry immutable source record IDs through block processing and overlap
   resolution. Use bounded disk spooling/reordering for sequential output.
   Preserve original metadata and every unrelated point byte. Commit the LAZ,
   semantic sidecar and output binding as a recoverable result bundle.
4. Compare every label and point record against the frozen batch implementation
   over several chunk sizes, ordering patterns and overlaps. Test cancellation,
   disk-full/restart, queue saturation and corrupt input without partial success.
5. Measure first-result and complete-file latency, p50/p95/p99 steady-state
   latency, sustained full-I/O throughput, GPU/host/disk high-water marks and
   exact point coverage under bounded memory. Report concurrent-load conditions.

Physical AGX work requires a fresh nonconflicting allocation and unchanged Jetson
stack. Build an SM87 plan from preserved ONNX and rerun independent numeric,
record/metadata and memory checks. No AGX performance promise follows from SM89.

Sparse LitePT-S/TorchSparse++ is a different follow-on project with coordinate,
feature, sparse-operator and complete-network contracts; see the
[separate assessment](LITEPT_SPARSE_PLAN.md). Neither dense model interchangeability
nor a Python TorchSparse wheel proves native sparse-transformer compatibility.
