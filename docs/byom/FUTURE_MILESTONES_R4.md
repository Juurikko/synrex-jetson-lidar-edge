# Separate development proposals after the source release

<!-- Copyright (c) 2026 Tuomas Pietilä. SPDX-License-Identifier: CC-BY-4.0. -->

These are proposed milestones, not implemented release capabilities. They do not alter the
frozen dense U-Net model, R3 results or current source-preserving classifier. Each needs
its own versioned contracts, resource allocation, provenance review and qualification.

| Milestone | Technical scope | Acceptance evidence |
|---|---|---|
| Persistent CUDA/TensorRT execution | RAII session owns engine/context, stream, workspace and bounded reusable buffers; separate cold/session/per-cloud timing; explicit reset/failure isolation | Both existing model packages give unchanged labels; context reuse tested over differing cloud sizes; bounded allocations; cold and warm complete timings |
| Streaming without LAZ | Typed C++ point-batch API includes source IDs, coordinates, optional features, CRS/frame and stream completion; input geometry contract versioned | Arbitrary chunk boundaries reconstruct exact source identity; finite buffer/spool budget; backpressure, cancellation and missing/duplicate IDs tested |
| Optional async LAZ/COPC export | Bounded producer/consumer queue, independent encoder workers, flush/failure receipt and deterministic joins; full order restoration for native LAZ | Inference latency separated from completion-to-durable-output; independent laz-rs verification; no point loss or silent export failures; COPC index permutation proof |
| Native GPU geometry/voxel processing | Generic GPU reductions, bounds, voxel assignments, feature sums/counts and deterministic ordering under an explicit contract | Compare current CPU geometry and actual tensors bytewise where exact, otherwise justified numeric tolerances; boundary/overflow/overlap/rare-voxel tests; profile complete application |
| LitePT-S / TorchSparse++ | Separate sparse adapter describes coordinates, active voxels, batch indices, topology and sparse ops; pin original alternative replacing spconv, audit rights and supported operators before import | No dense-adapter compatibility claim; reference full-network logits/point predictions, operator edge cases and engine/plugin portability; no Python/PyTorch native inference dependency |
| Jetson Thor target | Inventory actual allocated target, architecture, OS/JetPack/CUDA/TensorRT, storage and thermals; compile target-native runtime and engines from frozen ONNX | Both models, independent point/metadata preservation, numerical tolerances, sustained thermal/power measurements, cold/warm complete throughput, relocatable install |

Persistent execution comes first because the accepted cached-context benchmark shows model
execution is a small portion of current process-level time. It does not establish a live
streaming API or include LAZ input/output. Keep independently measured cold, resident and
export-inclusive throughput boundaries. Reuse the frozen fixtures as regression oracles.

Streaming must resolve current cloud-minimum block anchors and whole-voxel feature means
before claiming identical results. Choose either declared fixed world-space anchors or a
two-pass/spooled finite-cloud mode, with explicit end-of-block rules. A newly anchored live
contract is a behavior change and requires new model qualification. Defer publication of a
streaming rate until its whole application has actually run.

The existing [sparse plan](LITEPT_SPARSE_PLAN.md) records the TorchSparse alternative's
operator scope and incomplete learned-network parity. A Python wheel replacing spconv is
not sufficient for a native sparse TensorRT adapter. Unclear overlay licensing remains an
import blocker. Start with an original small sparse model/operator graph and independent
licensing; implement LitePT-S only after its own rights, topology and inference contracts
are established. This is a separate project from dense U-Net compatibility.

For Thor, NVIDIA documents a Blackwell-based target and its own JetPack software family;
consult the [official platform](https://developer.nvidia.com/embedded/jetson-developer-kits) and
[current Jetson documentation](https://docs.nvidia.com/jetson/archives/) at execution time.
Select toolchain/architecture from that physical target, not from the existing SM87/SM89
builds. [TensorRT portability constraints](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html)
require a new compatibility assessment and target-native engine builds; no RTX/Orin plan
reuse is promised. Thor qualification is NOT_RUN. Existing AGX Spatial Lab reservations
remain independent and must never be displaced to perform these proposals.
