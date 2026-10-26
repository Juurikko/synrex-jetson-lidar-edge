# Physical AGX Orin qualification plan after RTX4090 qualification

<!-- Copyright (c) 2026 Tuomas Pietilä. SPDX-License-Identifier: CC-BY-4.0. Original expression only; upstream material and facts excluded. -->

This is an execution plan, **not AGX execution evidence**. No AGX connection, process change,
stack installation, power/clock change or GPU allocation was performed during the RTX4090 work.
The accepted CPU checkpoint `eaf3ce2` remains an ancestor of the qualified improvements.

## Entry gate: explicit nonconflicting allocation

The Spatial Edge Lab autopilot may be active. An operator must supply an allocation identifier,
device identity, permitted time window and nonconflict confirmation for CPU, RAM, GPU and disk.
An idle utilization sample alone is insufficient. Do not stop, suspend, renice or restart an
existing workload to create availability. If the allocation cannot be established, mark all
physical AGX gates NOT_RUN and retain the existing system unchanged.

Only after that gate, capture read-only target facts: aarch64/Linux identity, existing Jetson/L4T,
CUDA and TensorRT versions, free RAM/disk, active processes, thermal state and current power/clock
settings. Keep operational paths/addresses in private logs. Do not change the Jetson software
stack, clocks, power mode or thermal configuration. Confirm that the complete run fits inside
the window and stop launching new work if an allocation conflict appears.

The desktop runner's `nvidia-smi` resource probe is **not an AGX resource-allocation mechanism**.
If that utility/query is unsupported on Jetson, prepare a Jetson-specific read-only inventory
and resource guard under the operator's allocation before executing the runner. Do not disable
the guard or pretend missing GPU telemetry is an idle reading.

## Exact source and target-native build

Transfer a Git bundle of the qualified descendant of `eaf3ce2` through the approved private
channel, verify the bundle SHA256 and Git revision, and check out that exact revision in a
new directory. Include pinned laz-perf/json sources and notices. Verify the worktree is clean.
Never transfer private reference source, pretrained weights, datasets or the RTX4090 engine.
For the separate trained U-Net qualification below, transfer only the original
new pilot ONNX/weights and bounded Etalab-2.0 test examples with their notices;
the exclusion above applies to restricted or unreviewed external assets.

Use existing native aarch64 CUDA/TensorRT 10.x development installations and OpenSSL 3 headers.
An x86_64 binary or SDK wheel from the RTX4090 workspace is not an AGX installation artifact.
Missing development prerequisites are blockers to report, not permission to change the system.

```bash
cmake -S . -B build-agx -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSYNREX_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87 \
  -DLAZPERF_SOURCE_DIR="$PWD/.deps/laz-perf" \
  -DNLOHMANN_JSON_INCLUDE_DIR="$PWD/.deps/json/single_include"
cmake --build build-agx -j 2
ctest --test-dir build-agx --output-on-failure
build-agx/synrex-native target
```

Confirm the executable is aarch64, SM87 matches the actual GPU, and the native dependency graph
contains no Python/PyTorch. Record the real SDK/runtime identities, binary hashes, compiler
flags and source revision. Engine-package target metadata must describe the AGX, not the RTX4090.

## Genuine inference qualification

Use an isolated development venv only for original fixture export, oracle and laz-rs verification;
the classifier must run as a native process. Export the analytic Conv3D graph from committed
source, check it with ONNX, and build/deserialise a new TensorRT engine on the AGX. Do not copy
or reuse an RTX4090 plan or accept a target-contract override.

Execute the equivalent of `tools/qualify_dense_gpu.py` with the verified allocation and adapted
resource probe, using at least 50,013 points. Require all of the following:

1. Real successful TensorRT `enqueueV3` completion for every nonempty block, with positive CUDA
   event intervals for feature preparation, inference and reconstruction.
2. Exact independent ONNX-oracle labels for all 50,013 immutable source indices; nonconstant
   results exercise LAS classes 2, 5 and 6. Preserve the raw-intensity class histogram.
3. Independent laz-rs decode checks of original point order, all 41 unrelated record bytes,
   public header, non-LASzip VLRs, Extra Bytes descriptors and metadata padding.
4. At least two native invocations with identical labels and classified LAZ hashes. Repeat one
   with no Python executable available through PATH and inspect loaded native libraries.
5. Bad tensor name/shape, invalid class mapping, mismatched target, corrupt SHA256/engine,
   unsupported PDRF and output-overwrite cases all reject without publishing output.
6. The complete opt-in GPU suite passes, including first/last overlap policy, intensity reduction,
   FP32/FP16 I/O, alternate grid and relocated native classification.

Compare expected labels and preserved point records across targets. TensorRT plans are expected
to differ by target; do not require engine byte identity. Investigate any numerical class change
against the independent oracle before accepting it. Historical SegFormer outputs and benchmarks
are excluded from this qualification.

## Timing and receipts

Capture decode, geometry, CUDA preparation, TensorRT inference, CUDA postprocessing, LAZ encoding
and full invocation separately using the committed native instrumentation. CUDA-event values are
stream stage intervals; CPU/writer/end-to-end values are wall-clock durations. Initialization,
transfers and orchestration remain included in end-to-end time. Preserve every raw run rather
than reporting only the fastest result. Record power/thermal context without changing it.

Install into a new prefix, relocate the install and engine package, and rerun classification with
system vendor dependencies. Verify no build-tree RUNPATH, Python fallback or private artifact
dependency. Independently verify the relocated output as well.

Save sanitized receipts, source/dependency hashes, independent verification and explicit PASS,
FAIL or NOT_RUN gates. Preserve the allocation record privately. Finish within the allocation;
leave the autopilot, software stack and pre-existing files intact. Only after these checks can
physical AGX qualification be marked PASS. Publication remains a separate owner-approved action.

## Second-model extension after RTX4090 U-Net qualification

Use the qualified 62,103-parameter U-Net ONNX and exact source/data hashes from
[the U-Net report](UNET_QUALIFICATION.md). Build a new SM87-compatible TensorRT
FP16 engine locally; the RTX4090 plan is incompatible. Revalidate actual FP16
layers, logits/point-label agreement and all seven explicit mappings. Use the
same installed binary for original Conv3D and U-Net, recording its unchanged hash.
Verify genuine real PDRF8/38-byte held-out LAZ with laz-rs, including metadata,
every unrelated byte, source order and repeated labels. Re-run all modern-format
and negative tests plus relocated inference.

Record target-specific latency, throughput, sampled memory, thermal context and
FP16 numerical drift. The small desktop pilot does not establish AGX performance
or reliable bridge/water accuracy. Reuse immutable trained artifacts for cross-target
comparison; training on the AGX is unnecessary. Allocation is still a prerequisite,
and AGX execution remains NOT_RUN until this entire extension actually executes.
