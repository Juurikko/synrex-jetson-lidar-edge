# SynRex Native LiDAR BYOM MIT R0 — local RC1

<!-- Copyright (c) 2026 Tuomas Pietilä. SPDX-License-Identifier: CC-BY-4.0. Original expression only; upstream material and facts excluded. -->

Prepared GitHub release text; **local only, not published, not pushed**.

This candidate adds a configurable native dense-voxel classification runtime to the existing
MIT LAZ preservation utility. Users can export or supply a self-contained ONNX graph, build
a target-native TensorRT engine, package its tensor/geometry/feature/class contracts and replace
compatible packages without recompiling the runtime. Classification contains no Python or PyTorch.

Included: C++/CUDA source, CMake build/install, package validator/builder, TensorRT 10 engine
builder/executor, original analytic Conv3D model exporter, CPU and opt-in GPU integration tests,
explicit LAS class mapping, original source-index reconstruction and retained metadata-preserving
LAZ writer/verifier. Original source is MIT; documentation is CC BY 4.0; NVIDIA/Apache/other
dependencies remain separate. No private implementation or restricted pretrained artifact is included.

Validation: native CPU and CUDA/TensorRT builds pass. CPU Release and ASan/UBSan suites each
pass 50 cases; four physical GPU cases are skipped. GPU-linked CPU checks also pass.
**Physical TensorRT initialization, CUDA inference and BYOM LAZ end-to-end qualification are
NOT_RUN.** The AGX was not accessed because no nonconflicting allocation was verified. This
candidate is for source review and qualification; public release readiness is FAIL until those
gates pass. Historical SegFormer throughput is not a measurement of this runtime.

See [quickstart](QUICKSTART.md), [adapter](ADAPTER.md), [qualification](QUALIFICATION.md),
[licensing assessment](PROVENANCE.md) and the machine-readable `evidence/byom_*.json` receipts.
The local archive directory contains source and native install candidates with SHA256 checksums;
vendor libraries are excluded. No actual GitHub release, draft PR, remote tag or visibility change
has been created. Owner approval is required before any later push/publication.
