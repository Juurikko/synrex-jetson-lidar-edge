# LinkedIn engineering demonstration — drafts only

<!-- Copyright (c) 2026 Tuomas Pietilä. SPDX-License-Identifier: CC-BY-4.0. -->

**Not sent.** Review the release candidate and film first; public upload/posting requires
owner approval. The current film is explicitly a review candidate. If publishing a final
cut, update its review/unpublished end card truthfully and rerun decoding/provenance QA.

## Post draft

How fast can a native LiDAR pipeline classify every original point while maintaining a
bounded semantic map?

SynRex Streaming R1 reached approximately **2 million freshly classified points/s** on
one real FRACTAL scene, in repeated 60-second RTX 4090 replay tests. A denser scene and
full-map churn qualified at approximately **1.5 million/s**. The failures at 2M in those
harder external-transport workloads remain in the evidence.

In a separate ten-minute full-churn run at approximately 1M/s: **600,014,848 fresh
classifications, 100% coverage, zero drops and 39.85 ms p99 replay-to-validated-output
latency**.

The runtime is native C++/CUDA/TensorRT: persistent model/context/buffers, bounded queues,
original point identities and timestamps, independent output validation, incremental
map deltas and optional atomic LAZ/COPC export. No Python/PyTorch runs in production inference.

The video shows genuine source geometry, actual frozen-model predictions, recorded native
map updates and real mistakes. Camera movement is illustrative; input is software replay,
not a live sensor feed. The small seven-class U-Net achieved **48.756% mIoU** in its separate
held-out evaluation. Throughput is not a claim of survey accuracy or larger-model performance.

Streaming R2 packages the engineering evidence and reproducible source for review.
The MIT source grant, model notices and IGN data licence have separate scopes. Physical
sensor, AGX Orin and Thor paths are not yet qualified.

Project: https://github.com/Juurikko/synrex-jetson-lidar-edge

Developed by Tuomas Pietilä, with OpenAI Codex assistance in engineering and verification.

#LiDAR #CUDA #TensorRT #PointCloud #Geospatial #Cpp #OpenSource

## First-comment draft

The public R4 RC2 foundation is already available. The separate Streaming R2 candidate
contains build/replay instructions, exact source/model/output hashes, source-indexed
COPC/QGIS examples, benchmark receipts, preserved FAILs and media provenance. Check the
release notes for publication status rather than assuming a local candidate is released.

The approximately 2M result applies to stable scene A's two 60-second external TCP runs.
The ten-minute full-churn run was approximately **1M/s**, not 2M/s. Fresh neural points,
map observations, changed-voxel deltas and file export are counted separately.

Data: IGN/FRACTAL, Etalab 2.0, pinned revision
`36e00245ef92fd180f655f6ee410f2e935693cc2`; source updated 2025-04-05T09:37:48Z,
retrieved 2026-10-26. Original presentation: Tuomas Pietilä, CC BY 4.0. Upstream
model-architecture, NVIDIA and QGIS/Qt notices remain separate. No bulk dataset or
model binary is committed into ordinary source history.
