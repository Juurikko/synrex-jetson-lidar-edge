# Streaming R2 RC1 — engineering showcase

<!-- Copyright (c) 2026 Tuomas Pietilä. SPDX-License-Identifier: CC-BY-4.0. -->

**Local release candidate for owner review. No public push, upload, release or LinkedIn
post has been made.** R4 RC2 is already published; this is a separate, versioned streaming
candidate. Its source bundle includes the complete accepted history and historical FAILs.

The qualified native runtime is **`f39fa296332217eb77bd73dd7e1e3515be3f367e`**. R1's final
evidence seal is **`14bcb4f2c512d735291f796ac905584564e74ddf`**. Later R2 commits add audit,
packaging, verification and media tooling/documentation, without changing any native
runtime file, model weight, preprocessing arithmetic or accepted benchmark receipt.
The candidate's exact packaging commit and every asset hash are in `RELEASE_MANIFEST.json`.
R4 remains `byom-unet-fractal-r4-rc2`, peeled source
`3f1f3fabd3d92d69c02394936188502678a6b9ed`.

## Included source and assets

- Native C++17/CUDA/TensorRT persistent inference; two-frame queues and explicit failures.
- Binary v2 loopback transport, deterministic native sender, receiver and subscribers.
- Complete fresh original-point classifications, separate map deltas and snapshots.
- Bounded revision-preserving intrusive eviction index, with switchable reference implementation.
- Atomic asynchronous first-observation LAZ bundles and optional offline verified COPC/QGIS.
- CPU, sanitizer, native GPU, numerical, identity, map, transport and export regression tooling.
- Frozen synthetic Conv3D and trained U-Net packages; frozen checkpoint/ONNX for target rebuilds.
- Licensed A/B/C source fixtures, genuine native example outputs, COPC projects and R4 screenshots.
- A **54-second, 1080 × 1350, 24 fps** video, opening frame, thumbnail and contact sheet.
- Real recorded scene-C events, exact verified map states, coverage/provenance and rights manifests.

Large model/data/video artifacts are separate release assets, absent from normal Git history.
No vendor SDK, compiler, QGIS/Qt executable, bulk dataset or restricted private source is shipped.
The package is **not** under a blanket MIT licence; see [rights](STREAMING_R2_RIGHTS.md).

## Measurements retained without alteration

| Native external TCP workload | Accepted bounded result |
|---|---|
| Stable scene A, two 60 s trials | 1,999,600.512 / 1,999,582.696 fresh points/s; p99 30.933 / 31.457 ms |
| Dense scene C, two 60 s trials | Approximately 1.5M fresh points/s, complete map observations |
| A full churn, 4,096 and 65,536 cells, two 60 s trials each | Approximately 1.5M fresh points/s |
| Ten-minute A/65,536 full churn | 600,014,848 fresh points / 600.026055098 s = 999,981.322/s |
| Ten-minute coverage / drops | 100% / zero |
| Ten-minute latency p50 / p95 / p99 | 30.4087 / 37.7487 / 39.845888 ms |
| Ten-minute receiver peak RSS / GPU peak | 456.97 MiB / 524 MiB |
| Ten-minute average utilization / power | 10.796% / 9.330 W, whole-trial sampled scope |

The ten-minute test is **approximately 1M/s**, not 2M/s. Dense/churn external TCP trials
at 2M/s remain **FAIL**, including their warmup/measured failure phases. Accepted earlier
in-process 3M A/B, 2.5M dense C and 2M full-churn results are preserved with their separate
measurement boundary. Small-window queue failures, pinned-staging negative results,
out-of-order failures and unsupported-filesystem atomic-export failures remain intact.
No new sustained-rate campaign was run for R2.

The trained U-Net retains its separately qualified **48.756% seven-class mIoU** on
256 held-out tiles / 24,902,618 points. It is a 62,103-parameter bounded engineering model;
these performance numbers do not generalize to larger networks, arbitrary scenes or
survey accuracy. Software replay is not live physical sensor capture.

## Review gates and limitations

Fresh source reconstruction, CPU/sanitizer tests, native two-model inference, 31-window
byte-exact trace parity, TCP smoke, asynchronous LAZ preservation, COPC source-index
verification, visual provenance and video decoding have explicit R2 receipts.
[Audit](STREAMING_R2_AUDIT.md) · [media QA](STREAMING_R2_MEDIA.md) ·
[reproduction](STREAMING_R2_REPRODUCIBILITY.md).

QGIS standalone provider/map export passes on all 400,654 scene-C source indices.
Two fresh desktop startup attempts stalled before the code worker: those attempts are
retained as FAIL. The video labels the successful output as a **QGIS map export**, not
a fresh desktop screenshot. Genuine separately verified published R4 desktop screenshots
and projects are also supplied. Scripted 3D rasterization provides the fly-through;
fresh QGIS 3D frame export is NOT_RUN.

The media-only JSON recording has 274.478 ms p99 acceptance-to-file-output latency:
**FAIL against 250 ms**, including large JSON writes. It is a short correctness recording,
not a new rate qualification, and is never used to replace accepted TCP measurements.

The map has no TTL, calibrated confidence or semantic fusion. It retains last accepted
observations, with explicit late-observation handling. One ordered identity/subscriber,
loopback-only unauthenticated transport, complete-frame delivery, no late-join resume,
explicit transform requirements and the original supported LAS layout remain the scope.
Physical sensor, AGX Orin and Thor inference are **NOT_RUN**. No AGX/Thor access occurred.

Publication is **NOT_RUN pending owner approval**. Candidate engineering readiness does
not authorize publication. No accepted tag or public main is changed.
