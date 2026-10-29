# Streaming R2 RC4 — RTX 4090 software replay research prerelease

<!-- Copyright (c) 2026 Tuomas Pietilä. SPDX-License-Identifier: CC-BY-4.0. Original expression only; upstream material and facts excluded. -->

Native C++ / CUDA / TensorRT fresh classification and bounded semantic mapping,
with source, tests, factual receipts and the approved unchanged 37-second film.
This is a sanitized public snapshot with a new public commit descending directly
from public main. It is not a public tag pointing at the qualified private commit.
The source-equivalence manifest describes every frozen tracked file.

| Frozen result | Scope and source |
|---|---|
| ~1.9996M fresh points/s | Two accepted 60-second scene-A RTX replay trials; [CSV](../../evidence/streaming_r1_transport_trials.csv) |
| ~1.5M/s dense/full churn | Accepted 60-second trials; same CSV; export disabled |
| 600,014,848 points in 600.026055098s | Separate ten-minute full-churn run, ~1M/s, 100% coverage, zero drops; [receipt](../../evidence/streaming_r1_long_stability.json) |
| 39.845888ms p99 | Replay source to validated subscriber in that separate endurance run; no sensor/Jetson latency claim |
| 48.756% held-out seven-class mIoU | Frozen 62,103-parameter U-Net; [independent matrix](../../evidence/unet_r2_independent_metrics.json) |

Replay source: each workload loops one finite licensed FRACTAL tile in 16,384-point
frames. Scene A has 50,096 records (about 40 repetitions per second at 2M/s); scene C has
400,654. Full-churn runs, including the ten-minute run, loop scene A and shift each frame
by 64 m in x. Every pass runs fresh inference, but the rates describe pipeline throughput
on repeated input, not distinct points. Accuracy at streaming-window context was not
separately evaluated.

The small U-Net has visible accuracy limitations. These are scene/model-specific
engineering results, not survey-grade accuracy or state-of-the-art claims. The
historical JSON diagnostic **274.477963ms p99 FAIL** and harder **2M dense/churn
FAIL** trials remain intact. Optional LAZ/COPC export has separate verification
and is outside headline throughput. Film playback is not benchmark timing.

The lean assets are sanitized source, audit/reconstruction/rights/claims records,
release notes, checksums and the unchanged film. No private bundle, unfiltered
owner-review archive, model/data payload, SDK, font or encoder binary is uploaded.
The separately licensed frozen model/examples remain available from the
[unchanged public R4 release](https://github.com/Juurikko/synrex-jetson-lidar-edge/releases/tag/byom-unet-fractal-r4-rc2).

Original source: explicitly scoped MIT. Original presentation/docs: CC-BY-4.0.
IGN/FRACTAL information: Etalab-2.0, source updated 2025-04-05, dataset revision
`36e00245ef92fd180f655f6ee410f2e935693cc2`, retrieved 2026-10-26.
Architecture, trained model, external libraries, fonts and NVIDIA components keep
their own notices. See [audit](PUBLICATION_AUDIT.md) and [third-party notices](../../THIRD_PARTY_NOTICES.md).

Developer/project lead: Tuomas Pietilä, with OpenAI Codex assistance under owner
specifications and review. Future onboard UAV research on Jetson Thor and AGX Orin
is conditional on physical allocation and separate validation. No such hardware,
live sensor, flight or pose measurement is asserted by this prerelease.

Start with the [public source quickstart](QUICKSTART_RC4.md). Earlier candidate
publication states in preserved receipts describe the historical qualification;
they are not rewritten into fresh passing receipts.
