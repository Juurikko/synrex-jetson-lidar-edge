# Qualified file-to-classified-file measurements

<!-- Copyright (c) 2026 Tuomas Pietilä. SPDX-License-Identifier: CC-BY-4.0. Original expression only; upstream material and facts excluded. -->

**Developer and project lead: Tuomas Pietilä ([@Juurikko](https://github.com/Juurikko)).**

These figures are transcribed from the accepted physical-device evidence recorded on 24 October 2026, at source commit `69eca500738b0412d1a2a0b6ba0edeac20012aec`. No target benchmark was rerun while preparing this repository. The [machine-readable record](../evidence/benchmark.json) includes the full recorded precision and frozen output digests; the table presents time rounded to six decimals.

| Scene | Points | Full product seconds | Points/s, rounded | Independent verification |
|---|---:|---:|---:|---|
| Test file 1 | 468,416 | 6.644378 | 70,498 | PASS |
| Test file 2 | 886,928 | 6.237315 | 142,197 | PASS |
| Test file 3 | 1,308,277 | 7.280629 | 179,693 | PASS |
| Test file 4 | 2,126,455 | 10.110652 | 210,318 | PASS |
| **Total / aggregate** | **4,790,076** | **30.272974** | **158,229** | **PASS** |

Aggregate throughput is **sum(points) / sum(elapsed seconds)**, not the unweighted average of scene rates. The 210,318 points/s headline is the largest tested file's result, not a sustained or four-file average. The exact machine-readable summed time is 30.27297449300204 seconds; rate rounding uses the unrounded observations.

## Hardware and operating profile

- Physical NVIDIA Jetson AGX Orin 64GB, aarch64, SM87 / compute capability 8.7.
- CUDA 13.2.86, TensorRT 10.16.2.10, engine internal FP16.
- **MODE_50W**. This is a selected operating profile, not a measured 50 W power draw or energy-per-point result.
- Qualified target-native engine and existing relocatable Product Runtime R1 reused without rebuilding or tuning for these runs.

## Timing boundary

A parent monotonic clock starts immediately before product launch and stops after process exit. The interval includes loading, model package verification/extraction, TensorRT initialization, native geometric preprocessing, GPU occupancy/inference/reconstruction, class-label production, classified-LAZ writing and product output hashing. Inputs and package were prehashed before the interval; independent offline verification is a subsequent check and is not included in the product time.

Each file was executed once in its own process. There were no warmups, repetitions or cache flushing. Normal OS file caching was allowed. This is neither controlled cold-cache performance nor an always-warm inference benchmark. Inference-service internal timing excludes launcher/package/writer work and is not substituted for the full-file measurement.

## What the numbers do not establish

No confidence interval, sustained real-time streaming rate, guaranteed latency bound, concurrent workload result, flight-to-map time, onboard sensor integration, radio throughput, energy efficiency or thermal endurance is inferred. There is **no world-record or state-of-the-art claim**, and no same-device speedup against another model is asserted.

The frozen engine emitted a device-plan portability warning during the accepted executions. It nevertheless produced the exact qualified outputs on this target. This result does not certify engine portability across GPU/device models; broader target support requires its own build and qualification.

The current model is an evaluation/reference artifact derived from TreeAIBox and treated as noncommercial. ECLAIR inputs and their genuine predictions are excluded. The repository supplies factual measurements and synthetic illustrations, not restricted benchmark data or a license to redistribute it. Owner review of intended reporting under those terms remains a publication gate.

## Reproduction boundary

The public synthetic demo exercises LAZ preservation and independent verification on a CPU. It does not reproduce these timings, run neural inference or establish model quality. Exact hardware reproduction requires the qualified native product, appropriately licensed model/data, target environment and original protocol; those assets are not part of this repository.
