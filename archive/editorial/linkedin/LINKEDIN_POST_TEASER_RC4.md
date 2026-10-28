<!-- Copyright (c) 2026 Tuomas Pietilä. SPDX-License-Identifier: CC-BY-4.0. -->
<!-- DRAFT ONLY. Suitable before streaming source publication; no upload or post authorized. -->

600 million fresh LiDAR classifications in ten minutes.
The interesting part is keeping the whole pipeline honest.

I’m preparing SynRex Native LiDAR Streaming R2, my C++/CUDA/TensorRT implementation with persistent inference, versioned TCP replay, fresh per-point semantic predictions, incremental bounded map updates and optional GIS-ready LAZ/COPC export.

Measured on RTX 4090 with prerecorded IGN/FRACTAL replay:

- Up to ~2M fresh points/s with map observations: scene A, two qualified 60-second trials.
- ~1.5M/s in separately qualified dense-scene and full-churn trials.
- A separate ten-minute full-churn run: 600,014,848 classifications at ~1M/s, 100% coverage, zero drops and 39.85 ms p99 replay-to-validated-output latency.

The film shows genuine geometry and trained-model predictions. Camera motion is illustrative; this is replay, not live UAV capture. The small 62,103-parameter seven-class U-Net scores 48.756% held-out mIoU. It demonstrates the system; it is not survey-grade or state-of-the-art accuracy.

If a port can sustain onboard operation on Jetson AGX Thor—or Jetson AGX Orin at a potentially lower practical rate—classifying georeferenced LiDAR and exporting incremental semantic maps before landing becomes an exciting research direction for inspection, change detection and obstacle/context perception. Neither platform’s streaming throughput nor flight operation is demonstrated. Power, memory/sensor bandwidth, CPU geometry, transport, SLAM/GNSS/IMU, battery/thermals and accuracy all need qualification.

The R4 BYOM foundation is public: https://github.com/Juurikko/synrex-jetson-lidar-edge

Streaming R2 source and this RC4 film remain a local candidate under review. No streaming release URL yet.

I developed SynRex with OpenAI Codex assistance. Original native code is MIT within documented scope; model/data/third-party rights are separate.

#LiDAR #UAV #Robotics #EdgeAI #Geospatial
