<!-- Copyright (c) 2026 Tuomas Pietilä. SPDX-License-Identifier: CC-BY-4.0. -->
<!-- DRAFT ONLY. Use only after owner publishes Streaming R2 and replaces the release placeholder. RC4 is local today. The film truthfully retains its local-candidate disclosure. -->

Two million LiDAR points per second—freshly classified, with map observations updated.
Here’s what I measured, and what I want to explore next.

I built SynRex’s native C++/CUDA/TensorRT pipeline with persistent inference, versioned TCP replay, per-point predictions, incremental bounded semantic-map updates and optional GIS-ready LAZ/COPC export.

On RTX 4090, using prerecorded IGN/FRACTAL LiDAR replay:

- Up to ~2M fresh points/s: scene A, two qualified 60-second trials.
- ~1.5M/s: separately qualified dense-scene and full-churn trials.
- 600,014,848 classifications: a separate ten-minute full-churn run at ~1M/s, 100% coverage, zero drops, 39.85 ms p99 replay-to-validated-output latency.

These are pipeline results with map observations, not inference-only speed or live drone capture. The 62,103-parameter seven-class U-Net achieves 48.756% held-out mIoU: a systems demonstration, not survey-grade accuracy.

If porting and sustained onboard qualification succeed on Jetson AGX Thor—or Jetson AGX Orin at potentially lower throughput—a UAV could classify georeferenced LiDAR, incrementally map inspection/change/context information and export results before landing. That’s an R&D opportunity, with no demonstrated Jetson throughput or flight result. Power, bandwidth, CPU geometry, transport, positioning, thermals and accuracy remain decisive.

I developed this with OpenAI Codex assistance. Original native code is MIT within its documented scope; model/data/third-party rights remain separate.

Public R4 foundation: https://github.com/Juurikko/synrex-jetson-lidar-edge

Streaming R2: [ADD VERIFIED STREAMING R2 RELEASE URL AFTER OWNER PUBLICATION]

#LiDAR #Robotics #EdgeAI #Geospatial
