# Final RTX performance analysis

<!-- Copyright (c) 2026 Tuomas Pietilä. SPDX-License-Identifier: CC-BY-4.0. -->

Derived from the accepted [untraced receipts](../../archive/evidence/unet_r2_release_benchmarks.json),
without new GPU benchmarking. The [derived values](../../archive/evidence/unet_r3_performance_analysis.json)
are reproducible with `tools/analyze_unet_performance.py`. Cold means a fresh process,
not a reboot or guaranteed cold disk cache. Each scene has three cold runs, three
persistent warmups and ten measured persistent cycles. All repeated output labels
are identical. Native inference has no Python/PyTorch dependency.

All times below are means in milliseconds. Preprocessing is CPU geometry plus CUDA
feature reduction; LAZ decoding is listed separately. Reconstruction is the fused
CUDA argmax/class-map/source-index scatter stage. Peak is tracked requested CUDA
allocation, with device-wide sampled maximum shown separately.

| Scene | Points | Cold launch→exit | Persistent | Geometry / CUDA prep | TRT | Reconstruction | LAZ decode / encode | Tracked peak MiB / device sampled MiB | Complete points/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0447_6390 | 50,096 | 732.685 | 1.685 | 6.587 / 2.130 | 0.942 | 0.0256 | 22.987 / 35.983 | 6.295 / 526 | 68,373 |
| 0922_6376 | 97,258 | 440.536 | 1.336 | 12.796 / 1.420 | 0.812 | 0.0262 | 36.554 / 57.648 | 8.539 / 518 | 220,772 |
| 0944_6564 | 400,654 | 989.948 | 4.595 | 56.829 / 1.928 | 1.400 | 0.0573 | 219.600 / 369.693 | 22.886 / 524 | 404,722 |
| 0947_6571 | 76,649 | 461.534 | 1.329 | 10.174 / 1.427 | 0.817 | 0.0262 | 43.748 / 72.615 | 7.645 / 518 | 166,074 |

Aggregate complete-process throughput is **237,991 points/s**:
624,657 points /2.624704 seconds (sum of mean scene durations). With equal repeat
counts this equals1,873,971 processed points /7.874112 measured seconds. This is
point-weighted throughput, not the arithmetic mean of scene rates. Cached
persistent processing is69.830 million points/s across these scenes, excluding
LAZ I/O and cached geometry. It must not be advertised as complete throughput.

Engine/context initialization plus initial upload takes230–298ms. The first small
scene invocation takes1,376ms launch-to-exit, versus410–412ms repeats; it remains
included in the732.685ms mean. Native main-entry E2E excludes loader/teardown,
whereas the complete process timer includes both. Large-scene LAZ decode/encoding
accounts for589ms out of990ms complete latency, compared with1.400ms TRT and
.0573ms CUDA reconstruction. Cold initialization/loader/teardown dominate smaller
scenes; LAZ I/O dominates larger scenes. These receipts do not justify optimizing
the neural graph before I/O/session reuse. Host orchestration/transfers/hashing
are in complete time but not all individually instrumented. CUDA stage times are
stream event intervals; CPU/process times are monotonic wall durations.

Tracked requested allocator peaks range6.295–22.886MiB and exclude CUDA context
and vendor allocations outside callbacks. Device-wide sampled maximum is518–526MiB,
including the130MiB desktop baseline. Sampling around60–80ms may miss a transient
resident peak. Neither figure is an exact total resident GPU memory high-water mark.

The earlier [SegFormer3D AGX receipt](../../archive/evidence/benchmark.json) records
**210,318 points/s** for its2,126,455-point largest scene and **158,229 points/s**
aggregated over4,790,076 points/30.272974s. The dense U-Net RTX largest scene is
400,654 points at404,722 complete-process points/s; its aggregate is237,991.
These are observations from different experiments, **not a controlled model
comparison or evidence of an architectural speedup**. Differences include RTX4090
x86_64 WSL versus AGX Orin ARM64/SM87 at historical50W mode; tiny62,103-parameter
seven-class U-Net versus the earlier SegFormer graph and ontology; FRACTAL50m
patches versus different larger historical scenes;32³/4-feature dense inputs
versus the historical fixed representation; TensorRT builds, host/codec/storage,
initialization and timing boundaries. Historical E2E timing does not establish
identical launch/teardown accounting. No synthetic Conv3D or persistent-only rate
is used for that comparison. No new AGX benchmark is claimed.
