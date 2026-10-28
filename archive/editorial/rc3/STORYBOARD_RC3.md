# Streaming R2 RC3 — cinematic editorial revision

49 seconds, 1080 × 1350, 30 fps, H.264/yuv420p, silent. Media only. RC2 baseline is local tag `streaming-r2-rc2-qualified` / `8586a6d`. All generated payloads are under `build/streaming-r2-review/streaming-r2-rc3/`. No model, native runtime, benchmark, accepted contract or earlier release artifact is changed.

| Time | Actual image and movement | Text / evidence scope |
|---|---|---|
| 0–5 | Genuine classified urban cloud from frame zero; elevated three-quarter forward/lateral track | UP TO 2 MILLION LiDAR POINTS/S; native GPU classification + semantic mapping; RTX 4090 recorded replay; scene A sustained 60-second scope |
| 5–9 | Reverse-side closer urban track, roof planes, courtyard, walls and ground | Real source points; actual frozen-model predictions |
| 9–13 | Coupled elevated canopy track with exposed sloping ground | Actual forest recording predictions; no reconstructed surfaces |
| 13–17 | Restrained 10.3° crane-scene orbit with translating target | Seven classes, errors retained; 62,103-parameter 3D U-Net |
| 17–28 | Aligned moving height-shaded original XYZ → true predicted-class wipe and return | Same point coordinates and camera vectors on each side; height shading is an explanatory input display, not original RGB; 48.756% held-out mIoU |
| 28–34 | Real classified forest source points in 18 m XY crop around [25,25]; actual recent native delta cell centres drawn as thin 3px round annotations | Bounded 1 m semantic-map updates; cumulative new/reobservation counts, actual recorded emulated source timestamp; uniform event-count retiming of the 4.156687229-second diagnostic capture into 6 seconds |
| 34–40 | Same genuine capture replayed from its start with crop [15,35] and another coupled oblique camera | Repeat explicitly labelled; no new run, sensor trajectory or fabricated evolution; same timing method |
| 40–42.4 | Urban geometry keeps moving behind compact lower metrics card | Up to ~2M/s in qualified scene-A sustained 60-second RTX replay runs; ~1.5M/s accepted dense/full-churn trials |
| 42.4–45 | Continuous same real urban movement; separate stability figures | 600,014,848 fresh classifications in a separate 10-minute full-churn run, ~1M/s, 100% point coverage, zero drops |
| 45–49 | Clean classified urban hero, elevated oblique, subtle coupled motion, no cubes | SynRex Native LiDAR Streaming; C++ / CUDA / TensorRT; github.com/Juurikko/synrex-jetson-lidar-edge; R4 public, streaming R2 local candidate |

Every actual camera vector, downward pitch, scene, reveal boundary, map crop/event ordinal and source timestamp appears in `evidence/CAMERA_FRAMES_RC3.json`. The shot table and projection-based parallax evidence appear in `docs/CAMERA_TABLE_RC3.md` and `evidence/CAMERA_AND_PARALLAX_RC3.json`. Pitch is calculated from eye and target, not inferred from labels. All planned frames lie within 30–45° down. Targets translate with the camera; the scene is never made readable merely by retreating to a distant overhead viewpoint.

## Visual and temporal provenance

The deterministic CPU perspective point rasterizer has a nearest-depth buffer. It submits original source XYZ, translated only by the recorded local origin. It builds no surfaces, buildings, inferred occupancy or synthetic predictions. Scene clipping, normal per-pixel occlusion and the explicitly recorded map spatial crop are the only geometric exclusions. Semantic point colours use the unchanged verified seven-class palette. Display splats cover pixels, not invented physical volumes. Cameras are illustrative local view transforms, never acquisition poses.

Urban scenes use R4 independently verified source/output/sidecar/model bindings: `TEST-0438_6393-002595911` (83,641 points) and `TEST-0741_6244-006630086` (85,222 points). Forest `TEST-0944_6564-005900112` uses 400,654 actual short-recording neural predictions. The complete RC1 manifest is checked, the recording is independently reconstructed without neural execution, and all prediction bytes and every map snapshot are compared. Source/model/event/prediction/cache/font/renderer hashes are retained.

The map chapter deliberately juxtaposes genuine classified source observations with exact event annotations. The full source context is visible throughout the replay; its geometry does **not** pretend to accumulate over time. Only the outlined actual delta cell centres and cumulative counters advance. Cell outlines use white for a first insertion within the recent 2,200-event window, cyan for a reobservation. Reobservation does not imply a changed class. Markers are a 2D overlay on projected true cell centres and can be seen through points; no hidden-surface or uncertainty inference is claimed. The bounded map has 24,458 cells, 376,196 reobservations and 6,510 class-changing updates in this capture. Camera crop is 18 m XY radius, not an occupancy cap. Counters cover the entire capture, not only the crop. All 400,654 events are replayed in exact order in each pass. No event interpolation or invented temporal evolution occurs.

Onscreen replay time is derived from each original source point's recorded Unix-nanosecond timestamp (emulated clock starting at 1700000000000000000). Its roughly 2.4 s source-clock span is distinct from the capture's 4.156687229 s wall duration. Neither is physical sensor GPS time. Per-delta native wall-clock timestamps are unavailable; timing is therefore disclosed as event-count retiming, not exact wall-clock animation. Each six-second chapter stretches that diagnostic capture and the second repeats it.

Independent COPC checks restore original point order using `SynRexSourceIndex`, compare XYZ and semantic IDs to each COPC's own bound sidecar, and check preserved QGIS project EPSG:2154. Forest COPC labels differ at 9,939 source IDs from the separate JSON recording; they must not be substituted for film predictions. Both artifacts are preserved and checked against their own evidence. Explicit viewing CRS assignment is retained for the urban projects where headers lack CRS. New QGIS GUI playback is NOT_RUN; this film is a headless CPU perspective render. Preserved genuine QGIS views/projects remain in the baseline package.

## Claims and limitations

The trained seven-class 62,103-parameter FRACTAL 3D U-Net achieved 48.756% held-out mIoU. This is bounded proof of engineering, with visible model errors, not survey-grade or state-of-the-art accuracy. The semantic map is a bounded representation, not a photorealistic reconstruction. Observations, fresh predictions and map upserts are separate artifacts.

The preserved diagnostic JSON capture's p99 **274.477963 ms FAIL** remains in the archive. It supports media correctness, not accepted latency claims. The film claims no latency number, physical-sensor, Orin, Thor or airborne inference qualification. It performs no new inference, training or performance qualification. The separate ten-minute run is approximately 1M/s, never 2M/s. Original accepted benchmark receipts remain unchanged.

## Reproduction and gates

1. In the isolated RC3 worktree, create `.venv` and install `requirements-streaming-media-rc2.lock`. Put a **copy** of the verified complete RC1 payload at `build/streaming-r2-review/streaming-r2-rc3/inputs/streaming-r2-rc1/`. Never edit originals.
2. Run `.venv/bin/python tools/streaming_r2_rc3_qa.py` only after video creation; its `audit()` can be called separately before edits to reproduce RC2 frame extraction, actual pitch diagnosis and preservation checks.
3. Run `.venv/bin/python tools/streaming_r2_rc3.py --stage proof`. Decode the two-second camera proof and four-second ending proof; inspect multiple decoded frames plus representative stills. Document `PROOF_REVIEW_RC3.json` with renderer/video hashes. First map proof was rejected for empty margins and reframed before the accepted proof. Full render requires a PASS bound to the renderer hash.
4. Run `--stage render`, then `tools/streaming_r2_rc3_qa.py`. QA decodes all 1,470 native-resolution frames, checks exact byte count, dimensions/frame rate/codec/container duration, projected source-point parallax, per-frame cameras, text bounds, native deltas and COPC parity; it creates a complete reduced preview and performs headless realtime playback.
5. Inspect decoded final contact sheets, full-size key frames and 360px mobile opening; save explicit agent editorial review bound to final video hash. Automated motion/provenance does not substitute for that judgement. The review is not human owner acceptance, calibrated-display testing or GUI playback. Record limitations as NOT_RUN.
6. Commit RC3 tools/docs/evidence locally after gates. Build a source snapshot/bundle, complete owner-review archive and SHA256 manifests using `tools/streaming_r2_rc3_package.py`. Verify every compressed member before creating the final qualified **local** tag. Retain a clean worktree. No push, release, social publication or device contact.

## Rights

Original media tooling: scoped MIT; original presentation: CC BY 4.0. IGN FRACTAL information: Etalab 2.0, source update 2025-04-05. Model, architecture, dependencies and fonts retain separate notices. The complete baseline rights documentation is bundled. No blanket MIT claim is applied to data/model/presentation and no external renderer, encoder or font binaries are redistributed. Public R4 BYOM source availability does not imply public R2 streaming availability. PUBLICATION=NOT_RUN.

<!-- Copyright (c) 2026 Tuomas Pietilä. SPDX-License-Identifier: CC-BY-4.0. Original expression only; facts and upstream material excluded. -->
