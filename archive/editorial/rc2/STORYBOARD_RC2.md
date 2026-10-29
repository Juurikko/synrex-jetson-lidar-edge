# Streaming R2 RC2 storyboard and reproduction

Copyright 2026 Tuomas Pietilä. Original presentation CC BY 4.0; underlying IGN/FRACTAL information Etalab 2.0. Model and dependencies retain separate notices.

49 seconds, 1080 × 1350, 30 fps, H.264/yuv420p. Silent, suitable for autoplay. All motion starts immediately; there are no static title cards or fades.

| Time | Picture and camera | On-screen claim and limits |
|---|---|---|
| 0–4 | Full-frame, low oblique lateral/forward track past genuine urban points | First-frame “UP TO 2 MILLION LiDAR points / second”; “Native GPU classification + semantic mapping”; RTX 4090, recorded replay, qualified scene-A 60-second scope |
| 4–8 | Close lateral pass past real roofs and tower; visible depth and occlusion | Actual frozen U-Net predictions; consistent seven-class palette |
| 8–11 | Close forward/lateral vegetation pass in scene C | Original geometry, no reconstructed surfaces |
| 11–14 | Short controlled orbit of second urban scene, including real crane geometry | Same frozen model, different real geometry; errors retained |
| 14–25 | Moving aligned wipe and reverse wipe; identical points and perspective on both sides | Input uses height shading; output uses native trained-model semantic IDs. Held-out mIoU 48.756%; no ground truth impersonates output |
| 25–28 | Genuine full native map capture, continuously advancing ordered events and moving low camera | 4.16 s native replay shown in 3 s by uniform event-count retiming, about 1.386× total-duration acceleration |
| 28–33 | Detail replay 1, crop at actual event-activity density peak, controlled low orbit | Same capture; 14 m XY crop; outside-crop inactive event intervals skipped; emulated replay clock disclosed |
| 33–38 | Detail replay 2, different active crop, controlled low orbit | Same capture; white outlines=newly inserted cells, cyan=updated existing cells; reobservations need not change class |
| 38–41.25 | Moving point geometry behind translucent scope/metrics card | Up to 2.0M/s, scene A, two qualified 60 s tests; 1.5M/s qualified dense and full-churn tests; fresh classifications and map observation updates |
| 41.25–45 | Same continuous moving geometry; separate stability card | 600,014,848 fresh points in 10 minutes; 100% coverage; zero drops; approximately 1.0M/s, separate from 2M tests |
| 45–49 | Close moving urban points, legible repository end card | Native C++ / CUDA / TensorRT; MIT BYOM foundation; R4 public, streaming R2 not yet public; model/data licences separate |

## Rendering and authentication

The reproducible headless renderer is `tools/streaming_r2_rc2.py`. This is an inspectable deterministic CPU perspective renderer with a true per-pixel depth buffer, not a desktop QGIS session. Numba compiles the raster loop. All original cloud points are submitted; only normal projection/clipping and occlusion remove pixels. No meshes, guessed labels, generative imagery, added scenery, confidence or sensor motion are used.

The two urban views use independently reverified R4 source/semantic sidecars/output bindings present in the RC1 archive: `TEST-0438_6393-002595911` (83,641 points) and `TEST-0741_6244-006630086` (85,222 points). Scene C is `TEST-0944_6564-005900112` (400,654 points). Source XYZ is translated only by a recorded local origin for viewing. Camera movement is illustrative, not an acquisition path. Building and vegetation errors are preserved.

The map is rebuilt in original order from 400,654 accepted native delta events. There are 24,458 first insertions, 376,196 reobservations, and 6,510 updates that change semantic ID. No eviction occurs in this capture. Cell occupancy is never interpolated. Literal 1 m cell centres are displayed using 0.84 m cubes with visible gaps, shaded faces in the same semantic palette and event-type edge annotations. Outlines persist over a recent 9,000-event ordinal window; this is a display annotation, not sensor time, confidence or lifetime. A cell is white only if first inserted within that recent window; reobservations are cyan. The displayed counters cover the cumulative capture, not just the crop. Detail passes repeat recorded events and do not imply a second run or new sensor data. Retiming is ordinal-based; individual native wall-clock event timestamps are not available. Emulated timestamps are not sensor GPS time.

Historic diagnostic JSON p99 **274.477963 ms FAIL** remains intact. It is correctness media, not an accepted throughput/latency benchmark. Qualified measurements come from the unchanged accepted R1 receipts bundled inside the RC1 baseline. This revision runs no inference, training, optimization or new performance benchmark. No live sensor, Jetson AGX or Thor qualification is claimed.

## Reproduce

1. Work in the RC2 worktree. Preserve original RC1 archive SHA256 `a73aa7ce4b4bfd27377bf2d691e213bba70881ca5fefe817c3f460214b650805` and original MP4 SHA256 `4840f129c253c903ec4141932a5536d68cb27fd83ac6db0f7223078c7c786725`.
2. Extract a copy of that archive under `build/streaming-r2-review/streaming-r2-rc2/inputs/`. Do not alter RC1 source assets.
3. Create `.venv`; install `requirements-streaming-media-rc2.lock`. System DejaVu fonts are required; their hashes are in the provenance receipt. No font/encoder binaries are redistributed.
4. Run `.venv/bin/python tools/streaming_r2_rc2.py --stage proof`. It checks all 648 baseline manifest payloads, independently reconstructs predictions/map snapshots, checks semantic bindings, and renders a 4-second opening proof and representative stills.
5. Inspect the decoded proof before setting `OPENING_PROOF_RC2.json` status to PASS with an explicit review record. Run `--stage render` for the complete film. Camera positions, reveal coordinates, crop centres, event stops and text boxes are recorded for every frame.
6. Run `.venv/bin/python tools/streaming_r2_rc2_qa.py`. It also performs full-rate headless playback and validates the preview. Review the decoded contact sheets, mobile reduction and representative decoded frames. Record the visual review in `VISUAL_REVIEW_RC2.json` and run QA again with `--reuse-decoded` to reuse the completed hash-bound decode.
7. Local source/evidence commits and candidate tag only. Run `.venv/bin/python tools/streaming_r2_rc2_package.py` to create a complete owner-review archive with manifests and a source bundle. Review `FINAL_REPORT_RC2.json`. Publication remains NOT_RUN.

The source receipt seal is `eb4cab0`, RC1 candidate source is `e33679c`, and accepted native source is `f39fa29`. `CAMERA_FRAMES_RC2.json` binds each of the 1,470 frames to its exact illustrative camera. Visual provenance includes hashes of the renderer, font, source, model, native events, predictions and derived display cache. The complete baseline and its historical failures remain in `inputs/streaming-r2-rc1/` within the owner archive.

<!-- Copyright (c) 2026 Tuomas Pietilä. SPDX-License-Identifier: CC-BY-4.0. Original expression only; facts and upstream material excluded. -->
