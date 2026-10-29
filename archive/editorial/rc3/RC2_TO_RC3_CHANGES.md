# RC2 → RC3 editorial correction

The owner rejected RC2's low forward-facing views and oversized map cubes despite historical technical PASS. RC3 treats those as editorial failures and changes media only.

| RC2 defect | RC3 change | Evidence |
|---|---|---|
| Most cameras around 0–12° down; crane slightly upward; map detail 16–20° | Coupled translating targets and eyes, elevated three-quarter perspectives consistently within 30–45°; closer reverse-side urban views and restrained orbit | Shot diagnosis, exact per-frame vectors, camera pitch table and source-point parallax |
| Roof/ground planes flattened; vegetation forms horizon wall | Rooftops, courtyards, ground and walls in the same view; canopy seen from above and from the side; medium/close crop | Decoded proof and full-film sheets, RC2→RC3 comparison |
| Explicit 0.84m cube renderer fills 1m map occupancy into large green block walls at 25–38s | Cube renderer never called by RC3. Genuine classified source geometry carries small round actual-delta annotations; no mosaic, block explosion or replacement filler | Code dispatch, map event reconstruction, decoded late sequence |
| Metric card hides most of scene | Shorter five-second benchmark chapter and compact lower card preserve upper moving geometry | 40–45s decoded frames |
| Final camera only 1.25–1.59° down | Four-second genuine oblique urban hero with subtle motion, requested title, stack and repository text | Decoded ending proof and final 45–49s frames |
| Technical PASS was interpreted too broadly | Separate proof gate, agent editorial review and explicit display/owner limitations; retained historical FAILs | PROOF_REVIEW_RC3, VISUAL_REVIEW_RC3, VIDEO_QA_RC3 |

Actual RC2 final ten seconds contain point-cloud/metric cards rather than the cubes. The cube motif occurs earlier in the late map chapter. Both that motif and the closing composition are replaced. This diagnosis follows decoded evidence rather than assigning a made-up ending effect.

Frame zero remains a moving genuine classified cloud with a large throughput headline and explicit RTX recorded-replay scope. The synchronized original-XYZ/true-prediction reveal keeps identical geometry and changing camera. Qualified ~2M/s 60-second scene-A tests, ~1.5M/s dense/full-churn tests and the separate ten-minute 600,014,848-point ~1M/s run remain distinct. The 48.756% mIoU and model limits remain visible. Diagnostic JSON p99 274.478ms FAIL remains evidence, not an accepted latency claim.

No native runtime, model, benchmark, data source, taxonomy, original tags or RC1/RC2 package was modified. Forest COPC is separately bound to its own native-export sidecar; its 9,939 differences from short JSON-recording predictions are documented and not hidden. New GUI QGIS playback and human owner approval are NOT_RUN. Local review qualification is not publication approval. PUBLICATION=NOT_RUN.

<!-- Copyright (c) 2026 Tuomas Pietilä. SPDX-License-Identifier: CC-BY-4.0. Original expression only; facts and upstream material excluded. -->
