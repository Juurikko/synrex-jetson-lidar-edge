# RC4 editorial review

Copyright (c) 2026 Tuomas Pietilä. SPDX-License-Identifier: CC-BY-4.0.

Decision: **PASS as a local owner-review candidate**. The complete weak RC3 28–40 s map chapter is removed. The 37-second film retains the approved opening, oblique urban/canopy/second-scene cameras, aligned source/prediction comparison, both truthful benchmark cards and genuine classified-building ending. It adds no footage or visual effect. The duration is intentionally below the suggested range rather than padded. This is agent editorial judgment, not owner approval or calibrated mobile-display qualification.

The actual qualified RC3 MP4 and final RC4 MP4 were decoded at native resolution: all 1470 and 1110 frames, respectively, with exact frame byte counts and FFmpeg error escalation. The full RC4 preview also decoded successfully. SSIM was computed for every final frame against its EDL-matched original frame: minimum **0.995395**, mean **0.9971923342**. One H.264 CRF 16 re-encode is the only image-generation step; no rasterizer, point data, model output or camera is regenerated.

## Visual findings

The actual source contact sheet, both new full-film contact sheets, both all-transition sheets, local map-pulse sequences and mobile-readability sheet were inspected. Every source transition is represented by multiple decoded frames before/at/after the boundary. Nine-frame sheets extend each boundary by ±0.5 s; aggregate sheets show −2, −1, 0, +1, +2 frames. The new join at final frame 840 and the closing cut at frame 990 were also examined in their full nine-frame sheets. Samples at 28.200–28.333, 34.400–34.533 and 38.767–38.900 s expose the original local contour/outline churn. This is not an inference from a still-image brightness score.

RC3 map outlines visibly sweep/change across already classified points in short-lived screen-space patterns. Renderer and camera metadata explain why: the 2200-observation window is shorter than the roughly 2226-observation/frame advance. Across both map passes, the recorded candidate active-cell counts range 0–788 and are zero in 81/360 frames (candidate cells before projection/clipping; not a count of every visible outline pixel). At frame 1020 / 34 s, the event counter falls from 400654 to 2226 as the crop and camera switch. The new RC4 frame sequence contains neither map pass, hence no ring annotations, map-start counters, repeated capture or crop-reset animation. An alternative calm map shot was not inserted: the existing approved canopy shot already gives a stronger real classified-point view, and a map replacement would broaden the narrow edit.

The final 28 s hard cut keeps the same urban scene on both sides. Its change in framing/text is clear and brief; there is no flash, wipe or filler between comparison and benchmarks. The benchmark cards retain 2.4 and 2.6 seconds and switch directly at 30.4 s. The original cut to the oblique hero remains at 33 s, followed by four seconds of continuous camera motion, a legible repository address and honest availability text. No caption is truncated, no repeated/held frame is inserted, and no new transition is needed. The approved explanatory source/prediction divider remains unchanged within 17–28 s.

## Luminance and motion evidence

Luminance proxy is BT.709-weighted decoded RGB on a regular 3px grid across the entire frame; values are in 0–255 units. Per-frame arrays are saved for both films. Screen pairs exceeding |mean change| >2 or >10% sampled pixels changing by >20 are retained for review, including legitimate cuts and ordinary point-cloud motion.

| Pair / scope | RC3 | RC4 | Interpretation |
|---|---:|---:|---|
| Maximum absolute mean change within shots | 0.612452 | 0.455113 | Neither proves absence of small local pulses; code/visual review is required |
| Canopy → second scene at 13 s | −15.754004 | −15.775830 | Approved single hard cut, not a repeated luminance oscillation |
| Map entry at RC3 28 s | +12.052301 | Removed | Entire distracting chapter removed |
| Map reset at RC3 34 s | −2.052572 | Removed | Event/crop/view restart removed |
| Map → benchmark at RC3 40 s | −21.668625 | Removed | No longer a forest-to-urban jump |
| New comparison → benchmark at RC4 28 s | — | −7.231420 | One deliberate hard cut within the same urban scene |
| Benchmark → hero, 45 s / 33 s | +6.407736 | +6.408698 | Approved cut retained, no added transition |

The >10% pixel-change screen also flags roughly 7.6–9 s in RC4 (peak 10.6494%); matching RC3 urban frames reach 10.3222%. The source camera moves over high-frequency point samples, while frame mean changes stay small. The scene and camera are approved and SSIM confirms the same source content. These flags are retained as expected geometric sampling/motion, not hidden or relabelled as no pixel changes. The removed map intervals themselves had only 0.426/0.612 maximum continuous mean changes, illustrating why a full-frame brightness heuristic alone would have missed their editorial defect. No photosensitivity certification is claimed.

## Text, provenance and claims

At 432 × 540 review size, the headline, scene titles, source/prediction labels, accuracy disclosure, benchmark numbers/contexts and closing address are readable. Provenance/footer copy remains smaller fine print; it is present, uncut and unchanged from accepted RC3, with full wording also in the caption/audit. Native source font metadata has minimum 22px, headline 57px and numerical benchmark headline 58px; all retained layouts remain within the accepted source safe bounds. The 2.4/2.6-second benchmark cards preserve owner-approved timing; viewers may pause for supporting fine print.

Opening frame 0 reads `UP TO 2 MILLION LiDAR POINTS/S`, with RTX 4090, recorded replay and qualified scene-A context. The actual benchmark panels retain ~2M/s scene A, separate ~1.5M/s dense/churn and a **separate** ten-minute 600,014,848-point full-churn run at ~1M/s, 100% coverage/zero drops. The 39.85 ms p99 belongs to that ten-minute test and appears in the draft, not as a new video overlay. The 274.478 ms diagnostic JSON FAIL stays in negative evidence, never as accepted latency. Held-out 48.756% seven-class mIoU and the 62,103-parameter model stay disclosed. Public R4 is distinguished from local R2; the future-release main draft uses the exact required placeholder.

No ground truth is presented as prediction. The original seven-class palette, source geometry, genuine trained-model predictions, source IDs, map-state records and model hashes are preserved. Existing forest COPC and film outputs remain separately identified rather than falsely claiming they have identical context/predictions. All original RC3 manifest assets were rehashed, earlier baseline/archive hashes verified, and all earlier tracked files remain unchanged. RC4 archive copies carry the complete original input payloads and an exact committed source snapshot/bundle.

## Verification scope and reproduction

Use the existing qualified RC3 media environment (NumPy, Pillow and imageio-ffmpeg); no GPU is required. From this RC4 worktree run `tools/streaming_r2_rc4.py --stage freeze`, `--stage render`, then `--stage qa`. Inspect decoded review derivatives and record a hash-bound `VISUAL_REVIEW_RC4.json` before changing QA status to PASS. `--stage package` requires those reviewed gates and a clean committed tree; it creates the archive, reads every compressed member independently and only then creates the local qualified tag. `owner-review/` holds the archive SHA256, verification receipt and local tag identity; those external seals avoid recursive self-hashing. No public upload/push/release action exists in this workflow.

After packaging, run `python3 tools/streaming_r2_rc4_readback.py build/streaming-r2-review/streaming-r2-rc4/owner-review/synrex-streaming-r2-rc4-owner-review.tar.gz` for the separate standard-library verification: it derives expected hashes from the archive's embedded checksums/manifest, reads every member, checks the full gzip stream through EOF/CRC, and cross-checks the external seal. It does not import the packager or use its in-memory file table. The external independent-readback receipt is intentionally outside the archive to avoid a hash cycle.

Light regression scope: Python syntax/import validation, baseline Git blob equality, all original payload/model hashes, exact benchmark arithmetic and evidence binding, both full decodes, all-frame SSIM, text/layout/mobile review, preview decode, and complete archive readback. No native runtime change, costly GPU campaign, device contact, Spatial Lab service operation, new training, new sensor integration or throughput extrapolation is necessary or performed. PUBLICATION=NOT_RUN.
