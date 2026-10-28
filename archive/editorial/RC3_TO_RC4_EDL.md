# RC3 → RC4 edit decision list

Copyright (c) 2026 Tuomas Pietilä. SPDX-License-Identifier: CC-BY-4.0.

Source: qualified `streaming-r2-rc3-qualified`, commit `8e78417e732968d06ff0b9e63f45c394fe03837d`. MP4 SHA256 `acce2565328a8763b4ce8c44a697ff891be407144a65b1bf5e596bf89298e40a`. Source is 1470 frames / 49 seconds at 30 fps. All intervals below are half-open; the end frame is excluded.

| RC3 seconds | RC3 frames | RC4 seconds | RC4 frames | Decision |
|---|---|---|---|---|
| 0–5 | 0–150 | 0–5 | 0–150 | Keep frame-zero title, RTX 4090/replay disclosure and first building camera |
| 5–9 | 150–270 | 5–9 | 150–270 | Keep approved urban roof/wall fly-through |
| 9–13 | 270–390 | 9–13 | 270–390 | Keep genuine classified canopy/ground view |
| 13–17 | 390–510 | 13–17 | 390–510 | Keep second real scene and compact-model disclosure |
| 17–28 | 510–840 | 17–28 | 510–840 | Keep aligned original-XYZ/prediction comparison and 48.756% mIoU disclosure |
| 28–34 | 840–1020 | — | — | Delete entire first map-event crop: short-lived white/cyan cell outlines and busy counters |
| 34–40 | 1020–1200 | — | — | Delete entire repeated capture and abrupt crop/camera/counter reset |
| 40–42.4 | 1200–1272 | 28–30.4 | 840–912 | Keep scene-A ~2M/s and separate dense/full-churn ~1.5M/s benchmark panel |
| 42.4–45 | 1272–1350 | 30.4–33 | 912–990 | Keep separate ten-minute 600,014,848-point, ~1M/s, full-coverage/no-drop panel |
| 45–49 | 1350–1470 | 33–37 | 990–1110 | Keep approved genuine classified building hero and public-R4/local-R2 status |

The only new join is RC4 frame 839 → 840 (27.966667 → 28.000000 s), joining RC3 frame 839 → 1200. The original benchmark panel change at 42.4 s becomes 30.4 s; hero cut at 45 s becomes 33 s. Existing shot cuts at 5, 9, 13 and 17 s remain. No caption is cut internally; both benchmark cards retain their original 2.4/2.6-second dwell and the ending retains four seconds. No map filler, hold, repeated frame, dissolve, brightness pulse, fullscreen transition wipe, pixelation or synthetic geometry is added. The approved moving original/prediction divider remains a data comparison, not a new transition effect.

The film is 37 seconds, one second below the suggested ~38–45 s range. This is intentional: the full weak 12-second passage is removed, every approved retained shot keeps its duration, and no material is added merely to reach 38 seconds. The original and final are silent: no audio exists to splice, fade or retime. H.264 High / yuv420p / 1080 × 1350 / 30 fps / faststart, CRF 16, one new encoding generation. Every retained decoded frame is compared to its exact RC3 counterpart via SSIM, alongside full decode of both films.

## Frame/timestamp defect diagnosis

`tools/streaming_r2_rc3.py:Film.map_overlay` selects `active = last >= stop - 2200`, drawing fixed 3px white/new or cyan/reobserved round cell-centre outlines. `Film.frame` advances about 2,226 source observations each film frame (`400654 / 180`), slightly more than the entire 2,200-event recent window. Consequently most active annotations can vanish and be replaced in the very next 1/30 s frame. `new = first >= stop - 2200` further switches outline colour according to first insertion. These are genuine observations, but their visibility is a high-frequency overlay policy, not a continuous stable map presentation. It can look flashy locally even when full-screen mean luminance changes little. RC4 removes all these frames rather than declaring the effect fixed from a global-intensity threshold.

At RC3 frame 1019 / 33.966667 s the event counter reaches 400,654. Frame 1020 / 34.000000 s starts again near event 2,226. `stop < event_pos` clears the recent-state tracking (`event_pos=0; last[:]=-1`), while the camera changes to the opposite-side crop centred at [15,35] instead of [25,25], both with 18 m radius. `replay_repeat=True` and the disclosure “repeated capture” confirm the second six seconds replay the same observations, not new native capture. This simultaneously restarts counters and changes crop/viewpoint. Both passes occupy frames 840–1199 inclusive and are absent from RC4.

Decoded comparisons use the actual H.264 source, not just renderer stills. RC4 stores full-resolution sampled PNG/JPEG derivatives, RC3/RC4 contact sheets and nine-frame transition sheets at ±15, ±6, ±2, ±1, 0, +1, +2, +6 and +15 frames where available. These cover RC3 cuts at 5, 9, 13, 17, 28, 34, 40, 42.4 and 45 s and RC4 cuts at 5, 9, 13, 17, 28, 30.4 and 33 s. `evidence/RC3_FRAME_LUMINANCE_RC4.json` and `RC4_FRAME_LUMINANCE_RC4.json` record every frame’s luminance proxy and changes. The original building shots, comparison and ending remain the exact source frame sequence, subject only to the documented H.264 re-encode.
