# Streaming R2 media provenance and QA

<!-- Copyright (c) 2026 Tuomas Pietilä. SPDX-License-Identifier: CC-BY-4.0. -->

The review film is **54 seconds, 1080 × 1350, 24 fps, H.264/yuv420p**, silent with
on-screen explanation. It is an engineering demonstration of recorded data, not a live
physical sensor feed. It contains no generated scenery, stock footage, fabricated flight,
reference labels disguised as predictions, invented confidence or synthetic map state.
Owner publication approval is still required.

## Shot list

| Time | Genuine content and measurement boundary |
|---|---|
| 0–5 s | Original dense scene-C airborne geometry, height shading, EPSG:2154 |
| 5–10 s | Actual fresh native window predictions; exactly the same camera and source positions |
| 10–15 s | Illustrative orbit around those same predictions; camera motion is labelled |
| 15–20 s | A distinct real 85,222-point scene, `TEST-0741_6244-006630086`; independently verified published R4 native predictions from the same frozen model |
| 20–30 s | 25 incremental map states, reconstructed only from independently verified recorded native deltas; playback slowed for inspection |
| 30–36 s | Clearly labelled IGN reference versus actual scene-C predictions, same camera; real errors retained |
| 36–44 s | Accepted R1 external TCP results, exact ten-minute count/coverage/p99, and retained 2M dense/churn FAIL |
| 44–48 s | Genuine new QGIS provider/map export of the independently verified scene-C COPC |
| 48–54 s | Scoped MIT public foundation, real GitHub reference, local candidate and platform limitations |

Main point views display every third source record for rendering only. All **400,654**
scene-C records received fresh neural classifications: one persistent TensorRT context,
**117 actual enqueue calls**, 25 complete windows, no cached map labels, no dropped or
duplicated source IDs. Every source XYZ, intensity, supplied return attribute and emulated
timestamp is compared with the original. All seven predicted IDs occur; tiny rare-class
counts are not presented as quality validation.

The scene-C replay has **21,938** disagreements with the provided IGN reference labels.
Those mistakes are retained. This single-scene count is not the separately qualified
**48.756% seven-class mIoU** over 256 tiles / 24,902,618 points. Ground-truth classes do
not enter native features. The second scene's source, native output, semantic sidecar,
binding and model hashes are independently checked before rendering.

## Exact provenance

The [visual receipt](../../evidence/streaming_r2_visual_provenance.json) binds scene C,
model manifest, engine, input chunks, recorded events, native receipt, source-order labels
and snapshot arrays. The [media receipt](../../evidence/streaming_r2_media_qa.json) additionally
binds the second scene, renderer, font, QGIS export, video and every presentation image.
[Independent decoded review](../../evidence/streaming_r2_decoded_review.json) binds chapter
frames extracted from the final encoded video, rather than trusting pre-encode images.

Native map deltas are checked per accepted observation against an independent ordered
oracle, including victim identity/order and timestamps. A separate map is reconstructed
from the **received deltas**, then compared with that oracle at every displayed frame.
All snapshot keys, semantic IDs and timestamps are retained, capacity 65,536, peak/final
24,458 cells. These are genuine retained map semantics; a displayed resident voxel is not
claimed to be a newly classified source point. Unknown confidence remains unavailable.
There is no TTL or fabricated aging process. Record-order playback is not an acquisition
trajectory; source positions are never translated for the media recording.

The seven colours come from the accepted QGIS style, unchanged:
`#8a9297`, `#a77c43`, `#2c9b59`, `#f2594b`, `#357bd9`, `#b96ed0`, `#f7b62e`
for IDs 0–6. LAS projection `[1,2,200,6,9,17,201]` is explicit and separate.
The 3D rasterizer chooses actual point colours, with a z-buffer and labelled illustrative
camera. It does not interpolate or generate semantic labels. H.264 pixels are lossy
presentation; authoritative integer semantics remain in the separately hashed data.

The QGIS shot is from the original R1 asynchronous first-observation export, not from
the media replay's differently bounded final window. Both are actual frozen-model outputs
on the same source, but numerical equality between those **different contexts is not
asserted**. [Independent COPC audit](../../evidence/streaming_r2_copc_parity.json) restores
source indices and checks all original classified point bytes and semantic IDs. The
[real QGIS provider](../../evidence/streaming_r2_qgis_render.json) independently reads all
400,654 source indices/classes and embedded EPSG:2154. No CRS fallback is assigned.

## QA result and retained failures

- **PASS:** all 1,296 encoded frames decode without errors; duration, dimensions, frame
  count and portrait aspect match; eight chapter frames independently extracted.
- **PASS:** all text bounding boxes fit the portrait safe area; minimum 22 px, chapter
  titles 68 px; decoded frames inspected for clipping, point-cloud visibility, legibility
  and correspondence with the provenance/timeline. No private endpoints appear in the film.
- **PASS:** complete coverage, independent per-delta map reconstruction, semantic taxonomy,
  source/model/output binding and genuine reference/prediction distinction.
- **PASS:** six CPU adversarial tests reject hash-rebound semantic, ID, time, eviction,
  missing-delta and reused-label alterations.
- **FAIL, retained:** two fresh QGIS desktop startup attempts stalled before the worker;
  only owned isolated processes were stopped. The successful standalone map export is
  accurately labelled. Published R4 desktop screenshots remain genuine, independently
  reverified and available in the package.
- **FAIL, diagnostic only:** JSON media recording acceptance-to-file p99 **274.478 ms**
  exceeds 250 ms. Large JSON output is outside the qualified TCP performance boundary;
  this run supplies visual provenance, never a sustained-rate claim.
- **NOT_RUN:** new QGIS 3D export, live physical capture, Orin/Thor rendering/inference,
  public upload or owner acceptance. Scripted 3D rendering fulfils the animation path.

The [opening frame](../../media/streaming-r2/opening-frame.png),
[thumbnail](../../media/streaming-r2/thumbnail.png) and
[contact sheet](../../media/streaming-r2/contact-sheet.png) are small committed previews.
The full film, lossless chapter frames, decoded review frames and actual QGIS images are
release assets. All include the source/licence attribution; see
[rights](STREAMING_R2_RIGHTS.md) and [reproduction](STREAMING_R2_REPRODUCIBILITY.md).
