# Streaming R2 RC1 — owner review index

<!-- Copyright (c) 2026 Tuomas Pietilä. SPDX-License-Identifier: CC-BY-4.0. -->

**Prepared locally. No public push, PR, release upload or LinkedIn post.**
Candidate source is `e33679c459593ebf39bc86f1813ccf511613aa67`; subsequent seal commits
add qualification records and review documentation only. Native source remains accepted `f39fa29`; R1 seal
`14bcb4f`, R4 releases, frozen weights and all historical PASS/FAIL receipts are preserved.

The review assets are in workspace-relative `build/streaming-r2-review/`:

- `streaming-r2-rc1.tar.gz`: complete 173,366,802-byte showcase/source bundle.
- `streaming-r2-rc1-source.tar.gz`: 27,478,882-byte source-only archive.
- `streaming-r2-rc1-qualification.json` and `SHA256SUMS`: independent identity/gate records.
- Extracted `streaming-r2-rc1/media/synrex-streaming-r2-review.mp4`: 54-second portrait film.
- Extracted `media/decoded-review/decoded-contact-sheet.png`: actual encoded-frame review.
- Extracted `qgis/synrex-fractal.qgz` and `qgis-r4-foundation/`: genuine geospatial projects.
- Extracted `models/`, `recording/` and `native-export/`: separately licensed verified assets.

[Exact archive/video hashes and final gates](../../evidence/streaming_r2_release_candidate.json) ·
[fresh candidate reconstruction](../../evidence/streaming_r2_candidate_reconstruction.json) ·
[archive byte verification](../../evidence/streaming_r2_archive_integrity.json) ·
[source history audit](../../evidence/streaming_r2_history_audit.json).

| Required engineering gate | Status |
|---|---|
| Source reconstruction | **PASS**: fresh bundled checkout; 444 source blobs; CPU native 7/7, Python 96 PASS |
| Runtime correctness | **PASS**: fresh two-model RTX execution, 31 exact trace windows, TCP/map/export regressions |
| Visual provenance | **PASS**: genuine two-scene model outputs; 400,654 newly classified scene-C points; verified native-delta map states |
| Video QA | **PASS**: all 1,296 frames decode; 1080 × 1350 / 24 fps; actual chapter frames inspected |
| Scoped licensing/provenance | **PASS**: source/model/data/media/upstream scopes separate |
| Release integrity | **PASS**: all 649 actual archive file payloads checked; historical hashes unchanged |
| Publication readiness | **PASS for owner-review candidate**; approval/publication **NOT_RUN** |

The fresh source also passed sanitizer tests 7/7 and fresh RTX CTest 7/7; five R4 physical
GPU preservation tests passed. Five GPU tests were correctly skipped on the local CPU
checkout; this is not substituted for their separate physical execution.

Two new desktop startup attempts remain **FAIL**; genuine QGIS provider/map export passes,
and historical published desktop screenshots are independently verified. New QGIS 3D is
**NOT_RUN**; the film uses a labelled actual-data scripted 3D camera. The diagnostic JSON
media recording's 274.478 ms p99 remains **FAIL against 250 ms** and supplies no performance
claim. Neither exception alters the accepted TCP results or weakens a runtime gate.

Retained performance scope: approximately 2M fresh points/s on stable A, repeated 60 s;
approximately 1.5M on dense C/full churn; the ten-minute full-churn run is approximately
**1M/s**, exactly 600,014,848 points with 100% coverage, zero drops and 39.85 ms p99.
Separate held-out model quality remains **48.756% seven-class mIoU**. Physical sensor,
Orin and Thor remain **NOT_RUN**; no AGX/Thor workload was accessed.

[Release notes](../../docs/streaming/STREAMING_R2_RELEASE.md) · [reproduction](../../docs/streaming/STREAMING_R2_REPRODUCIBILITY.md) ·
[media QA/provenance](../../docs/streaming/STREAMING_R2_MEDIA.md) · [rights](../../docs/streaming/STREAMING_R2_RIGHTS.md) ·
[LinkedIn post and first-comment drafts](STREAMING_R2_LINKEDIN.md).

Owner review should cover the film, source/model/data redistribution scopes, wording and
the exact candidate identities. Approval to publish is a later explicit action. The review
film's unpublished/review end card must be updated truthfully for any later public final
cut, followed by the same media QA. No additional runtime optimization is proposed.
