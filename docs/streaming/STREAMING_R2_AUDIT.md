# Independent Streaming R2 audit

<!-- Copyright (c) 2026 Tuomas Pietilä. SPDX-License-Identifier: CC-BY-4.0. -->

R2 is a release/showcase campaign, with no native performance optimization or retraining.
The [pre-change freeze](../../evidence/streaming_r2_baseline_freeze.json) protects **252**
native/evidence/media/streaming-documentation files, including every accepted historical
PASS and FAIL. Frozen refs and SHA256s are checked again before packaging.

## Source and executable identities

Qualified native source is `f39fa296332217eb77bd73dd7e1e3515be3f367e`.
`71ca052` subsequently added qualification evidence, auxiliary controllers/verification
and offline export durability work; it did **not** change native production files.
`14bcb4f2c512d735291f796ac905584564e74ddf` is the final R1 evidence/history seal.
It would be incorrect to describe every post-f39 file as evidence-only: Python/export
tooling also changed. R2 retains that accepted auxiliary tooling and distinguishes it
from the native code used for performance measurements.

A fresh exported source tree was reconstructed at the R1 seal. Public laz-perf 3.4.0 and
nlohmann/json 3.11.3 were cloned afresh at the recorded pins; no private source tree was
consulted. A new native RTX build was reconstructed from the exact qualified f39 archive,
using the already installed matching public SDK environment. New build paths produce
different executable hashes; these are separately recorded, never substituted into old
benchmark receipts. R2 staging requires byte-identical native files, frozen models and
unchanged accepted refs. Final candidate source/bundle verification adds a fresh checkout
and exact exported-Git-blob comparisons.

| Fresh regression gate | Result / evidence |
|---|---|
| CPU native tests | **PASS 7/7** |
| CPU Python baseline | **PASS 90**, five physical GPU tests correctly NOT_RUN locally |
| ASan + UBSan native | **PASS 7/7**, leak/error halting enabled |
| Genuine-recording adversarial CPU | **PASS 6/6**, no CUDA dependency |
| Fresh native RTX CTest | **PASS 7/7** |
| R4 physical native preservation | **PASS 5/5**, 31 unrelated tests deselected |
| Synthetic + trained U-Net through same native binary | **PASS**; no runtime rebuild between models |
| Independent stream/offline tensor/logit byte comparisons | **PASS 31 windows**, exact input/logit bytes |
| External TCP native smoke, both models | **PASS**, full source coverage / independent map oracle |
| Duplicate, sequence gap, clock discontinuity, truncation, stalled subscriber | Native **FAIL as required**; fault harness PASS; failures retained |
| Clean reconnects | **PASS** |
| Native asynchronous LAZ export + independent laz-rs | **PASS**, all 400,654 source records; exact nonclassification bytes and metadata |
| COPC source-index reconstruction | **PASS**, original classified bytes and seven semantic IDs exact |
| Actual QGIS provider/map rendering | **PASS**, all 400,654 source indices and semantic IDs |

[Reconstruction receipt](../../evidence/streaming_r2_reconstruction.json) ·
[GPU regressions](../../evidence/streaming_r2_gpu_regressions.json) ·
[byte-exact traces](../../evidence/streaming_r2_trace_audit.json) ·
[TCP](../../evidence/streaming_r2_transport.json) ·
[faults](../../evidence/streaming_r2_transport_faults.json) ·
[export](../../evidence/streaming_r2_export.json).

The long/sustained measurements were **preserved and independently audited**, not rerun
as another optimization campaign. Their original source/binary/model identities remain
in the accepted R1 receipts. Short media and correctness regressions carry no new
sustainable-throughput claim. Every counted point in the accepted trials is reconciled
with fresh model execution; map observations/deltas and prepared-block model-only work
remain separate. The exact 600,014,848 / 600.026 s / 39.845888 ms result remains approximately
1M/s; stable scene A's approximately 2M result is two separate 60-second runs.

## Public foundation and source hygiene

The already-published R4 asset was downloaded independently, not merely trusted from a
local filename. SHA256 remains
`a2b6fa566b8982859e2e7b7f1a986291d0f056bb2460ccc8199a7f540dd0d82c`.
The unchanged archive verifier checked **482 manifest files**, exported source blobs,
native data/semantic pairs and COPC source-index restoration. GitHub's release metadata
agrees with that digest. [Public-foundation receipt](../../evidence/streaming_r2_public_foundation.json).

Current-source and new-object scans check secrets, private paths/endpoints, forbidden
data/model binaries and licensing. One non-connectable device display label exists in
unchanged, already-public R4 evidence; it is explicitly accounted for rather than
silently rewriting the accepted receipt. New public receipts replace private execution
roots with portable placeholders and retain untouched raw-receipt SHA256s. No private
infrastructure address, access alias or credential is newly published. Native model/data
and large media assets are kept outside ordinary source history.

README and active R4 release documentation now state that R4 is published, via a new
traceable documentation change. Historical tags, release archives and candidate-stage
evidence remain unchanged. MIT source, model/architecture, IGN data, original presentation,
NVIDIA SDK and QGIS/Qt rights remain separately scoped.

## Explicit exceptions and publication gate

The two new QGIS desktop startups remain **FAIL**; the successful map-provider export
and separately verified historical desktop images have accurate labels. New QGIS 3D is
**NOT_RUN**; the actual-data scripted renderer supplies animation. The JSON media
recording's 274.478 ms p99 is **FAIL against the original 250 ms gate**, and is not used
as performance evidence. Neither exception changes accepted runtime measurements.

Native correctness, source preservation, visual provenance, decoding and package
integrity are independently reviewable. Public publication is **NOT_RUN** until explicit
owner approval. No AGX/Thor or unrelated workload was touched; no push, PR, release upload
or social message was sent. The final candidate receipt records source/asset hashes and
the engineering gate matrix separately from publication authorization.
