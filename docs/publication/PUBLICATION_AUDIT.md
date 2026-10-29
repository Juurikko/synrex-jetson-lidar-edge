# Public snapshot, provenance and rights audit

<!-- Copyright (c) 2026 Tuomas Pietilä. SPDX-License-Identifier: CC-BY-4.0. Original expression only; upstream/facts excluded. -->

The owner authorized public GitHub publication conditional on complete source,
privacy and rights gates. The public RC4 snapshot uses one new commit directly
parented by already-public main `a9a104621c2092a383622667a52c924d4dfee0e9`.
The public tag and release branch must both resolve to that new commit. There is
no claim that they resolve to the frozen private qualification commit.

The qualified reference is `8a62143afa4d4033e02b6b492d355b8f04451844`.
Only its tracked tree was exported. Its private Git ancestry, old tags, raw receipt
blobs and review bundles were not imported into public Git. Source equivalence is
proven by the [497-file SHA256 comparison](PUBLIC_SOURCE_EQUIVALENCE.json), not
by moving a private qualified tag. The reference and its assets remain unchanged.

Two historical receipts have public derivatives: seven unnecessary artifact-path
keys in `evidence/streaming_r2_rc3_preservation.json`, and the encoder-installation
path in `evidence/streaming_r2_rc4_render.json`. Stable relative identifiers,
executable basename and explicit omission notices replace location metadata.
Every original digest, number, baseline identity, measurement and gate is retained.
The [transform record](RECEIPT_TRANSFORM_AUDIT.json) contains exact source lines
and original/public hashes are in the equivalence manifest. Raw originals are
excluded from the public tree, archives and reachable object set.

Every executable, script, test, build file, runtime configuration, geometry,
numerical measurement and model identity remains byte-identical. No code-like
implementation change was necessary. `LICENSE_SCOPE.json` is an explicitly
enumerated metadata-only scope correction, not runtime configuration.

## Original tooling rights

The nine RC2/RC3/RC4 tools now have precise source-manifest membership. This is
supported by author creation/revision provenance and original additive grants,
rather than inferred from headers alone. See the per-file
[provenance review](TOOL_PROVENANCE_REVIEW.json).

| Scope | Positive evidence |
|---|---|
| Three RC2 tools and media lock | `docs/streaming/rc2/LICENSE_SCOPE_RC2.json`, explicit original-source MIT list |
| Three RC3 tools | Original author introduction commit, original implementations against existing project/public APIs, scoped MIT declaration in `docs/streaming/rc3/STORYBOARD_RC3.md` Rights, owner authority record |
| Three RC4 tools | `docs/RC4_MEDIA_LICENSE.md`, explicit original-tooling MIT grant |
| Owner authority / AI assistance | `evidence/license_activation.json`; human-led specifications/review, Codex-assisted source and documentation |

Seven newly scoped historical documentation files receive appended SPDX/copyright
metadata matching their existing original-expression grants. Historical text is
preserved; these documentation-only changes are listed in the manifest.

Dependency APIs are imported, not vendored. No TreeAIBox/SegFormer implementation,
restricted pretrained model, unqualified weight, NVIDIA binary, font or encoder
is included. The original source provenance boundary remains disclosed; this is
not a clean-room or independent-invention claim.

## Separate asset and dependency scopes

Original source is MIT only within the precise manifest. Original docs/presentation
are CC-BY-4.0. IGN/FRACTAL survey information and derived semantic/geometry media
retain Etalab-2.0 attribution: revision `36e00245ef92fd180f655f6ee410f2e935693cc2`,
source update 2025-04-05T09:37:48Z, retrieved 2026-10-26. The approved video has
unchanged SHA256 `f8c8852a2d5ba55cc524da9a224692e5a9fd2ef9b7be2e844814517c5e39f8b9`.
It shows authentic predictions with illustrative cameras and no acquisition pose.

The trained checkpoint is separately published in R4, with original trained-weight
MIT, Adrian Wolny architecture and IGN notices retained. The R4 asset was downloaded
again and its exact published SHA256 and checkpoint digest verified. No weight or
dataset is newly redistributed in RC4 assets. External NVIDIA SDKs/TensorRT plans,
libraries, QGIS/Qt, DejaVu fonts and FFmpeg retain their own terms. The full notices
remain in `THIRD_PARTY_NOTICES.md` and `LICENSES/`; no blanket archive MIT claim.

## Reconstruction, measurements and review

Fresh public pins were fetched independently: laz-perf 3.4.0 and nlohmann/json
3.11.3, with commits verified, plus the exact Python development/BYOM locks.
Release CPU build and 7 CTests passed. ASan/UBSan build and all 7 CTests passed.
Python: 90 passed, 11 skipped (5 GPU, 6 genuine-recording tests). An initial
archive-only test run had one preparation failure because Git metadata was absent;
creating a new local Git repository from those same source bytes resolved it.
No implementation edits or GPU work were used to obtain the passing recheck.

The [claims audit](PUBLIC_CLAIMS_AUDIT.json) independently filters trial CSVs,
checks raw endurance fields and recomputes seven-class mIoU from the confusion
matrix. Original performance receipts and negative trials remain intact. The
37-second film is a visual illustration, not a performance-timing source. Optional
LAZ/COPC output is separately verified and outside headline throughput. There is
no new physical sensor, flight, pose, Jetson AGX Orin or Jetson Thor measurement.

Before any push, deterministic scanners inspect the full staged tree and every
public-reachable commit/tree/blob, including obsolete versions from public base,
and every newly introduced object. Checks cover credential/key patterns, private
paths and endpoints, unexpected executable/model/data assets, filenames, sizes,
licence scope and archive member contents. Each pattern exception is bound to
specific reviewed object/line hashes: TensorRT version strings; a privacy regex;
and already-public non-connectable R4 platform labels/receipt filenames. Public
provenance URLs remain attributed. None permits a new private path or secret.

The final external object audit, clean commit identity, archive and asset hashes
are distributed as release audit assets. Draft assets must be freshly downloaded
through authenticated GitHub and match exact hashes before public prerelease
publication. No unfiltered owner-review archive or Git bundle is a release asset.
Historical NOT_RUN/FAIL receipts remain factual records of their original runs.
The original R4 release is preserved separately. LinkedIn posting is not performed.
