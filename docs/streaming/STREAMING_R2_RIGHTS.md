# Streaming R2 provenance and rights manifest

<!-- Copyright (c) 2026 Tuomas Pietilä. SPDX-License-Identifier: CC-BY-4.0. -->

Developer/project lead: Tuomas Pietilä. Coding, verification, automation and original
presentation were assisted by OpenAI Codex under owner specifications and review.
No private implementation, restricted dataset, upstream pretrained checkpoint or stock/
AI-generated visual is imported for R2. The existing design-informed provenance remains
disclosed; no clean-room or independent invention claim is added.

| Material | Exact source / treatment |
|---|---|
| Native runtime | Public R4 foundation, accepted R1 native source `f39fa29`; byte-identical in R2; scoped MIT original source |
| New replay/verifier/renderer/packager/tests | Independently authored tooling against public interfaces; MIT; paths registered in `LICENSE_SCOPE.json` |
| Documentation/original presentation | CC BY 4.0, Tuomas Pietilä; no personal claim over facts, standard formats or third-party material |
| IGN/FRACTAL survey geometry/reference taxonomy | Dataset revision `36e00245ef92fd180f655f6ee410f2e935693cc2`; Etalab 2.0; IGN attribution retained |
| Dataset date | Source latest update **2025-04-05T09:37:48Z**, retrieved **2026-10-26**; carried with derived media and model/data assets |
| Trained model | Frozen checkpoint SHA256 `788b65c42bb5d50d9509a56d38368f51890c4a93300d53578e290d193456c7c4`; original trained weights MIT with IGN attribution; architecture notices retained |
| U-Net architecture | External pytorch-3dunet `7d047d2802561d075d6ce44679d7e696138c47df`, Adrian Wolny MIT; imported optional SE notices preserved; no upstream checkpoint |
| Target-specific TensorRT plans | Generated from original models; model notices plus external NVIDIA SDK terms; not portable RTX→Orin/Thor; no NVIDIA SDK binaries shipped |
| Synthetic Conv3D | Original analytic nonconstant test model; MIT; separate three-class contract, not the seven-class trained showcase |
| QGIS/Qt | Separately installed GPL/upstream application and components; no implementation or executable redistributed; interface in historical screenshots retains upstream rights |
| PDAL/laz-perf/laspy/lazrs | External dependencies with retained upstream notices; no blanket MIT relicense |
| NumPy/Pillow/pyproj | External offline tools; distribution component notices apply; no libraries bundled; absent from native inference |
| DejaVu/FFmpeg | System font rendering and external encoder only; existing full font notice retained; no font/encoder binary or music bundled |

The generated asset manifest lists **every file's SHA256 and size**. Original source
licence membership is explicit in [LICENSE_SCOPE.json](../../LICENSE_SCOPE.json).
[Third-party notices](../../THIRD_PARTY_NOTICES.md), [model notice](../../examples/unet/MODEL_NOTICE.txt),
[full data/weight assessment](../byom/UNET_PROVENANCE.md) and
[source provenance review](../byom/PROVENANCE_REVIEW_R1.md) remain authoritative boundaries.
This is an engineering rights/provenance record, not a legal certification.

## Asset scopes

- `source/` and `source.bundle`: original source/document/media scopes follow the shipped
  `LICENSE_SCOPE.json`; historical evidence is factual metadata, not a new licence grant.
- `models/`: frozen originals and target-specific plans with `MODEL_LICENSE.txt`,
  `MODEL_NOTICE.txt`, upstream architecture and IGN attribution; no vendor runtime library.
- `data/benchmarks/`, `qgis/data/source/` and the published R4 example source: IGN Etalab 2.0.
- Native classified LAZ, COPC, semantic labels, source-indexed provenance, recorded point
  events and map states: derived IGN information; retain Etalab attribution and model notices.
- `media/` and small committed preview PNGs: original presentation CC BY 4.0 over IGN
  Etalab geometry/information; model predictions are genuine. QGIS/Qt screenshot elements
  retain upstream rights. The standalone video credits IGN, Etalab and the source-update date.

## Identity and publication boundaries

Scene C is `TEST-0944_6564-005900112.laz`, source SHA256
`50106aa5d6465707523bf9252ac8ade9e683266b61a94f5eadde6eaeb2276274`.
Its embedded WKT is EPSG:2154; no CRS is guessed for this presentation. A/B benchmark
fixtures have no CRS VLR and remain explicitly local replay coordinates in those tests.
No vertical datum, acquisition trajectory or physical sensor timestamp is invented.

The R2 video point views use the source-record-order media replay predictions. The QGIS
map view uses the separately verified R1 asynchronous first-observation export from the
same source/model. Window context is not identical; these are **not** claimed to be
byte-identical prediction sets. Both have exact source coverage and genuine model outputs.
The media report binds both paths independently. Reference labels appear only in the
clearly labelled quality comparison, never as inference features or model predictions.

No large data/model binary is added to ordinary Git objects. No credentials, access
endpoints or newly introduced machine names are included. One non-connectable device
display name exists in already-published R4 historical evidence; it is reported by the
audit and preserved with that accepted receipt, not repurposed as an access method.
All new qualification commands use portable path placeholders in public receipts.

The public R4 archive's SHA256 is unchanged:
`a2b6fa566b8982859e2e7b7f1a986291d0f056bb2460ccc8199a7f540dd0d82c`.
R2 assets remain local for owner review. Source release, model/data redistribution and
LinkedIn posting are distinct publication actions, all **NOT_RUN** until owner approval.
