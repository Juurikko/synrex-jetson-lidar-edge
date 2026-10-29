# Public semantic LiDAR: a realistic extension to existing national data services

<!-- Copyright (c) 2026 Tuomas Pietilä. SPDX-License-Identifier: CC-BY-4.0. Original expression only; upstream material and facts excluded. -->

**Concept note — not an agency programme, procurement proposal, deployed service or endorsement.** Tuomas Pietilä ([@Juurikko](https://github.com/Juurikko)), with OpenAI Codex-assisted engineering in the underlying project. Context checked against agency product descriptions in October 2026.

## Start with what Finland **already** offers

Finland's **National Land Survey (Maanmittauslaitos, NLS/MML) already publishes classified LiDAR point clouds**. It would be incorrect to suggest national agencies do not distribute classified "raw" point clouds.

| Dataset/product | Existing capability and access |
|---|---|
| [MML laser scanning 0.5 points/m²](https://www.maanmittauslaitos.fi/en/maps-and-spatial-data/datasets-and-interfaces/product-descriptions/laser-scanning-data-05-p) | **Open**, already classified LAZ derived from denser source; access via MapSite and file interfaces, subject to open-data attribution conditions |
| [MML laser scanning 5 points/m²](https://www.maanmittauslaitos.fi/en/maps-and-spatial-data/datasets-and-interfaces/product-descriptions/laser-scanning-data-5-p) | Classified LAZ, **restricted access/licensing**; not a public redistribution source at original density |
| [MML elevation model, 2 m](https://www.maanmittauslaitos.fi/en/maps-and-spatial-data/datasets-and-interfaces/product-descriptions/elevation-model-2-m) | Open raster ground elevation model |
| [Finnish Forest Centre canopy-height models](https://www.metsakeskus.fi/fi/avoin-metsa-ja-luontotieto/metsatietoaineistot/metsavaratiedot) | Existing open-data **1 m canopy-height** raster, distinct from MML's ground elevation product |
| [KALLIO national scanning programme](https://www.maanmittauslaitos.fi/laserkeilaus-ja-ilmakuvaus) | From 2026, 20 points/m² acquisition is planned/being introduced for much of Finland, with separate 5 points/m² areas and derived 0.5 points/m² product; this is **not a claim that open 20 p/m² tiles are now generally downloadable** |

The existing MML low/medium/high vegetation classifications are **height ranges relative to ground**, and the product description warns that these point sets include features other than vegetation. They are **not equivalent to all-purpose semantic object recognition**. MML's 2020–2023 dense LAZ distribution uses LAS 1.2 and its 2024-onward distribution LAS 1.4; **neither wording guarantees compatibility** with this repository's narrow PDRF7/42-byte writer.

For an international parallel, the USGS [3D Elevation Program](https://www.usgs.gov/the-national-map-data-delivery/gis-data-download) already offers downloadable LiDAR point clouds and DEMs. The underlying concept of public classified LiDAR is established, not a new invention.

## The opportunity: richer **versioned semantic products**

Rather than replacing established authoritative LAS classification, agencies and their partners could **evaluate optional, quality-assessed semantic layers** that preserve the source points and enable more use cases:

- **Additional semantic interpretation** where defensible from available density/data: terrain, distinct vegetation structure, buildings, infrastructure features and other application-specific objects.
- **Versioned point-level annotations or separately published derivatives** rather than silently overwriting established authoritative ground/error classes. Publish exact source tile IDs, acquisition date, CRS/vertical datum, model/version, label ontology and validation evidence.
- **Compact thematic products** for broad distribution—e.g., class masks, vectorized objects, coverage summaries, confidence/uncertainty indicators or change-detection candidates—without requiring every user to download whole dense point-cloud tiles.
- **On-demand access** using agency product catalogues, tiles and documented licensing, where the technical and legal services exist.
- **Measured quality rather than assumed AI accuracy:** per-class precision/recall/IoU, geographic transfer and uncertainty calibration; independent point-order and nonclassification-attribute checks; professional validation before authoritative use.

Potential beneficiaries include forest managers, flood/land-use planners, municipalities, infrastructure owners, grid/power-line operators, surveyors, environmental researchers and makers of robotic/autonomous systems. Sensitive infrastructure, privacy, national-security rules and licensed high-density data **must** be handled according to the provider's rules. Not every proposed class will be observable reliably from 0.5 points/m².

## Small, realistic evaluation (proposal)

1. **Choose an open and rights-cleared 0.5 p/m² pilot tile**, documenting the actual LAS version, point data record format, VLR/EVLR structure and license attribution. Do **not** assume the public C++ utility accepts it unchanged.
2. **Compare existing classifications with one narrowly defined new use case**, such as better structure/vegetation separation. Obtain independent reference labels; avoid describing new model labels as ground truth.
3. **Publish two separate outcomes**: (a) a retained-source-data semantic-layer/sidecar proof of concept, and (b) a genuinely lightweight thematic map usable without raw LAZ transfer. A sidecar/extra-dimension product, format conversion or expanded writer **requires new code** and is not included in this repository.
4. **Measure semantic accuracy, time, file size and usability** on documented inputs. Compare with MML's existing products and professional baselines, not only with unclassified point clouds.
5. **Address governance first**: publish only when dataset rights, reidentification/sensitive-feature considerations, completeness, official responsibility and maintenance/versioning are satisfied.

The accepted private SynRex demonstration establishes file-to-file native performance on four LAS 1.4/PDRF7 research tiles. Its largest file was 2.13M points in 10.11 s on an AGX Orin. **That single-device, one-shot result does not qualify Finland-wide processing capacity, national service costs, segmentation accuracy on Finnish MML data or operational distribution.** The published [model-free LAZ utility](../../BRING_YOUR_OWN_MODEL.md) can help with integrity-preserving delivery for **supported file formats**, not with creating semantic AI labels or modifying licensing conditions.

## Question for public discussion

> Finland already shares classified LiDAR and derived elevation/canopy data. Should the next layer be **versioned, quality-assessed semantic point-cloud annotations and compact thematic products**, so forestry, surveying, infrastructure and environmental applications can consume useful information without reproducing the same heavy processing independently?

This is a discussion of possible public data infrastructure, not a proposal that MML currently lacks classified point-cloud downloads.

### Sources

- [NLS/MML — open 0.5 p/m² classified data](https://www.maanmittauslaitos.fi/en/maps-and-spatial-data/datasets-and-interfaces/product-descriptions/laser-scanning-data-05-p)
- [NLS/MML — restricted 5 p/m² classified data](https://www.maanmittauslaitos.fi/en/maps-and-spatial-data/datasets-and-interfaces/product-descriptions/laser-scanning-data-5-p)
- [NLS/MML — current KALLIO scanning programme](https://www.maanmittauslaitos.fi/laserkeilaus-ja-ilmakuvaus)
- [NLS/MML — elevation model, 2 m](https://www.maanmittauslaitos.fi/en/maps-and-spatial-data/datasets-and-interfaces/product-descriptions/elevation-model-2-m)
- [Finnish Forest Centre — canopy-height products](https://www.metsakeskus.fi/fi/avoin-metsa-ja-luontotieto/metsatietoaineistot/metsavaratiedot)
- [USGS — downloadable point clouds and DEMs](https://www.usgs.gov/the-national-map-data-delivery/gis-data-download)

**License:** Original prose CC BY 4.0, © 2026 Tuomas Pietilä (original protectable expression only); agency facts/source materials remain theirs and are linked, not relicensed.
