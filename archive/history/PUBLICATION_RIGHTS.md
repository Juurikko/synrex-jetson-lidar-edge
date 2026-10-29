# Source-by-source release audit

<!-- Copyright (c) 2026 Tuomas Pietilä. SPDX-License-Identifier: CC-BY-4.0. Original expression only; upstream material and facts excluded. -->

**Developer and project lead: Tuomas Pietilä ([@Juurikko](https://github.com/Juurikko)).**

> **Superseding publication status — 2026-10-26:** [Juurikko/synrex-jetson-lidar-edge](https://github.com/Juurikko/synrex-jetson-lidar-edge) is published and its public visibility is verified. **PUBLIC_GITHUB_RELEASE=PASS · PUBLIC_VISIBILITY=VERIFIED.** Initial public commit: [`4405474`](https://github.com/Juurikko/synrex-jetson-lidar-edge/commit/4405474c73c25b8c19a5c3bf1b2d70fe30ed6d73). The original staging report below is retained as history; its statements about no creation/push, no remote, pending publication or `NOT_RUN` describe that earlier stage and are superseded for this repository's release. Licensing, personal/upstream attribution, noncommercial-reference restrictions and the limited model-free release scope remain unchanged. Evidence files and manifests retain their initial-commit contents.

This repository is a **local public-release candidate**, not a published repository or a complete inference product. The owner requested staging, testing and a clean local commit only. Source inputs were read-only. No restricted data/model artifacts or private runtime source files are included in tracked content.

## Original source audit

The original engineering source/evidence was inspected at commit `69eca500738b0412d1a2a0b6ba0edeac20012aec`. [Source review identities](../evidence/source_review.json) record Git blob hashes and individual dispositions without private repository links or raw paths.

| Reviewed original material | Assessment | Release disposition |
|---|---|---|
| Product Runtime R1 native C++ launcher | Native packaging/target/hash gates, relocation, license opt-in, child execution. No blanket public license found for reviewed private runtime files. | Original code and operational contracts excluded; architecture described in prose. |
| Native classified-LAZ writer | Separably useful integrity design; original source lacks established public release grant in the inspected material. | Original not copied. New public-focused implementation staged, with design relationship disclosed; newly staged original code is MIT under owner confirmation. |
| Native geometry and compact plan | Frozen TreeAIBox-compatible grouping, voxelization and scatter semantics. A C++ translation does not by itself remove derivative licensing questions. | Original implementations excluded, including model-dependent preprocessing. Factual contracts/decisions summarized. |
| CUDA occupancy/argmax/scatter and inference service | Native integration around the existing model, with model/target-specific contracts. | Original implementations excluded. No claim of a full downloadable classifier. |
| Original C++ point-record verifier | Useful validation approach, but public rights to original text not established here. | Original excluded; new verifier staged. |
| Original independent Python verification routine | laz-rs verification plus private operational imports/paths. | Original excluded; new standalone verifier has no operational imports or source-host paths. |
| Trained reference model/config/ONNX/TensorRT engine | TreeAIBox-derived evaluation lineage, identified as noncommercial; model artifacts are not personally authored or automatically commercially licensable model IP. | All model artifacts and derived research code excluded. Attribution and provenance only. |
| Benchmark report, receipts and preservation summaries | Fact source; raw files contain private paths, machine identifiers and unrelated operational context. | Raw records excluded. Sanitized numerical observations, output digests, timing limits and source commit included. |
| Model lineage/license notices | Identifies NRCan TreeAIBox commit `5380ddec2799f55adb43b63407dd12ccf7abefae` and CC BY-NC 4.0 treatment. | Read-only audit; no trained weights, genuine predictions or model source redistributed. |
| ECLAIR scene files, screenshots and predictions | Noncommercial restrictions were part of the source-of-truth task constraints. | None included or used for media; synthetic fixtures are invented. |

The original notices describe SynRex native engineering separately from the trained model. This supports documenting the separation; it is not proof of ownership of every source file or a grant to put model-derived code under MIT/Apache. No such permissive relicense is performed.

## Newly staged public-focused material

| Files | Provenance | Rights state |
|---|---|---|
| `src/preserve_laz.cpp` | New C++ implementation for this task after review of the preservation design and original source. Applies externally supplied labels; no model/preprocessing code copied. This is not a clean-room claim. | MIT for the original implementation; Tuomas Pietilä confirmed ownership/licensing authority with the design/source relationship disclosed. No excluded source is licensed. |
| `src/verify_laz.py` | Newly written raw-metadata parser and explicit laz-rs byte/digest verifier. | MIT for original source/tests, personally controlled by Tuomas Pietilä. |
| `src/synthetic_laz.py` | Original deterministic invented records and height-rule fixture. Not a derivative of an actual point cloud. | MIT for original source; generates no pretrained-model output. |
| `tests/test_preservation.py` | New tests for the staged utility and independent verifier. | MIT for original source/tests, personally controlled by Tuomas Pietilä. |
| `tools/*.py`, CMake/configuration, documentation and audit manifests | Newly prepared staging/reproduction/review work with AI coding assistance at the owner's direction. | MIT for original source/configuration; CC BY 4.0 for original documentation; factual audit records are preserved without personal copyright claims over their facts or upstream material. |
| Accepted R1 MP4, thumbnail and diagrams | Original procedural publication media, copied byte-for-byte from accepted R1; font notice retained. | CC BY 4.0 for personally controlled original expression, with font notices retained; no new public disclosure is performed. |
| `media/preview.gif` | Short, scaled excerpt of the accepted R1 MP4; source hash/filter recorded. | CC BY 4.0 for the original-media excerpt; not genuine model predictions. |

**MIT** is active for the confirmed original source/tests/configuration and **CC BY 4.0** for confirmed original documentation/media. Tuomas Pietilä explicitly confirmed ownership/licensing authority; notices apply only to protectable original expression. [LICENSE_SCOPE.json](../../LICENSE_SCOPE.json) controls scope. The retained historical review-only notice is superseded for these scopes; upstream licenses and excluded private/model-dependent code remain untouched. AI assistance is disclosed and is not itself a copyright claim. No ownership of facts, ideas, formats or third-party expression is claimed.

## External dependencies and notices

laz-perf 3.4.0 is fetched at a pinned public commit and compiled from the external checkout, preserving its Apache-2.0 headers. laspy, lazrs, NumPy and development tools are installed separately from pinned packages; neither their code nor binaries are committed. [Third-party notices](../../THIRD_PARTY_NOTICES.md) and the license inventory identify those boundaries. No dependency is relabeled as personally authored by Tuomas Pietilä.

The DejaVu permission notice is retained for already-rendered text. No font binaries, stock footage, vendor product imagery, music or generated paid-API assets are included.

## Excluded sensitive material

No ECLAIR files or derivatives; no checkpoint, ONNX graph, TensorRT plan or `.srxmodel`; no private source repository URLs; no credentials; no boot IDs, machine addresses or storage paths; no private deployment scripts; no unrelated project content. The target hardware/software versions and MODE_50W are intentional technical disclosures already required by the brief. Media does not expose a private terminal or dataset screenshot.

## Remaining release questions

1. **Resolved for included originals:** Tuomas Pietilä is developer/project lead and confirmed ownership/licensing authority for original source, documentation and procedural media; scoped MIT / CC BY 4.0 licenses are active. This does not resolve excluded private-runtime or third-party/model rights.
2. Owner authorization to disclose the private engineering results and public-reference lineage details.
3. Confirmation that factual reporting/marketing of the restricted-reference evaluation is permitted under applicable data/model terms, or additional permission if required. Excluding restricted assets does not itself settle every reporting question.
4. Final human review and an explicit later instruction to create/push/publish the named public repository.

These are publication gates. The runnable model-free demonstration and local quality checks can be reviewed now. Commercial deployment of a classifier with this reference model is not authorized by this repository.
