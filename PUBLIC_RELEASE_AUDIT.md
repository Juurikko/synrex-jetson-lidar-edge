# Public release audit — local staging R0

<!-- Copyright (c) 2026 Tuomas Pietilä. SPDX-License-Identifier: CC-BY-4.0. Original expression only; upstream material and facts excluded. -->

**Developer and project lead: Tuomas Pietilä ([@Juurikko](https://github.com/Juurikko)).**

> **Superseding publication status — 2026-10-26:** [Juurikko/synrex-jetson-lidar-edge](https://github.com/Juurikko/synrex-jetson-lidar-edge) is published and its public visibility is verified. **PUBLIC_GITHUB_RELEASE=PASS · PUBLIC_VISIBILITY=VERIFIED.** Initial public commit: [`4405474`](https://github.com/Juurikko/synrex-jetson-lidar-edge/commit/4405474c73c25b8c19a5c3bf1b2d70fe30ed6d73). The original staging report below is retained as history; its statements about no creation/push, no remote, pending publication or `NOT_RUN` describe that earlier stage and are superseded for this repository's release. Licensing, personal/upstream attribution, noncommercial-reference restrictions and the limited model-free release scope remain unchanged. Evidence files and manifests retain their initial-commit contents.

**Local staging outcome: PASS. Public-release authorization: pending.** The repository contains a substantive runnable preservation component and independent tests, plus the engineering writeup and accepted R1 media. It is neither a complete classifier download nor a claim that the new demo is the benchmark binary.

## Source and asset decisions

The pinned private engineering source was accessed read-only. The Product Runtime R1 launcher, native writer, native geometry/compact plan, CUDA I/O, inference service, two verification implementations, benchmark receipts, model manifest/lineage and license notices were inspected. Individual source identities/dispositions are in [source review](archive/evidence/source_review.json).

Original private runtime/model-dependent source is **excluded** because a blanket public license and complete derivative rights were not established. A newly written, design-informed C++ preservation utility and offline verifier are licensed under MIT after Tuomas Pietilä confirmed authority over the newly staged originals. The original source review is acknowledged; clean-room independence is not claimed. Model/data artifacts and genuine predictions are excluded entirely.

The accepted R1 MP4, thumbnail, portrait and landscape diagrams are copied byte-for-byte and hash-checked. A small GIF excerpt is derived solely from the accepted original video. No restricted imagery, third-party stock media, vendor logos or private terminal screenshots were added. The source publication workspaces remain unchanged.

## Quality gates

| Gate | Review result |
|---|---|
| Accessible README with real technical depth | Full measured file boundary, truthful scope, architecture/validation links, actual code demo and embedded relative GIF/MP4 assets |
| Exact benchmark and limitations | Four recorded observations, correct weighted aggregate, full initialization/file-output boundary, MODE_50W and normal-cache/one-shot qualifications |
| Source substance | Native label-application/preservation layer, independent laz-rs byte verifier, invented fixture generator and 19 tests |
| Runnable checks | Release and sanitizer suites pass, including multichunk LAZ and negative integrity cases |
| Rights discipline | No permissive relicense of third-party/model-derived source; owner-authorized MIT original code and CC BY 4.0 original documentation/media; unchanged upstream notices and explicit exclusions |
| Source provenance | File-level manifest and source audit; Tuomas Pietilä credited as developer/project lead; copyright scope and upstream novelty remain separate |
| Public/private boundary | No private repository links, source-host paths, boot identifiers, credentials or operational infrastructure required to understand the project |
| History/repository hygiene | New isolated repository, one initial local commit, no remotes, ignored builds/data/dependencies; post-commit fsck/object scan |
| Accepted work preservation | 210-file SHA256 baseline unchanged for publication R0/R1 inputs |
| Target safety | No Jetson access/modifications; no unrelated project workloads or files touched |

Detailed code/media rights decisions are in [PUBLICATION_RIGHTS.md](PUBLICATION_RIGHTS.md). The [reproduction report](REPRODUCIBILITY_REPORT.md) records executed checks and practical limits. Automated inventory/pattern checks complement manual review; they do not establish legal ownership or rule out every possible secret.

## Release decision

The local candidate is ready for technical and rights review. Personal ownership/licensing authority for the newly staged originals is confirmed; the scoped MIT / CC BY 4.0 grants are active. Review factual benchmark disclosure and restricted-reference reporting terms, then explicitly authorize public publication in a separate task. Until then: **PUBLIC_GITHUB_RELEASE=NOT_RUN**. Commercial model deployment and releasing the full classifier remain separate work.
