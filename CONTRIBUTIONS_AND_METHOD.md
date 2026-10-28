# Contributions, attribution and engineering method

<!-- Copyright (c) 2026 Tuomas Pietilä. SPDX-License-Identifier: CC-BY-4.0. Original expression only; upstream material and facts excluded. -->

**Developer and project lead: Tuomas Pietilä ([@Juurikko](https://github.com/Juurikko)).**

This is Tuomas Pietilä's human-led engineering project, developed using **AI-assisted software engineering, including OpenAI Codex**. “SynRex” is the existing project/product label used in the technical evidence and accepted media; it is not an attribution of development, authorship or copyright to a company.

## Human leadership and engineering responsibility

Tuomas Pietilä leads the architecture, specifications, engineering decisions, verification requirements and review of this work. The project brief defines the input/output boundary, qualified hardware, benchmark protocol, preservation criteria, rights constraints and the distinction between verified results and future concepts.

Human-led decisions include keeping an already-georeferenced LAZ-to-classified-LAZ boundary, preserving original point order and nonclassification records, matching the characterized model/preprocessing contract, using scatter-safe source-index planning, retaining full launch-to-exit timing, and requiring independent LAZ verification. Tuomas reviewed and accepted the publication's overall design, storytelling, benchmark figures and technical accuracy.

Human-led verification means setting acceptance criteria, interpreting receipts/test results, reviewing claims and deciding what is accepted or released. It does not mean every compiler command or automated test was manually executed by Tuomas. Historical physical-device receipts remain the evidence for the Jetson result; the local synthetic tests are evidence only for the separate public-focused preservation component.

## AI-assisted implementation, testing and documentation

OpenAI Codex assisted with coding, implementation review, test drafting, local build/test automation, documentation, release-hygiene checks and procedural presentation work under the project's specifications and constraints. The staged C++ preservation utility, Python verifier and synthetic fixtures were prepared through this assisted workflow. Tuomas Pietilä remains the named developer/project lead and the human responsible for architecture, requirements, engineering review and release decisions.

AI assistance is disclosed as part of the method. It is not treated as an independent correctness certificate, proof of copyright ownership, or evidence that a new neural architecture was invented. Executed tests, reproducible checks and the frozen hardware evidence support the technical claims. Copyright and licensing are separately scoped in [PERSONAL_ATTRIBUTION_AND_LICENSING.md](PERSONAL_ATTRIBUTION_AND_LICENSING.md).

## Engineering work represented by the accepted evidence

The native deployment work integrates C++ geometry/preprocessing, CUDA occupancy and point reconstruction, TensorRT inference, source-index compaction, metadata-preserving LAZ output, relocatable packaging and component/model identity gates. Characterized conversion differences define a specific deployed FP16 result rather than a universal PyTorch bit-parity claim. Independent laz-rs decoding checks the laz-perf output through another implementation.

The physical-device evidence predates the public showcase preparation. This attribution pass does not access the Jetson, modify private engineering history, alter benchmark measurements or change accepted R0/R1 media. Historical timings, existing model/engine identities and qualified preservation limits remain as documented.

## Upstream contributions remain separate

- **NRCan TreeAIBox** and its research contributors supply the existing segmentation/reference lineage. The inspected reference pins commit `5380ddec2799f55adb43b63407dd12ccf7abefae` and treats that material as CC BY-NC 4.0. Those algorithms, weights, configurations and datasets are not represented as personally authored by Tuomas Pietilä.
- **laz-perf** and its contributors supply native LAZ compression/decompression under Apache-2.0; the tested dependency is commit `b7bbe26109dc986f42d4fc80b8de3d2b6ca634ce` (3.4.0).
- **laz-rs/lazrs**, **laspy**, NumPy and development tools retain their authorship and licenses. The verifier explicitly uses the independent laz-rs path.
- **NVIDIA CUDA, TensorRT and Jetson** supply the vendor compute/runtime platform. No vendor libraries, engines or product imagery are redistributed here.
- **DejaVu** font notices remain intact for the accepted original procedural media. The scene geometry is illustrative and is not an actual model prediction or drone-integration result.

The project does not claim invention of SegFormer3D, TreeAIBox research, LAZ compression or TensorRT. [Third-party notices](THIRD_PARTY_NOTICES.md) and the [rights audit](archive/history/PUBLICATION_RIGHTS.md) retain these boundaries. The active MIT/CC BY 4.0 licenses cover only confirmed personally controlled original work and do not relicense upstream material.

## Public component and contribution-history boundary

The public-focused preservation component is newly written after reviewing the design and original source; it is not presented as clean-room work or the frozen benchmark binary. Its synthetic tests do not reproduce the full classifier or historical GPU performance.

The candidate repository's single initial commit identifies **Tuomas Pietilä** as author and committer using an existing GitHub noreply identity associated with authenticated `@Juurikko`. The generic staging identity is not a project contributor credit. This correction is limited to the new public candidate; private engineering repositories/history and global Git configuration are unchanged. Commit metadata records attribution, not a cryptographic identity or copyright guarantee.
