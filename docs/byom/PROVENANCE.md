# Native BYOM R0 source and licensing assessment

<!-- Copyright (c) 2026 Tuomas Pietilä. SPDX-License-Identifier: CC-BY-4.0. Original expression only; upstream material and facts excluded. -->

Developer and project lead: Tuomas Pietilä. Engineering is AI-assisted with OpenAI Codex.

The [second-pass preprocessing review](PROVENANCE_REVIEW_R1.md) compares the actual generic
implementation against frozen private geometry during RTX4090 qualification. It supports the
scoped technical disposition below, with design exposure disclosed and no clean-room or external
legal certification claimed.

The initial assessment read the public README, architecture, licensing scope, source review,
writer, verifier, fixtures and qualification records; the private repository's entrypoint,
current state, frozen identities, engineering memo, package manifest/format, licensing notices
and relocation qualification. A source inventory and license/comment search identified the
private components' boundaries. No private algorithm bodies were imported into this runtime.
The private repository is read-only throughout this work.

The public `src/preserve_laz.cpp`, `src/verify_laz.py` and `src/synthetic_laz.py` already have
explicit personally authorized MIT scope. The native runtime links the public writer directly;
its only integration change exposes validation and label application as library functions and
guards the original CLI main. Its compression and metadata-preservation implementation is
retained. The existing public preservation tests remain in place.

The private native implementations have no blanket public license established by the records.
Several explicitly reproduce TreeAIBox block membership, inverse mappings or tensor semantics.
The existing public source audit already excludes those files. Merely being C++/CUDA, original
integration work or previously qualified does not establish authority to relicense them.
Consequently none of the private native source, headers, binaries, trained weights, ONNX or
TensorRT plans is included. Their historical test results do not qualify this implementation.
Per-file excluded-source identities and dispositions are recorded by `tools/audit_byom.py`.

The new `native/` implementation uses a generic, declared contract: a regular 3D grid anchored
at the source minima, half-open blocks, NCZYX tensors, sorted per-voxel feature reduction and
explicit first/last overlap resolution. These are general techniques, implemented here from
the documented contract and public NVIDIA APIs. It deliberately does not promise reproduction
of the frozen reference model's preprocessing. The original Conv3D fixture uses analytic
constants with no training or third-party weights. New code, tests and example sources are
assigned MIT within the owner's requested scope; documentation retains CC BY 4.0.

This is **not a clean-room independence claim**. Design and evidence from the prior project
were reviewed, including its existing public source-review disclosures. The defensible boundary
is retention of explicitly licensed public code, exclusion of uncertain private code and
restricted assets, and new generic implementations whose input specification and method are
recorded. No legal conclusion about excluded private code is asserted.

Third-party boundaries:

| Dependency | Treatment |
|---|---|
| laz-perf 3.4.0 | External Apache-2.0 source, compiled into native executables; original notices preserved separately. |
| nlohmann/json 3.11.3 | External MIT headers compiled into runtime; Niels Lohmann notice included separately. |
| OpenSSL 3 | System Apache-2.0 library, EVP SHA256; no source/binaries vendored. |
| CUDA and TensorRT 10 | NVIDIA SDKs and runtime libraries under vendor terms; external dependencies, never relicensed as MIT. |
| ONNX, ONNX Runtime | Apache-2.0 / MIT development tools for export and independent CPU oracle; absent from the classification process. |
| laspy, lazrs, NumPy | Existing separately licensed synthetic fixture/verifier dependencies; absent from the classification process. |

`LICENSE_SCOPE.json` enumerates original licensed files. `evidence/byom_provenance.json`
records per-file hashes and scope, including retained public code and new code. Generated
evidence describes facts; copyright is claimed only in original expression. Hashes identify
reviewed bytes, not ownership, legal clearance or authorship proof. MIT does not extend to
user-supplied model packages, NVIDIA libraries or third-party source.
