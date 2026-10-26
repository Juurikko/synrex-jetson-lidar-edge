# Third-party notices and dependency boundary

<!-- Copyright (c) 2026 Tuomas Pietilä. SPDX-License-Identifier: CC-BY-4.0. Original expression only; upstream material and facts excluded. -->

**Developer and project lead: Tuomas Pietilä ([@Juurikko](https://github.com/Juurikko)).**

No upstream library source, binary, dataset, model or font file is vendored. The `LICENSES/` copies below preserve notices for the libraries used in the synthetic demo and fonts used in the already-rendered media. Dependency installation is a separate step and does not relicense upstream material. For BYOM compilation, separate [NVIDIA API notices](LICENSES/NVIDIA-API-NOTICES.txt), [CUDA distribution license](LICENSES/NVIDIA-CUDA-License.txt) and [TensorRT SLA](LICENSES/NVIDIA-TensorRT-SLA.txt) are retained alongside the Apache and MIT texts; vendor binaries are not shipped.

| Material | Version/identity used | Upstream license / treatment |
|---|---|---|
| [laz-perf](https://github.com/hobuinc/laz-perf) | 3.4.0, `b7bbe26109dc986f42d4fc80b8de3d2b6ca634ce` | Apache-2.0; [license text](LICENSES/Apache-2.0.txt). The original upstream source headers identify contributors including Martin Isenburg and Uday Verma/Hobu. Fetched externally; original headers retained. |
| [laspy](https://github.com/laspy/laspy) | 2.7.0 | BSD-2-Clause; [notice](LICENSES/laspy-BSD-2-Clause.txt). Explicit laz-rs backend selected in independent verification. |
| [lazrs](https://github.com/laz-rs/laz-rs-python) | 0.8.2 | MIT; [notice](LICENSES/lazrs-MIT.txt). Python bindings to the Rust laz-rs implementation. |
| NumPy | 2.5.4 | Installed metadata reports BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0; wheel contains additional component notices. Not vendored; retain the installed distribution's full notices. |
| pytest / Ruff | Pinned in `requirements-dev.lock` | MIT; development dependencies, not vendored. |
| clang-format wheel | Pinned in `requirements-dev.lock` | LLVM/Clang upstream notices, Apache-2.0 with applicable LLVM exceptions; development tool, not vendored. |
| imageio-ffmpeg wheel / bundled FFmpeg | Pinned in `requirements-dev.lock` | MIT wrapper; bundled binary and codecs have separate FFmpeg notices/build licensing. Used locally for the GIF preview; no executable redistributed. |
| DejaVu font designs | Unmodified system fonts used by accepted media | Bitstream Vera permission notice and public-domain DejaVu changes; [complete notice](LICENSES/DejaVu-font-notice.txt). Rendered media only; fonts not vendored. |
| [NRCan TreeAIBox](https://github.com/NRCan/TreeAIBox/tree/5380ddec2799f55adb43b63407dd12ccf7abefae) | Frozen reference research commit | Inspected reference lineage identifies CC BY-NC 4.0. Code, weights, configuration, ONNX and TensorRT engine excluded; descriptive attribution only. |
| NVIDIA Jetson / CUDA / TensorRT | Physical qualification platform | Vendor products/licenses remain separate; no vendor libraries, logos, engine plans or product imagery bundled. |
| [nlohmann/json](https://github.com/nlohmann/json/tree/v3.11.3) | 3.11.3, `9cca280a4d0ccf0c08f47a99aa71d1b0e52f8d03` | MIT; [Niels Lohmann notice](LICENSES/nlohmann-json-MIT.txt). External headers compiled into native BYOM executables, source not vendored. |
| [OpenSSL](https://www.openssl.org/) | System OpenSSL 3 | Apache-2.0; external EVP SHA256 library. Retain distribution notices when shipping it. |
| [ONNX](https://github.com/onnx/onnx) | 1.17.0 development exporter | Apache-2.0; external Python package, absent from native classification. |
| [ONNX Runtime](https://github.com/microsoft/onnxruntime) | 1.20.1 independent CPU oracle | MIT; external development dependency, absent from native classification. Distribution component notices remain applicable. |

The BYOM R0 CUDA/TensorRT backend is original generic integration source under the scoped MIT grant. It calls vendor APIs and requires separately installed NVIDIA libraries. Local compiler/SDK wheels used for build validation stay in ignored dependency directories; no NVIDIA SDK source/binary is added to the release candidate. The original analytic Conv3D fixture has no third-party weights. Its generated model keeps explicit MIT metadata; user-supplied models retain their own terms. See [BYOM provenance](docs/byom/PROVENANCE.md).

The model's configured metadata has seven classes and its network eight channels; no new architecture or model ownership is asserted. Commercial inference requires appropriately licensed or SynRex-trained weights. Existing research, compression algorithms and vendor technologies remain attributed to their authors.

The source audit found no blanket public license covering the original private runtime sources. Those sources are not copied into this repository. Tuomas Pietilä has confirmed authority over the newly staged original source, documentation and media; scoped MIT / CC BY 4.0 grants apply to that work only. The design relationship remains disclosed and no excluded upstream/private implementation is relicensed.
