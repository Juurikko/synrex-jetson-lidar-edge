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
| NVIDIA Jetson / CUDA / TensorRT | Physical qualification platform | Vendor products/licenses remain separate; no vendor SDK libraries, logos or product imagery bundled. Generated target-specific model plans are separate release assets under their model/SDK notices, never ordinary source history. |
| [nlohmann/json](https://github.com/nlohmann/json/tree/v3.11.3) | 3.11.3, `9cca280a4d0ccf0c08f47a99aa71d1b0e52f8d03` | MIT; [Niels Lohmann notice](LICENSES/nlohmann-json-MIT.txt). External headers compiled into native BYOM executables, source not vendored. |
| [OpenSSL](https://www.openssl.org/) | System OpenSSL 3 | Apache-2.0; external EVP SHA256 library. Retain distribution notices when shipping it. |
| [ONNX](https://github.com/onnx/onnx) | 1.17.0 development exporter | Apache-2.0; external Python package, absent from native classification. |
| [ONNX Runtime](https://github.com/microsoft/onnxruntime) | 1.20.1 independent CPU oracle | MIT; external development dependency, absent from native classification. Distribution component notices remain applicable. |
| [pytorch-3dunet](https://github.com/wolny/pytorch-3dunet) | `7d047d2802561d075d6ce44679d7e696138c47df` | MIT, copyright Adrian Wolny; [notice](LICENSES/pytorch-3dunet-MIT.txt). External authentic UNet3D architecture, used only for training/export/validation. No pretrained weights used. |
| [squeeze/excitation](https://github.com/ai-med/squeeze_and_excitation) | Module imported by upstream pytorch-3dunet; no SE block instantiated/exported | MIT, copyright Abhijit Guha Roy; Anne-Marie Rickmann credited for 3D modules; [notice](LICENSES/squeeze-excitation-MIT.txt). No source vendored. |
| [FRACTAL](https://huggingface.co/datasets/IGNF/FRACTAL) | `36e00245ef92fd180f655f6ee410f2e935693cc2`, retrieved 2026-10-26 | IGN, Etalab-2.0; [licence](LICENSES/Etalab-2.0.md). Real LiDAR data/classified examples retain this licence and attribution separately. Dataset files are not committed. |
| PyTorch / scikit-image | 2.9.1+cu128 / 0.25.2 U-Net development environment | External training/export dependencies; retain each distribution's component notices. Absent from native classification. |

The U-Net environment also uses laspy 2.6.1, lazrs 0.8.0, ONNX 1.20.1 and ONNX Runtime
1.23.2 under the corresponding upstream licences above. This separate environment
does not replace the original CPU verification lock. TensorRT Python bindings are
used only for numerical qualification of the actual engine; classification uses
the native executable. New trained weights and generated models carry
[MODEL_NOTICE.txt](examples/unet/MODEL_NOTICE.txt), including MIT architecture
attribution and IGN data attribution. See [rights assessment](docs/byom/UNET_PROVENANCE.md).

The BYOM R0 CUDA/TensorRT backend is original generic integration source under the scoped MIT grant. It calls vendor APIs and requires separately installed NVIDIA libraries. Local compiler/SDK wheels used for build validation stay in ignored dependency directories; no NVIDIA SDK source/binary is added to the release candidate. The original analytic Conv3D fixture has no third-party weights. Its generated model keeps explicit MIT metadata; user-supplied models retain their own terms. See [BYOM provenance](docs/byom/PROVENANCE.md).

The historical excluded reference model's configured metadata has seven classes and
its network eight channels. This does not describe the separately trained seven-output
FRACTAL U-Net. Commercial inference requires appropriately licensed or SynRex-trained
weights. Existing research, compression algorithms and vendor technologies remain
attributed to their authors.

The source audit found no blanket public license covering the original private runtime sources. Those sources are not copied into this repository. Tuomas Pietilä has confirmed authority over the newly staged original source, documentation and media; scoped MIT / CC BY 4.0 grants apply to that work only. The design relationship remains disclosed and no excluded upstream/private implementation is relicensed.

R3 viewing interoperability additionally uses existing external **PDAL2.10.0**
(git version `eb9126`) under its
[BSD-3-Clause licence](https://github.com/PDAL/PDAL/blob/2.10.0/LICENSE.txt),
preserved in `LICENSES/PDAL-BSD-3-Clause.txt`. PDAL stages separate COPC viewing
copies; it is not linked into the native classification executable. The original
classified LAZ and its full metadata remain separate, hash-bound artifacts.

Existing **QGIS3.44.14-Solothurn** and its **Qt5** installation provide the desktop
viewer and genuine screenshot interface. They remain separately licensed external
applications; no QGIS/Qt binary or implementation source is redistributed here.
See [QGIS licensing](https://qgis.org/license/) (GPL2 or later) and the licences
in the user's installed distribution for its complete Qt/dependency terms.
QGIS/Qt interface elements in screenshots retain upstream rights. CC-BY4.0 covers
only the original presentation; IGN survey information remains Etalab2.0.

R4's optional QGIS export and original PDAL lifecycle diagnostic are outside the native
classification execution path. They invoke/link separately installed PDAL (BSD-3-Clause)
and QGIS (GPL-2.0-or-later) APIs; no QGIS/PDAL binaries or diagnostic SDK headers/import
libraries are bundled. Actual QGIS/Qt screenshots retain upstream UI rights, IGN FRACTAL
Etalab-2.0 data attribution and CC-BY-4.0 original presentation credit to Tuomas Pietilä.
The native classifier, writer, pretrained-asset exclusion and frozen model rights are
unchanged. See `docs/byom/QGIS_R4.md` for the independent conversion/reader evidence.

Streaming R2 uses external **Pillow 11.3.0** (HPND), **pyproj 3.7.2** (MIT, with
external PROJ/component notices), and the already attributed NumPy, laspy, lazrs,
DejaVu fonts and FFmpeg for offline verification and rendering. Their executable
libraries and fonts are not redistributed. Pillow/pyproj are absent from native
inference. The media contains licensed IGN geometry and genuine native predictions;
it is not procedural scenery. See the separate
[R2 rights manifest](docs/streaming/STREAMING_R2_RIGHTS.md).

RC2–RC4 offline tooling membership is reconciled in `LICENSE_SCOPE.json` against
its original additive grants and author provenance. RC2 also imports external
Numba/llvmlite (BSD-2-Clause), documented in
`docs/streaming/rc2/LICENSE_SCOPE_RC2.json`; no compiler/library binaries or source
are redistributed. Their installed distributions retain their complete notices.
Historical owner-review packagers may create private-history bundles; they are
preserved as original source and are **not** the public publication procedure.
The public package contains no Git bundle and no unfiltered review archive.
