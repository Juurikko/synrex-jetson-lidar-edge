# R4 source and QGIS showcase release candidate

<!-- Copyright (c) 2026 Tuomas Pietilä. SPDX-License-Identifier: CC-BY-4.0. -->

Local candidate for explicit owner publication approval. R4 adds a convenient verified
COPC/QGIS export, an executed reader-lifecycle diagnosis, final public presentation and
a separate future-development proposal. It changes no native inference code, model
weights, preprocessing, frozen predictions, dataset partitions or accepted benchmark.

R3 remains immutable: tag `byom-unet-fractal-r3-rc1`, source
`77ed8401bcb48e89d53ed0edf3b62b652b678f98`, audit
`6896277bc39eee20a32ee69a41173af29f751b8c`, archive SHA256
`b57436b963412dbacce4ca1edd972a98dd1e07df7553638af3306c4239f6592c`.
The current candidate is a descendant, with exact source/artifact identities recorded in
`evidence/unet_r4_release_candidate.json` after archive verification. No old tag/archive
is overwritten. Large datasets, weights, ONNX and plans stay in ignored release packages.

Open `qgis/ready/synrex-fractal.qgz` from the new archive, or follow the
[one-command reconstruction](QGIS_R4.md). The original R3 `qgis/direct` and `qgis/copc`
input packages are retained in the archive alongside the ready project. All ten frozen
native scenes and their authoritative semantic sidecars remain independently verifiable.
The ready COPCs carry original source indices plus all seven semantic IDs, and their
reordered decoded point values match the native classified files exactly.

Build the public source using [these dependency/build commands](RECONSTRUCTION_R4.md).
For CPU packaging/verification, install `requirements-dev.lock` and `requirements-byom.lock`.
Classification requires the existing matching CUDA/TensorRT 10 environment and a verified
nonconflicting GPU allocation. No private engineering repository is needed. NVIDIA SDKs,
PDAL and QGIS are external dependencies and are not bundled or silently installed.

For the original synthetic model from scratch, use the complete
[Conv3D exporter/package/engine/fixture qualification commands](QUICKSTART.md).
The release also contains its accepted model, input, expected labels and native output.
For the trained model use the frozen `models/unet-fp16.onnx`; build a new engine on the
allocated target when the included RTX plan does not meet its target contract:

```bash
build-gpu/synrex-native pack --spec examples/unet/dense_voxel_asprs_safe.json \
  --onnx "$RELEASE/models/unet-fp16.onnx" --license examples/unet/MODEL_NOTICE.txt \
  --out build/unet-onnx.srxmodel
build-gpu/synrex-native build-engine --model build/unet-onnx.srxmodel \
  --precision fp16 --out build/unet.engine
build-gpu/synrex-native pack --spec examples/unet/dense_voxel_asprs_safe.json \
  --engine build/unet.engine --license examples/unet/MODEL_NOTICE.txt \
  --out build/unet-target.srxmodel
```

To prove interchangeability with compatible packaged engines, keep the same executable
and change only the model/input/output arguments. Outputs must be fresh paths:

```bash
build-gpu/synrex-native inspect --model "$RELEASE/models/synthetic.srxmodel"
build-gpu/synrex-native inspect --model "$RELEASE/models/unet.srxmodel"
build-gpu/synrex-native classify --model "$RELEASE/models/synthetic.srxmodel" \
  --input "$RELEASE/data/synthetic/dense-synthetic.laz" --out build/conv3d-classified.laz
build-gpu/synrex-native classify --model "$RELEASE/models/unet.srxmodel" \
  --input "$RELEASE/data/source/TEST-0737_6255-006722653.laz" \
  --out build/unet-classified.laz --semantics build/unet-semantics.json
.venv/bin/python tools/seal_semantics.py \
  --source "$RELEASE/data/source/TEST-0737_6255-006722653.laz" \
  --output build/unet-classified.laz --sidecar build/unet-semantics.json \
  --package "$RELEASE/models/unet.srxmodel" --taxonomy examples/unet/semantic_classes.json \
  --destination build/unet-binding.json
```

If repackaged for a new target, substitute `build/unet-target.srxmodel` in the inference
and binding commands. New TensorRT builds may produce different artifact hashes and FP16
results within explicitly checked tolerances; they are not the byte-identical frozen RTX
artifact. Keep original package manifests and outputs intact. The
[training/export workflow and freeze protocol](UNET_R2_QUICKSTART.md) document the
original learning process; no further tuning against the consumed test is permitted.

| Frozen artifact | SHA256 |
|---|---|
| Learned weights | `788b65c42bb5d50d9509a56d38368f51890c4a93300d53578e290d193456c7c4` |
| FP16 U-Net ONNX | `4a043dfacefa9e45f2e80e604b565f7ae90fb35ff81831af8f26cfd3d59ae047` |
| FP32 U-Net ONNX | `316dfd710b455f2324a86e18f39c7df960525fe94fe3702579030f014887fb44` |
| RTX U-Net engine | `2f20d4fd17c1e1eb3399ac450f84a2f1c7737ab69e2098ae383dc1911a1455cd` |
| U-Net package manifest | `2d1206be62385669477a180d98c1b330c1caf40d516625f64682284012f7a325` |
| Original Conv3D ONNX | `e9a55b97e1c38e9be057f7f23f395be00418c80d6e160736af53e7a64d43067c` |

The accepted native FP16 seven-class mIoU is **48.756%**; complete RTX4090 throughput is
**237,991 points/s** across four untraced real scenes. These are existing verified
measurements, not new measurements of this export tool. See [full timings](PERFORMANCE_R3.md),
[class quality, confusion matrix and rare-class caveats](UNET_R2_RELEASE.md).
Historical SegFormer AGX results are not a controlled hardware/model/data comparison.
Physical AGX qualification remains NOT_RUN; its reserved Spatial Lab workload is untouched.

MIT covers the qualified original software and original learned parameters, subject to
preserved Wolny/pytorch-3dunet notices. Apache-2.0 laz-perf, MIT JSON, OpenSSL and NVIDIA
terms remain separate. IGN FRACTAL source/classified data, labels and derived geometry
retain Etalab 2.0 attribution and source-update/retrieval dates. Documentation/presentation
is CC BY 4.0 for original expression; QGIS/Qt interface rights are separate. There are no
restricted TreeAIBox assets or private-repository dependencies. Exposure to reference
work is disclosed in the [source provenance review](PROVENANCE_REVIEW_R1.md), not presented
as a clean-room process or legal certification.

Source release and artifact upload remain held for explicit owner approval. The prepared
README, release notes, technical showcase, screenshots, complete source bundle and model/
example packages are reviewable locally. Do not upload SDKs, test/train archives or private
references. Proposed streaming, sparse and Thor work is [separate](FUTURE_MILESTONES_R4.md).
