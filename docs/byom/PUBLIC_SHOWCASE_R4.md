# Native LiDAR semantics, two model packages, one runtime

<!-- Copyright (c) 2026 Tuomas Pietilä. SPDX-License-Identifier: CC-BY-4.0. -->

Local technical publication draft; publishing requires Tuomas Pietilä's explicit approval.

SynRex classifies georeferenced LiDAR using native C++, CUDA and TensorRT. One configurable
dense-voxel adapter runs both an original analytic Conv3D integration model and a compact
trained 3D U-Net, selected through SHA256-verified `.srxmodel` packages. The executable is
unchanged between models. Python/PyTorch are absent from native classification, while
training and independent verification use separate tooling.

The application decodes LAZ, builds geometry and occupied-voxel features, prepares tensors
on CUDA, executes TensorRT, reconstructs source-index predictions and writes classified
LAZ. Within its qualified LAS 1.4/PDRF 6–8 contract it preserves every unrelated decoded
point byte, source order and supported metadata. A seven-channel semantic sidecar avoids
confusing model ontology with LAS standard codes. Aggregate vegetation uses custom LAS
200, permanent structures 201, and IGN bridge retains its full semantic ID alongside the
bridge-deck projection LAS 17.

The original Conv3D has deterministic nonconstant analytic weights and a 50,013-point
GPU oracle. The 62,103-parameter U-Net uses the MIT-attributed Wolny/pytorch-3dunet
architecture with occupancy, count, local height and intensity. It was trained from
scratch on a bounded official geographically separated IGN FRACTAL selection. No source
classification or label-derived feature enters inference. The selected checkpoint and
fresh 256-tile test were frozen before accepted evaluation; the test is not a tuning set.

Across 24,902,618 held-out points, native FP16 achieves **48.756% seven-class mIoU**, 60.585%
macro-F1 and 90.814% accuracy. FP16 and PyTorch FP32 differ on 7,917 labels (0.0318%);
rare water/bridge classes have higher disagreement rates. Rare-class accuracy remains a
material limitation. These are measured semantic predictions, not copied reference labels.

Four untraced RTX4090 scenes total 624,657 points and 2.624704 seconds of launch-to-exit
time: **237,991 complete points/s**. LAZ decode/encode and process initialization dominate
much of this boundary. The cached persistent-context microbenchmark excludes file I/O
and CPU geometry and must not be advertised as full application throughput. Historical
SegFormer AGX timings used a different device, model, ontology, data and measurement
conditions; they do not establish a controlled comparison. Physical BYOM AGX execution
remains NOT_RUN because Spatial Lab's allocation is reserved.

![Actual QGIS desktop, frozen neural predictions](../../media/unet-r4/TEST-0741_6244-006630086-qgis.png)

A one-command optional export creates source-indexed COPC views and a relative-path QGIS
project. In the actual Windows QGIS application, all ten scenes, 1,062,717 points, all seven
semantic categories, per-point source indices and semantic/LAS mappings pass verification.
Independent laz-rs decoding restores COPC order and checks all original point bytes. The
source-preserving native files remain separate. The direct 48-byte failure is reproduced
as a PDAL reader reuse bug; it is not hidden by discarding Extra Bytes or metadata.

Developer and project lead: **Tuomas Pietilä**, with AI-assisted engineering by OpenAI Codex.
Original qualified software and original learned parameters are MIT within their stated
scope; Wolny architecture/upstream notices, Apache laz-perf, NVIDIA libraries/plans and
other dependencies retain separate terms. IGN FRACTAL source/derived data and semantics
remain Etalab 2.0: [official dataset](https://huggingface.co/datasets/IGNF/FRACTAL), source
update 2025-04-05T09:37:48Z, retrieved 2026-10-26. Original presentation is CC BY 4.0;
QGIS/Qt interface rights remain upstream. There is no clean-room claim and no imported
restricted TreeAIBox model. Future sparse/streaming/Thor proposals are separate milestones.

[Reproduce the release](RELEASE_R4.md) · [QGIS diagnosis/export](QGIS_R4.md) ·
[performance evidence](PERFORMANCE_R3.md) · [quality/protocol](UNET_R2_RELEASE.md) ·
[source and model/data rights](UNET_PROVENANCE.md).
