# Frozen FRACTAL U-Net R2 release qualification

<!-- Copyright (c) 2026 Tuomas Pietilä. SPDX-License-Identifier: CC-BY-4.0. -->

The 256-tile frozen test milestone passes its predeclared engineering gates.
This release preparation changes only the LAS projection and adds verification,
packaging and presentation. It does not retrain, select a different checkpoint,
change preprocessing, or change model semantic predictions. Publication remains
held for explicit owner approval. AGX physical execution is NOT_RUN.

The original [Conv3D RTX qualification](RTX4090_QUALIFICATION.md), accepted CPU
checkpoint `eaf3ce2` and [R1 U-Net pilot](UNET_QUALIFICATION.md) are preserved.
R1 results and historical SegFormer measurements are not R2 measurements.

## Frozen lineage and bounded data

[Protocol](../../evidence/unet_r2_protocol.json) predates fresh test access;
[freeze](../../evidence/unet_r2_model_freeze.json) records epoch28 selected using
validation macro-F1, from a fixed40-epoch budget. The MIT upstream architecture
is pinned to `7d047d2802561d075d6ce44679d7e696138c47df`: UNet3D, DoubleConv,
channels8/16/32, three levels,62,103 parameters. Training source is `d7c8514`.
No pretrained parameters were used. The original test00 pilot was already consumed;
the fresh evaluation uses test01 only, selected/downloaded after this freeze.
This accepted test is now consumed; it must never be used for tuning this model.

| Official partition | Selected tiles | Original points | Archive/selection |
|---|---:|---:|---|
| Train | 1,000 | 91,309,139 | train00, first sorted members |
| Validation | 256 | 21,169,981 | val00, first sorted members |
| Frozen test | 256 | 24,902,618 | test01, evenly spaced sorted members |

[Selection paths, CRCs and hashes](../../evidence/unet_r2_dataset.json),
[inventory/class support](../../evidence/unet_r2_inventory.json),
[training configuration](../../evidence/unet_r2_training.json),
[history](../../evidence/unet_r2_training_history.json),
[preparation](../../evidence/unet_r2_preparation.json) and
[pre-freeze point validation](../../evidence/unet_r2_validation_points.json)
are retained. Partial HTTP ZIP retrieval does not establish a full archive hash;
exact extracted-member SHA256s are checked instead. No whole-dataset download.

Independent [duplicate audit](../../evidence/unet_r2_duplicate_audit.json)
finds1,512 unique compressed hashes and1,512 unique sorted physical XYZ hashes,
including within test. Sorting detects duplicates with reordered points/different
headers. Inventory finds no positive-area XY overlap across selected official
splits. These checks support no observed geographic/sample leakage in the selected
subset; they are not a survey-wide spatial-independence guarantee.

## Exact inference contract and label exclusion

[Final contract](../../examples/unet/dense_voxel_asprs_safe.json) uses static
input `features [1,4,32,32,32]` and output `scores [1,7,32,32,32]`, FP16 NCZYX.
One-metre voxels,32-metre regular XYZ blocks anchored at each cloud's XYZ minima,
zero overlap, first-wins policy. Tensor index is `(z*32+y)*32+x`; coordinates use
half-open cells. Features in order: occupancy, count/16, mean local Z/32,
mean intensity/65535. Empty cells are zero. Height is relative to block origin,
not label-derived terrain. RGB/NIR and every classification/reference field are
excluded. Each source point retains its immutable record index and receives its
occupied voxel's predicted output channel. No points are filtered from output.

Supervision uses occupied-voxel majority semantic ID, ties to the lowest ID,
ignores source artifacts65/66, inverse-square-root voxel weights capped20 and
normalized. Half uniform/half class-presence block sampling addresses rare classes.
Mixed-label voxel collisions remain a limitation; the point-level voxel-majority
upper-bound accuracy is96.880%, above the actual90.814% model accuracy.
Training's MaxPool3d backward warns of nondeterminism; seeded training is repeatable
as a workflow, not a promise of bit-identical independently retrained weights.

All1,206 actual CUDA input tensors match independent NumPy feature bytes exactly
([download audit](../../evidence/unet_r2_download_validation.json)). The native
reference-label-erasure test produces identical semantic IDs after source labels
are zeroed. Ground-truth classification is read only for training targets, scoring,
and disclosed post-freeze display selection, never for inference features.

## Semantic IDs and LAS projection

[Authoritative taxonomy](../../examples/unet/semantic_classes.json) preserves all
seven output channels. This correction changes only the LAS lookup table; all256
outputs retain exactly the frozen semantic IDs. The historical IGN projection
is retained as historical evidence, not the final recommended projection.

| Model ID | FRACTAL meaning | Original source codes | Final LAS code |
|---:|---|---|---:|
| 0 | Other/unclassified | 1 | 1 |
| 1 | Ground | 2 | 2 |
| 2 | Aggregate vegetation | 3,4,5 | 200 (user-defined) |
| 3 | Building | 6 | 6 |
| 4 | Water | 9 | 9 |
| 5 | IGN bridge | 17 | 17 (bridge-deck projection) |
| 6 | Permanent aboveground structure | 64 (IGN convention) | 201 (user-defined) |

[ASPRS LAS1.4 R15, Table17](https://www.asprs.org/wp-content/uploads/2019/07/LAS_1_4_r15.pdf)
defines5 as high vegetation,17 as bridge deck, and64–255 as user-defined.
Aggregate vegetation therefore does not use5. IGN's64 does not become a universal
ASPRS structure category. Bridge17 is a documented projection of the dataset's
source17; it does not establish bridge subcomponent accuracy. The full ID5 in the
sidecar remains IGN bridge. Use the sidecar for faithful semantic analysis.
Codes200/201 require the supplied legend; default LAS viewers may show them as
unknown. No classification lookup VLR exists in the selected test inputs, so no
conflicting embedded legend is carried into these outputs.

Native `--semantics` writes `synrex.point_semantics.v1`: source SHA256, package
manifest SHA256, exact source record order, full IDs0–6 and mapped-label SHA256.
The independently produced `synrex.semantic_output_binding.v1` envelope adds the
exact output LAZ, sidecar, taxonomy and package hashes after laz-rs verification.
It leaves both native artifacts untouched. Hashes provide integrity/binding,
not digital signatures or proof of authenticity. Keep each LAZ, sidecar and
binding together. [Final independent audit](../../evidence/unet_r2_final_output_validation.json)
checks the explicit policy independently of the package's own lookup, all seven
predicted classes, and all256 source/output pairs.

## Native FP16 quality and numerical agreement

[Independent recomputation](../../evidence/unet_r2_independent_metrics.json) reruns
the frozen PyTorch FP32 model, compares actual native trace predictions and
sidecars, and uses a separately implemented confusion-matrix reduction. It
reproduces the accepted matrices exactly. Rows are reference classes, columns
predicted classes; full7×7 matrices and per-file prediction hashes are in that receipt.
Source artifacts65/66 (1,447 points) are excluded only from scoring, never output.

| Class | Test support | Native FP16 IoU | FP16 F1 | FP16/FP32 disagreement by truth |
|---|---:|---:|---:|---:|
| Other | 214,815 | 16.582% | 28.447% | 0.0740% |
| Ground | 9,751,207 | 83.671% | 91.109% | 0.0516% |
| Vegetation | 13,897,305 | 88.194% | 93.727% | 0.00836% |
| Building | 758,113 | 69.393% | 81.931% | 0.0426% |
| Water | 242,219 | 33.906% | 50.642% | 0.4579% |
| Bridge | 26,677 | 17.214% | 29.372% | 0.4161% |
| Permanent structure | 10,835 | 32.333% | 48.866% | 0.1754% |

Native point mIoU **48.756%**, macro-F1 **60.585%**, overall accuracy **90.814%**.
The frozen PyTorch FP32 values are48.759%,60.597%,90.803% respectively.
Fixed geometry baseline macro-F1 is20.181%; training-majority baseline10.234%.
All seven classes have positive support and IoU≥.05. This passes the declared
mIoU≥.40 and macro-F1≥best baseline+.10 gates, but does not establish production
accuracy. Bridge/other recognition remains weak and rare support is small.

7,917/24,902,618 point IDs differ (agreement99.968208%). Water, bridges and
structures have disproportionately higher disagreement rates. Bridge IoU falls
**0.1634 percentage points**, vegetation0.00215 points; other classes improve
slightly. Report these differences rather than claim bitwise FP16 equivalence.
Logit RMSE0.008799 and max absolute error0.171763; PyTorch FP32 versus ONNX FP32
max absolute error0.000041008. Declared gates are RMSE<.03, ONNX max<.001 and
point agreement≥.995. The actual native FP16 logits also match separate execution
of the identical engine exactly. That latter comparison is not FP32 equivalence.

## Untraced application benchmarks

[Raw cold and persistent receipts](../../evidence/unet_r2_benchmarks.json) contain
four real scenes, three fresh-process invocations each, then three warmup and ten
measured cycles in a persistent engine/context/stream/buffer allocation. RTX4090,
x86_64 WSL, CUDA13.2, TensorRT10.16.1.11, no compute competitors observed; desktop
baseline130MiB retained. No target bulk transfer ran during final measurements.
Normal clocks/power and warm filesystem caches were retained; no cache flush,
clock locking or exclusive-machine assertion. No tensor tracing or loader tracing
in these timings; no sidecar serialization in benchmark outputs. Native commands
use PATH=/nonexistent. Every cold output is independently verified; repeats are
byte-identical. Warm cycles check identical LAS and raw semantic vectors and100%
point coverage.

| Real scene suffix | Points | Mean fresh-process E2E ms | TRT ms | Persistent processing ms | Cold Mpoint/s | Cached processing Mpoint/s |
|---|---:|---:|---:|---:|---:|---:|
| 0447_6390 | 50,096 | 323.147 | 0.901 | 1.458 | .155 | 34.36 |
| 0922_6376 | 97,258 | 352.595 | 0.937 | 1.629 | .276 | 59.71 |
| 0944_6564 | 400,654 | 894.506 | 1.543 | 3.723 | .448 | 107.62 |
| 0947_6571 | 76,649 | 363.792 | 0.929 | 1.505 | .211 | 50.93 |

| Mean cold phase, ms | 50,096 pts | 97,258 pts | 400,654 pts | 76,649 pts |
|---|---:|---:|---:|---:|
| LAZ decode | 23.582 | 35.222 | 219.915 | 43.381 |
| CPU geometry/voxel membership | 6.615 | 13.052 | 56.723 | 10.062 |
| CUDA tensor reduction | 1.395 | 1.689 | 1.752 | 1.390 |
| Engine/context init + initial upload | 248.365 | 234.994 | 227.623 | 226.130 |
| TensorRT inference | .901 | .937 | 1.543 | .929 |
| CUDA argmax + source-index reconstruction | .0253 | .0268 | .0567 | .0259 |
| LAZ preservation/encoding | 36.961 | 59.527 | 367.914 | 72.747 |

CUDA postprocessing **is the fused argmax/map/source-index reconstruction kernel**;
it has one CUDA-event interval, not separable per-operation timings. Transfers,
host vector construction, validation, scratch writing, hashing and launch overhead
are included in E2E but not all individually instrumented. Initial upload is
included in initialization. Phase sums therefore need not equal E2E. Persistent
mean CUDA prep/TRT/reconstruction ms are respectively .141/.576/.0198,
.102/.531/.0224, .259/1.075/.0527, .0762/.569/.0206.
Persistent throughput excludes cached decode/geometry and all LAZ encoding;
it must not be advertised as application throughput or a streaming service.

Tracked peak requested CUDA bytes:6,600,535 /8,954,015 /23,998,035 /8,016,215.
This includes adapter buffers and TensorRT allocation callbacks, excluding CUDA
context and vendor allocations outside callbacks. Device-wide sampled maximum
is518/520/518/518MiB (desktop included), sampled with nvidia-smi roughly60–80ms
apart despite a20ms sleep request. This is a sampled lower bound on resident peak,
not an exact total resident high-water mark. No synthetic/SegFormer extrapolation.

## Preservation, interchangeability and rights

All256 real sources use LAS1.4 PDRF8:145 records of38bytes,111 of48bytes.
The final [download verification](../../evidence/unet_r2_safe_download_validation.json)
and hash-bound audit explicitly use laz-rs, independent of native laz-perf.
Original record order, every decompressed byte except classification byte16,
including flags/intensity/RGB/NIR/extras, and qualified metadata are exact.
No reference labels are copied into classified outputs: full neural tensor traces,
nonconstant predictions, disagreement with truth and label-erasure invariance
provide direct evidence. Original compressed source stays separately retained;
no normalization or LAS conversion occurred.

Supported input remains compressed extended LAS1.4 with standard375byte header,
PDRF6/7/8 and bounded declared point/tensor limits; EVLR, waveform, COPC, legacy
LAS, unsupported compression/metadata, malformed values and oversized inputs
are rejected. Success here does not expand that contract to arbitrary LAS.

The same native binary executes the frozen synthetic Conv3D engine and trained
U-Net engine without recompilation. Its SHA256 is
`a604487922198add17f2a6004529e12702c3625629c339b6365e5ccb6f704d83`.
No Python/PyTorch libraries appear in any native classification loader log.
Training, export, scoring and independent validation are separate development
processes. Static dense support does not imply sparse-transformer compatibility.

[Rights assessment](UNET_PROVENANCE.md),
[preprocessing review](PROVENANCE_REVIEW_R1.md),
[third-party notices](../../THIRD_PARTY_NOTICES.md) and per-file provenance remain
explicit: original runtime MIT, attributed upstream architecture MIT, new
from-scratch weights MIT with IGN source/date attribution, real survey information
Etalab2.0, original documentation CC-BY4.0, NVIDIA plans/SDK under separate terms.
No private code/model or restricted pretrained asset is included. This is a scoped
technical provenance review, not independent legal clearance or clean-room proof.

The [quickstart](UNET_R2_QUICKSTART.md) reproduces the frozen procedure and exports.
[QGIS style](../../examples/unet/fractal-classification.qml) uses a point-cloud
classified renderer and explicit200/201 legends. Static XML/taxonomy checks PASS;
QGIS application loading is NOT_RUN because no QGIS is installed on the hosts.
[Actual source/prediction plots](../../media/unet-r2/heldout-semantics.png) show
all points without interpolation; scenes were selected for class support after
freeze, not for model selection. [Display manifest](../../evidence/unet_r2_showcase_media.json)
records exact source/output hashes and selection. The bridge example visibly
shows missed bridge points; this is intentionally not hidden.

A separate [streaming milestone](STREAMING_MILESTONE.md) and
[LitePT/TorchSparse++ assessment](LITEPT_SPARSE_PLAN.md) follow this dense release.
AGX qualification still requires a verified nonconflicting physical allocation;
its stack/workloads were not modified. Owner publication approval remains pending.
