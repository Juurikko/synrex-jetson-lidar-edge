# DENSE_VOXEL_SEGMENTATION_V1 contract

<!-- Copyright (c) 2026 Tuomas Pietilä. SPDX-License-Identifier: CC-BY-4.0. Original expression only; upstream material and facts excluded. -->

The normative runnable example is [dense_voxel.json](../../examples/dense_voxel.json).
`Contract` validates all fields and rejects unknown fields. This adapter supports a single
static batch-one dense segmentation input and output. It does not promise arbitrary ONNX support.

## Package and tensor schema

`manifest.json` has exactly `schema`, `contract`, `artifacts`, `target`. Schema is
`synrex.srxmodel.v2`. Contract contains exactly:

| Field | Meaning |
|---|---|
| `adapter` | Exactly `DENSE_VOXEL_SEGMENTATION_V1` |
| `input`, `output` | Each has `name`, `shape`, `dtype`, `layout` |
| `geometry` | Sizes, dimensions, overlap and overlap policy below |
| `features` | Ordered input channel definitions |
| `las_classes` | Ordered complete uint8 LAS mapping for output channels |
| `limits` | Positive bounded resource budgets |
| `licensing` | Nonempty `spdx`, `author`, `source`, `weights`, `notice` strings |

Tensor layout is contiguous `NCZYX`, index `c*V + (z*Dy+y)*Dx+x`. Shape is
`[1,F,Dz,Dy,Dx]` for input and `[1,K,Dz,Dy,Dx]` for output. F is the number of features
(1..32); K is the number of class-map entries (2..256). Names must be distinct, nonempty,
<=127 bytes and NUL-free. Dtype is independently `float32` or `float16` for each tensor.
Actual TensorRT tensors must match names, direction, dimensions and dtype and have linear
device storage. Dynamic shapes, auxiliary tensors, packed formats, host tensors, integer I/O
and user plugins are unsupported. Network internals may use any standard operators that the
installed TensorRT parser supports, subject to these exact I/O semantics.

`artifacts` has exactly `license` and one of `onnx`/`engine`. Each record contains exactly
`path` and `sha256`. Only fixed basenames `MODEL_LICENSE.txt`, `model.onnx`, `model.engine`
are allowed. Extra members, directories, symlinks, duplicate JSON keys, path traversal, empty
artifacts, artifacts >2 GiB and JSON >1 MiB are rejected. ONNX packages have null target;
engine packages have exactly the target fields listed in the quickstart. ONNX graphs must
be self-contained. Models/packages are trusted inputs, not a sandbox for hostile code or graphs.
Legacy tar packages are rejected. Package creation stages beside its destination and uses Linux
`renameat2(RENAME_NOREPLACE)`; it never replaces an existing package.

## Geometry and features

`geometry` has exactly `voxel_size_xyz`, `dimensions_xyz`, `block_size_xyz`, `overlap_xyz`,
`overlap_policy`. Each triple follows XYZ order. Voxel sizes are positive finite doubles;
dimensions are positive integers <=4096, with total V <=268,435,456. Block size equals
voxel size times dimensions within relative 1e-10. Overlap per axis is a fraction in [0,0.9].
Stride is `block_size * (1-overlap)`. Axes use the source's minimum scaled coordinate as grid
origin. Block indices are nonnegative integers. Only nonempty blocks are retained, in ascending
lexicographic `(bx,by,bz)` order. Geometry uses double precision, including LAS scale/offset.

Membership is half-open: `0 <= point-origin < block_size` on every axis. Voxel coordinates are
`floor((point-origin)/voxel_size)`, capped at `dimension-1` only to handle rounding at the upper
floating boundary. The sparse block-discovery loop uses conservative candidates and exact
membership tests; it does not clamp out-of-block points into a tensor. It fails if any source
point is uncovered or its block-index range is >=1e9. Every source record has an immutable
zero-based source index; coincident points remain distinct records. Within each block members
are sorted by linear voxel index and then source index, retaining one membership per source point.

Each feature has `kind`, `offset`, `scale`. Supported kinds:

| Kind | Value in an occupied voxel |
|---|---|
| `occupancy` | 1 |
| `count` | Number of member points |
| `mean_intensity` | Mean LAS uint16 intensity (0..65535) |
| `local_x`, `local_y`, `local_z` | Mean scaled coordinate relative to this block's origin |

CUDA sums use double precision in ascending source-index order, cast the reduced value to float,
then apply `(float(value)-offset)*scale` with finite float normalization constants and nonzero
scale. Tensor storage casts to FP16 where specified; overflow/nonfinite features cause failure.
**Empty voxels contain zero in every channel**, irrespective of feature normalization.
No stochastic sampling, cropping, training-time augmentation or hidden feature transform occurs.
The CPU reference tensor is a test/diagnostic implementation, never a classification fallback.

## Reconstruction and output semantics

For each occupied point's voxel, CUDA selects the largest class logit. Exact ties select the
lowest channel index. Nonfinite occupied-voxel logits cause the invocation to fail. Model
indices map through `las_classes`, with no implicit shifts or default ground/background class.
Duplicate mapping values are allowed. Source-index scatter operates on distinct destination
indices within a block. Sequential block execution defines `first_wins` and `last_wins`:
first_wins keeps the earliest block's mapped class; last_wins replaces it in later blocks.
No averaging, confidence voting or class smoothing is performed.

The label buffer starts with an out-of-range sentinel; all source points must be reconstructed
before writing output. All other point bytes, source order, source public header, non-LASzip
VLRs, Extra Bytes descriptors and metadata padding are preserved under the public writer's
narrow LAS contract. Compression VLR/chunk table bytes can change. Refer to the
[existing architecture](../../ARCHITECTURE.md) and [verifier](../../src/verify_laz.py).

## Resource and execution bounds

`limits` contains exactly `max_points` <=20,000,000, `max_blocks` <=100,000,
`max_memberships` <=100,000,000, `max_tensor_bytes` <=2 GiB. All are positive integers.
Tensor accounting conservatively uses `4*V*(F+K)` even for FP16 I/O. The file and plan reside
in host memory; device points, one input/output tensor pair and reusable largest-block buffers
reside in GPU memory. The runtime checks free memory before those allocations, retaining at least
64 MiB headroom; TensorRT context memory is additional. Allocation/runtime errors fail before
LAZ publication. These bounds do not establish coexistence with another GPU workload.

TensorRT API usage follows NVIDIA's [C++ API documentation](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/c-api-docs.html)
and [10.x named-tensor execution migration guidance](https://docs.nvidia.com/deeplearning/tensorrt/latest/api/migration/tensorrt-8x-to-10x-c-api-patterns.html).
SDK APIs are dependencies; no NVIDIA sample implementation is copied into project source.
