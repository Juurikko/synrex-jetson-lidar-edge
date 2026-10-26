# Separate LitePT-S sparse-adapter investigation

<!-- Copyright (c) 2026 Tuomas Pietilä. SPDX-License-Identifier: CC-BY-4.0. -->

This is a proposed next milestone, separate from the qualified dense U-Net and
Conv3D adapter. No sparse adapter or learned LitePT run is claimed here.

## Existing TorchSparse work comes first

The owner's GitHub replacement project was reviewed read-only, including its
source lock, patch series and qualification status. Its upstream is MIT
[TorchSparse](https://github.com/mit-han-lab/torchsparse), version 2.1.0 at
`385f5ce8718fcae93540511b7f5832f4e71fd835`. This is useful existing engineering
to retain and qualify; recreating a sparse backend from scratch is not proposed.

The reviewed records show a built 12-patch ARM64/SM87 wheel, repaired native
operators and scoped numerical tests across ImplicitGEMM, GatherScatter and
FetchOnDemand. The 13th patch adds generative-padding/duplicate guards, but a fresh
13-patch wheel is still unbuilt in those records. Native operator progress is not
learned semantic-model, accuracy, end-to-end performance or spconv parity evidence.

Important limits are explicit: dilation greater than one is rejected; odd FP16
channel widths with nontrivial kernel volumes are rejected; tested aligned FP16
widths do not establish arbitrary-width support. Kernel-map modes can have
different strided-padding semantics. Generative nonzero padding and duplicate
coordinates require the newer guards. Cache identities must include coordinates,
stride, padding, map mode, training/dataflow/sort settings. Negative-coordinate
support is narrowly scoped and opt-in. Coordinate snapshots add unmeasured costs.

Upstream's [MIT notice](https://github.com/mit-han-lab/torchsparse/blob/385f5ce8718fcae93540511b7f5832f4e71fd835/LICENSE)
is explicit. The reviewed overlay root had no licence file. Establish per-patch
authorship and redistribution permission, dependency/ABI locks and a complete
licence inventory before importing patches into the MIT runtime. No private
overlay source is copied into this project or its public showcase. Historical
operator receipts are not represented as new tests executed in this assignment.

## LitePT-specific gap

The [official LitePT source](https://github.com/prs-eth/LitePT) is MIT. Its
documented environment uses spconv and PointROPE; the architecture combines sparse
convolutions, hierarchical point operations and attention. Its
[official model card](https://huggingface.co/prs-eth/LitePT) marks the released
benchmark checkpoints MIT, but each chosen checkpoint, training dataset and any
redistribution conditions must still be locked separately. An airborne LiDAR
checkpoint from a third-party plugin is not covered merely by architecture terms.
Prefer a new FRACTAL-trained LitePT-S for the showcase if rights or taxonomy do
not match. No checkpoint is downloaded for this planning investigation.

An ARM64 Python wheel with native CUDA operators still depends on PyTorch/ATen and
its Python integration. It does not by itself meet the native runtime's execution
contract. A Python research worker can establish numerical feasibility under an
explicitly different contract; production integration must separately demonstrate
a C++ execution path without Python/PyTorch, including sparse kernels, attention,
PointROPE, allocation and scheduling. TensorRT plugin integration would require a
separate plugin ABI/licensing/build/target contract and tests. Dense ONNX success
does not establish sparse transformer support.

## Proposed sequence and exit gates

1. **Rights and source closure.** Pin LitePT-S, all operator dependencies, the
   existing patched TorchSparse source, trained parameters and dataset. Build the
   complete 13-patch artifact reproducibly in an isolated environment. Preserve
   upstream notices and establish overlay permissions. Do not change a live AGX stack.
2. **Operator semantic parity.** Enumerate every sparse operation actually used
   by LitePT-S, including coordinate ordering, stride, padding, kernel maps and
   weight layout. Compare against an explicitly licensed spconv reference and a
   small independent dense-coordinate oracle. Test forward values and, for
   training, input/weight gradients. Test odd widths, negative/duplicate/empty
   coordinates, cache reuse and integer overflow. Unsupported cases must reject.
   Checkpoint permutation and coordinate alignment are measured conversion steps.
3. **Learned-network parity.** Run identical licensed weights and preprocessed
   point sets through spconv and the repaired backend. Compare intermediate stages,
   final logits, source-point predictions, held-out confusion matrices and class
   IoU. Re-train where conversion semantics differ. No speedup claim before parity.
4. **Versioned sparse contract.** Design a new adapter identifier, provisionally
   `SPARSE_POINT_SEGMENTATION_V1`: int32 coordinates `[batch,x,y,z]`, NxF features,
   explicit voxel/coordinate origin and reductions, bounds, batch/stride/spatial
   range, duplicate policy, serialization/pooling inverses and source-point
   reconstruction. Define logits and active-coordinate identity together; an NxK
   output cannot be scattered by incidental row order. Pin backend/weight-layout
   ABI and target, hashes, precision and licence metadata in a versioned package.
   The existing dense contract remains a separate supported adapter.
5. **Native integration and qualification.** Prove the selected C++/CUDA execution
   route, with no Python/PyTorch classification dependency. Run real held-out LAZ,
   independent record/metadata verification, deterministic reconstruction, failure
   tests and relocated installation. Compare preprocessing, sparse-map creation,
   neural execution, postprocessing, writer, total latency/throughput and actual
   memory usage on the same data and target. Publish no borrowed performance numbers.
6. **Physical AGX window.** Only under a verified nonconflicting allocation, build
   for the installed compatible stack and repeat the complete gates. Preserve the
   Spatial Edge Lab autopilot. Missing allocation or prerequisites mean NOT_RUN.

The first deliverable is a reproducible backend/source and learned-parity report,
followed by a reviewable sparse adapter specification. Publication remains subject
to owner approval; this plan authorizes no push, stack change or AGX interruption.
