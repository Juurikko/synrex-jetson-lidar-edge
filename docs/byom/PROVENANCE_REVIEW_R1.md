# BYOM preprocessing provenance: second-pass technical review

<!-- Copyright (c) 2026 Tuomas Pietilä. SPDX-License-Identifier: CC-BY-4.0. Original expression only; upstream material and facts excluded. -->

Review date: 2026-10-26. Baseline: `eaf3ce2`. This is a separate source examination during
GPU qualification, performed with the project's AI engineering assistant. It is **not an
independent human/legal opinion or a clean-room certification**. The earlier authoring context
included private design/source exposure; AI assistance, translation and changed tensor sizes
do not remove derivative rights.

The review read the private frozen geometry and LAZ decode implementation, the generalized
SegFormer geometry/CUDA integration, their license records and the actual new public code.
Private files stayed read-only and are not transferred to the RTX4090 qualification workspace.

| Area | Frozen private implementation | New generic adapter |
|---|---|---|
| Block membership | Model-specific XY current/previous candidates, clipped block indices; boundary duplicate references | Sparse 3D point-driven tile discovery; conservative candidates and exact half-open membership on all axes; one source reference per block |
| Grid origin for voxelization | Minima of the points in each selected block | Regular tile origin anchored to whole-cloud minima |
| Tensor addressing | Frozen XYZ linear order and fixed model dimensions | Declared contiguous NCZYX, `(z*Dy+y)*Dx+x`, configurable sizes/shapes |
| Features | Frozen occupancy contract | Ordered feature reductions, including count/intensity/local coordinates and normalization |
| Reconstruction | Frozen unique/inverse arrays and compaction preserving the research model's duplicate behavior | Direct immutable source indices, per-voxel sorted groups, declared first/last overlap policy |
| LAS decoding | Standard integer XYZ, scale/offset and codec APIs | Standard LAS field decoding; audited public writer's narrow validation contract reused |

The shared concepts—sliding blocks, voxel occupancy, floor quantization, argmax and source-index
scatter—are disclosed design relationships. Matching these concepts is not itself evidence of
permission to reuse private expressive code. Inspection found no private implementation body or
distinctive model-specific sequence imported into the new files. The new contract, control flow,
data structures and observable semantics are materially different. This supports retaining the
**scoped MIT declaration for the new generic implementation**, rather than relabeling the frozen
private components. It is not proof that no copyright claim could ever be made.

The existing public preservation writer remains design-informed, explicitly owner-authorized MIT
source with a disclosed earlier source review; that relationship is not rewritten as clean-room
independence. All private runtime/TreeAIBox code and pretrained assets remain excluded. No pretrained
model or research dataset is needed by the original analytic Conv3D fixture.

Review outcome: **PASS for the technical provenance/disposition review**, with the boundary above.
The following remain outside that PASS: independent legal clearance, ownership of excluded private
source, commercial rights to TreeAIBox-derived models, and any claim that changing languages or
using AI makes a derivative work independent. Per-file identities and licenses remain in the
source-assessment, dependency and public-provenance manifests.
