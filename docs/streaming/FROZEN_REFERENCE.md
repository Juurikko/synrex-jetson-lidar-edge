<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Tuomas Pietilä -->
# Frozen trained CPU reference

`tools/runtime_frozen_reference.py` prepares and executes the frozen R2 U-Net
FP32 ONNX reference on one admitted validation source. It reads coordinates and
uint16 intensity, with no truth access, training, tuning, GPU acquisition or native
execution. Model and dense-spec identities are pinned to the existing R2 freeze.
Weights, architecture, data and dependencies retain their own notices; this tool's
MIT grant covers its original source only.

Prepare needs NumPy, laspy and laz-rs. Inference additionally needs ONNX Runtime's
CPU provider in the independently identified environment. Importing the tool or
running its unit tests requires no ONNX Runtime installation or learned inference.
Execution plumbing tests use a mock session with synthetic zero logits.

```bash
python tools/runtime_frozen_reference.py prepare \
  --source "$VALIDATION_SOURCE" --source-sha256 "$SOURCE_SHA256" \
  --spec examples/unet/dense_voxel.json --onnx "$FROZEN_FP32_ONNX" \
  --out new-prepared-directory
# Independently retain printed prepared_sha256; register an owned CPU job first.
python tools/runtime_frozen_reference.py infer \
  --prepared new-prepared-directory --prepared-sha256 "$PREPARED_SHA256" \
  --registered-cpu-job-id "$CPU_JOB_ID" --out new-execution-directory
```

The checkout defaults to the tool's repository; `--source-checkout` selects an
explicit frozen checkout. All outputs require new directories and exclusive files.
The registration string records lineage; it does not itself verify registration
or acquire resources. The existing campaign runner remains responsible for job
admission, source/environment identity, durable execution and resource accounting.

Preparation hashes and decodes one captured source byte sequence. Q0 uses the
whole source minimum; Q1 independently uses contiguous 16,384-point windows and
an EOF partial. Geometry uses binary64 scaled LAS coordinates and floor indexing;
feature reduction uses binary64 followed by float32 affine normalization, matching
the existing supervised dense reference without reading its labels. Decimal minima
and observation digests from `runtime_quality` describe window lineage separately
from executable floating point geometry. FP32 CPU reference and FP16 native
execution require a separately frozen mixed-precision numerical comparison.

The preparation binds model, adapter, checkout modules, observations and feature
blocks. Its source map must contain exactly the canonical paths for the executing
adapter, the selected checkout's `dense_training.py`, `runtime_quality.py` and
`runtime_comparison.py`; every role requires its actual hash. A moved adapter
rejects an old preparation even if the old file remains. Loaded quality/comparison
module paths are also checked. The actual checkout Git commit and actual dense
spec bytes must match the retained commit and pinned spec SHA256, before inference
and again before a successful receipt.

Before inference, the adapter verifies the retained manifest hash, actual model
bytes, each window's lineage, bounded NPZ/NPY metadata and exact original-point
coverage. It loads the captured model bytes with
only `CPUExecutionProvider`, two intra-op and one inter-op threads. Raw FP32 logits,
original-point prediction JSONL and a receipt are retained. Input changes fail
without a successful receipt. Partial outputs after a runtime failure remain
unqualified and require a new attempt directory.

Admitted sources use LAS 1.4 PDRF6/7/8 with standard 30/36/38-byte records;
extra dimensions are rejected before point-array decoding. Header-bounded decode
does not independently prove physical EOF beyond declared source records.

Bounds are 50,000 decoded full-source points, 1 MiB compressed source, 64 aggregate
nonempty blocks, 4,096 enumerated spatial blocks per window, 4 MiB per model/block,
16 MiB observations, 512 KiB manifest and 64 MiB aggregate prepared artifacts.
Retained inference output is bounded by these limits to under 80 MiB. These are
artifact/work bounds; vendor allocations, service time and system capacity are
separate measurements. The runner supplies the execution deadline.

Availability is one actual batch timestamp after complete decode, with unknown
capture time. Preparation and inference must share the Linux boot identity. This
schedule provides no sensor arrival, physical point-age or sustained capacity
claim; retained preparation-to-execution delay is not processing service time.

A source of at most 16,384 points gives identical Q0/Q1 input. Its equality tests
EOF execution mechanics, and cannot measure streaming-context loss. The unit
suite checks features, window slice isolation, bounds and identity rejection with
CPU fixtures; its future-minimum fixture is not an executable causality probe.
Actual offline-cutoff inference, native future-removed/appended/altered prefix
probes, matched native feature/logit traces, truth/taxonomy scoring and
acquisition-level uncertainty remain required execution. This adapter emits no
`runtime_quality` PASS and does not replace a preregistered quality protocol or
final integrated GPU qualification. See [quality evaluation](QUALITY_EVALUATION.md).
