<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Tuomas Pietilä -->
# Optional generic LIVE tensor diagnostics

This is an unqualified diagnostic implementation until actual source-bound GPU
execution and independent numerical review. CPU fixtures execute no neural model.
No context policy, training or transport schema changes are introduced. Tracing is
off by default; manifest replay does not expose it.

Only single-file `synrex-generic-live receive` accepts the complete option group:

```bash
--tensor-trace-root new-directory --tensor-trace-bytes 67108864 \
--trace-source-commit "$SOURCE_COMMIT"
```

The declared commit is caller metadata, not verified compiled Git provenance.
The actual `/proc/self/exe` SHA256, admitted engine artifacts, target, descriptor,
taxonomy and context-policy identities are retained. The owned controller's frozen
source/build/binary receipts establish provenance. Preserve those independently.

All options and a new root in an existing canonical parent are admitted before
backend initialization/readiness. Roots/parents using aliases or lexical `.`/`..` components are rejected for
these writable diagnostics. The admitted absolute root is captured once and used
for every write, even if a caller later changes its working directory. The exact CLI frame count is also the trace limit,
with at most 64 windows. The total regular-file budget must exceed 512 KiB and be
at most 256 MiB across the entire session. It includes admission, backend tensor
and manifest files and window lineage. A conservative 512 KiB admission reserve
and per-window metadata reserve count even when actual metadata is smaller.
The admitted contract must fit one unavoidable tensor-window minimum before any
backend is constructed. LIVE also requires the minimum for its exact declared
frame count before opening listeners; the public API window limit remains a cap.
Unknown actual geometry can still exhaust the reserved budget at runtime.

For each authorized frame, diagnostic CPU geometry computes a conservative tensor
plus metadata reservation before backend inference or window writes. This extra
planning does not affect the inputs or change the selected geometry mode. The
persistent backend receives a new window destination and explicit remaining byte
limit. Unsupported backends reject this budgeted method; it never falls back to
unbounded legacy tracing. Tensor and manifest writes are checked before writing.
Budget exhaustion or diagnostic failure poisons the session and both coordinated
authority links, preventing successful output completion. Partial directories are
retained as failed-attempt evidence and never overwritten or silently deleted.

`admission.json` binds execution/model/policy metadata. Each `window-N` contains
the existing dense tensor manifest, raw input/logit files, and `lineage.json`
binding descriptor/context, original source stream and source IDs, sequence,
frame times, exact encoded-input digest and actual backend-manifest digest.
The lineage is not a proof of source extraction or ground-truth identity; the
single-file CLI's existing independent source/coverage checks still apply.

`startup_trace_admission_seconds` measures trace CPU setup and admission I/O before
GPU construction. `diagnostic_service_seconds` covers per-window trace planning,
backend execution and all diagnostic metadata/hash/directory I/O before returning
classification. Its local steady-clock begin/prevalidation timestamps are explicit;
postprocessing has a separate subspan. These boundaries precede the existing
map/serialization/socket send and independent completion acknowledgment. No time
is subtracted from native wall/service measurements. CUDA event timings retain
their component scopes. No cross-host, physical capture-age, throughput or trained
quality claim follows from these diagnostics.

The legacy `InferenceSession::run(points, trace_path)` behavior remains available
for existing offline callers. The new budgeted method is separate. Native trained
Q0/Q1 comparison, matched preprocessing/logits, cutoff reference execution and
future-removed/appended/altered executable prefix probes remain required. See
[quality evaluation](QUALITY_EVALUATION.md) and
[generic single-file replay](GENERIC_SINGLE_FILE_V3.md).

Each block manifest also names `membership_points` (`N.points.u32.bin`) and
`membership_voxels` (`N.voxels.u32.bin`), with `membership_count` and SHA256 of
each file. Both files contain explicitly serialized little-endian uint32 values
in the actual backend host-vector traversal order, sorted by voxel then local
point index. Entry i in one file pairs with entry i in the other. Point indices
are zero-based in the exact input window, not global source IDs. Voxel indices
use `(iz * ny + iy) * nx + ix` (x fastest), including noncubic dimensions.
The manifest retains actual block execution order and the contract overlap
policy (`first_wins` or `last_wins`); repeated points across overlapping blocks
are preserved. The reservation adds exactly eight bytes per actual CPU-plan
membership and per-block metadata; it never charges `max_memberships` as though
that cap were observed. One-shot file traces use the same mapping serialization.

Each window reserves 4096 bytes each for lineage, timing and output timing before
inference. `timing.json` durably records receiver-local `service_started_steady_ns`,
`trace_payload_published_steady_ns` (backend files and lineage complete), and
`classification_validated_steady_ns`. Its own write follows these samples and is
included in Session service wall time. `receiver_frame_available_steady_ns` is the
LIVE caller's immediate observation after `receive_frame` has finished decoding,
authorizing and transcript accounting. It is null for direct Session API calls
unless the caller supplies a local observation; source-declared times remain
separate. Neither receiver timestamp denotes physical capture.

LIVE samples `output_ready_steady_ns` immediately after Session returns, after
its diagnostic writes and validation. Following successful synchronous output send,
`output-timing.json` retains this sample plus send-start/send-completion samples.
Its publication follows those samples and is counted within the pre-reserved
trace budget. Send completion does not mean independent terminal acknowledgment.
Receiver receipt fields `input_completion_observed_steady_ns` and
`output_completion_observed_steady_ns` follow the actual terminal acknowledgments;
receipt publication follows the observations. Earlier per-window records survive
later failure; missing records never imply successful completion. No file claims
to timestamp completion of its own publication, and no diagnostic time is subtracted.

The producer receipt retains its local steady-clock replay anchor, boot identity
and first/last scheduled frame availability. These describe declared synthetic
replay pacing, not physical point acquisition or receiver availability. Cross-process
clock comparison requires independently verified same host/boot and clock identity.
