<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Tuomas Pietilä -->

# Generic session, service and binary payload library

The generic API is separate from every legacy CLI, JSON stream, map and single-file
exporter. It accepts explicit package-v3 taxonomies, including non-FRACTAL seven
classes, repeated LAS projections and K=256/semantic ID255. Current evidence is CPU
contract execution with a test-only opaque backend plus native SDK syntax checks.
Actual generic engine file/stream interchangeability, stream socket I/O, CLI
selection, generic map/export propagation and trained quality remain unqualified.

`CpuAdmittedPackage(path)` captures model bytes once. Construct both peers with
their expected package, session ID and epoch, then perform `metadata_admission()`
on each connected endpoint. The authority's `create_session(options)` is the sole
public generic initialization operation. It uses that connection's privately
retained engine and negotiated capacity, verifies actual backend identity, completes
the fresh-challenge execution-ready exchange and returns a privately constructed
`generic::Session`. The peer calls `acknowledge_execution_ready()` after metadata
admission. There is no public arbitrary backend callback or completion assertion.
Calling the factory before admission, twice, or with an ONNX artifact fails closed.
Initialization or ACK failure destroys the partial session and poisons the gate.

The backend reuses the existing dense persistent implementation, geometry and
reductions. CPU checks reject capacities above the Contract or 1,000,000 points
before CUDA-owning members construct. Synchronous vendor initialization cannot be
preempted by this library; the absolute startup deadline is checked before and
immediately after construction, and a late result is destroyed rather than made
ready. Timeline arrays are unsupported in this wire output and are rejected before
initialization. No numerical threshold or legacy package policy is relaxed.

`Session::run(Frame)` validates full descriptor/session/epoch identity and encoded
input bounds, authorizes the retained gate, converts features and runs the retained
backend. It verifies full output coverage, semantic range, exact LAS projection
and encoded output bounds before returning `Classification`. A source identity
change or nonincreasing sequence/frame ID, or a regressing frame time, permanently
fails the session. This bounded ordering check does not prove distinct original
records across frames; E3 manifest accounting remains required.

`generic::Service` takes only a factory-created Session. One producer, one inference
worker and one writer use separate queues capped at eight frames each. It
overwrites incoming acceptance/submission clocks at the local Service boundary.
Reported service latency excludes accumulation and network delay. The service
drains before a terminal receipt, propagates worker/sink/validation failures and
counts cancelled/rejected work. An active service reports RUNNING; an empty service
does not produce a successful classification receipt. Source IDs are opaque uint64
values and are never forwarded to the legacy single-file exporter as row indexes.

The connection retains immutable artifact ownership after the admission wrapper is
destroyed. A produced Session retains the gate after the AuthorityTransport wrapper
is destroyed. The caller-owned socket must outlive all Sessions/Services and must
not be closed or reused during that lifetime. Control methods and Session/Service
are single-owner: do not call the transport concurrently with inference. Failure
shuts down that borrowed socket; the owner closes it.

`generic_codec.hpp` provides explicit `encode_frame`/`decode_frame` and
`encode_classification`/`decode_classification`. Codecs validate payloads but confer
no readiness authority. A descriptor parsed solely from a peer is insufficient to
construct an authority Session. Frame/output kinds 16/17 share the 12-byte `SRX3`
version3 header with the control transport. Reserved bits must be zero; the u32
big-endian payload length must match exactly. Whole-envelope bytes count against
the negotiated frame byte limit. Each point is a 48-byte record containing uint64
source ID/time, three IEEE754 binary64 coordinates and explicit optional intensity/
return fields. Encoding preserves double bits, including signed zero; coordinates
must be finite within ±1e9, frame span at most one second, source IDs unique within
the frame, and optional return attributes valid and paired. Strings are nonempty
printable ASCII up to 256 bytes. Host acceptance/submission clocks are omitted;
decode resets them. Output adds semantic/LAS byte pairs and at most 16KiB of flat,
finite, nonnegative numeric timing JSON with at most 64 fields. Remote timings
remain untrusted measurements and never establish local capture-to-output age.

CTest targets `generic_binary_v3_codec`, `generic_control_transport` and
`generic_session_service` cover the CPU boundaries. The test-only backend is linked
only into those test executables. Production backend linkage remains in
`native/generic_backend.hpp`, outside the public include tree. The GPU backend
deserializes retained engine bytes; the CPU-only backend reports NOT_RUN.
