<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Tuomas Pietilä -->

# Generic session, socket and map libraries

The generic API is separate from every legacy CLI, JSON stream and single-file
exporter. It accepts explicit package-v3 taxonomies, including non-FRACTAL seven
classes, repeated LAS projections and K=256/semantic ID255. Current evidence is CPU
contract execution with a test-only opaque backend plus native SDK syntax checks.
Typed socket I/O, descriptor-bound maps, strict snapshot admission and independently
reconstructed terminal output have CPU regressions. Actual generic engine
file/stream interchangeability, CLI selection, export bundles and trained quality
remain unqualified.

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

Both transports expose typed `send_frame`, `receive_frame`, `send_classification`
and `receive_classification` operations with a caller-supplied absolute
`Startup::Time` deadline. Readiness is checked before I/O or encoding. The fixed
header, expected message kind and whole negotiated envelope size are checked before
payload allocation; the codec then checks identity, count and all point/label fields.
Fragmentation does not reset the deadline, which is also checked after encoding or
decoding. These operations share the same single-owner connection as the Session;
they cannot run concurrently with that Session or its Service. A future concurrent
CLI must arrange separate owned connections or synchronized ownership explicitly.
The socket remains borrowed. Every error permanently fails and shuts down the
connection. Clean EOF has a distinct diagnostic from a truncated envelope and
throws; neither constitutes terminal success. The output wrappers below add
explicit terminal receipts and snapshot admission.

`generic::SemanticMap` is a CPU consumer of already classified v3 results. It does
not authorize inference. It validates complete labels, exact descriptor identity,
LAS projection and encoded result size before mutating the map. The source identity
and increasing frame sequence/time are fixed within one map. A changed taxonomy,
model or epoch needs a new consumer. Invalid input or failed snapshot encoding
permanently fails the consumer. There is no reset that can carry old cells into a
new model. The private shared map kernel preserves the legacy last accepted
observation, revision eviction and late timestamp policy; generic labels never pass
through legacy taxonomy validation.

Map deltas use `synrex.generic_map_delta.v1` and bind descriptor/session/epoch and
taxonomy digest. Snapshots use `synrex.generic_map_snapshot.v1`, retain exact
descriptor bytes (including the full ordered taxonomy), source identity and
`as_of_ns`. Semantic IDs remain distinct when multiple classes share a LAS code.
Snapshot consumers must validate and reconstruct this new schema; the legacy
snapshot decoder and single-file exporter do not accept it. Neither snapshots nor
deltas prove individual original-point coverage or provide calibrated confidence.

Snapshot capacity is 1–65,536 cells with a 16 MiB final encoded-byte guard. The
conservative bound is 1 MiB base plus 224 bytes per cell: exact descriptor bytes
are at most 64 KiB and JSON string escaping expands each byte by at most six;
four source strings each have at most 256 printable ASCII bytes. Remaining fixed
metadata fits within the base allowance. A cell has three signed coordinate keys
(the existing kernel limit is 1e15), a byte semantic ID, uint64 timestamp and age,
and null confidence fields. The unchanged cell encoding fits 224 bytes including
its array separator. The resulting bound is 15,728,640 bytes. Tests serialize and
independently reconstruct a full map with extreme keys, uint64 ages, escaped source
identities and an exact 64 KiB descriptor. `admit_snapshot` adds strict bounded
duplicate/depth/schema/type and exact identity,
key-order and timestamp/age validation without changing the serializer.
`OutputSender`/`OutputReceiver` independently reconstruct maps and use the
[terminal transport](GENERIC_TRANSPORT.md#terminal-output-admission) for bounded
snapshot comparison, transcript-bound completion and a fresh challenge ACK.
This is CPU output validation, not real model qualification.

CTest targets `generic_binary_v3_codec`, `generic_control_transport`,
`generic_binary_v3_socket_io`, `generic_taxonomy_map`, `generic_snapshot_admission`,
`generic_output_completion` and `generic_session_service`
cover the CPU boundaries. The test-only backend is linked only into fixture test
executables. Production backend linkage remains in
`native/generic_backend.hpp`, outside the public include tree. The GPU backend
deserializes retained engine bytes; the CPU-only backend reports NOT_RUN.
