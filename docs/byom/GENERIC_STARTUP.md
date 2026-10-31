<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Tuomas Pietilä -->

# CPU descriptor and startup library

`native/include/synrex/generic_startup.hpp` and `native/generic_startup.cpp`
provide CPU metadata admission and an authority-side startup gate in
`synrex::generic`. They implement CPU admission for the companion generic control transport and
[session/Service library](GENERIC_RUNTIME.md). Legacy CLIs do not select this path.
The companion [binary-v3 control transport](GENERIC_TRANSPORT.md) exercises it
over real sockets. Existing package-v3 rejection at every legacy streaming endpoint
remains in force. Generic file/stream GPU interchangeability and trained quality
are **NOT_RUN**; this module does not complete E2.

`CpuAdmittedPackage(path)` privately constructs and owns a const production
`EngineArtifact`, through private shared const ownership; the artifact privately owns its production
`Package` and captures the exact model bytes. The constructor performs envelope, Contract, artifact hash, taxonomy,
size and member admission. It exposes no mutable Package or Contract reference,
raw-state mutator, public-Package constructor or boolean admission waiver. Copying
is disabled. Only this wrapper can create an authority descriptor through
`Descriptor::from_package`; a descriptor parsed solely from a peer cannot supply
an authority's expected identity.

The wrapper retains captured manifest, taxonomy, derived Contract metadata and
hash-verified model bytes. A separate mutable Package or subsequent source-file
change cannot replace this retained identity. Descriptor construction does not create a GPU session. The authority factory
retains and consumes that same artifact after metadata admission; no later
pathname load or caller-selected Package can replace it.

The descriptor carries exact [taxonomy UTF-8 bytes](TAXONOMY.md), including their
whitespace and ordered class meanings, plus the taxonomy digest, K, complete
ordered LAS projection and its digest of exactly K raw uint8 bytes. Repeated LAS
codes preserve distinct semantic IDs, including semantic ID 255 at K=256. Neither
class count, taxonomy ID nor LAS projection substitutes for taxonomy identity.
The captured manifest digest binds the same bytes admitted by Package.

Package envelope `synrex.srxmodel.v3`, descriptor schema
`synrex.session.descriptor.v3`, dense adapter `DENSE_VOXEL_SEGMENTATION_V1` and
proposed binary schema `synrex.binary.v3` (version 3) have separate fields.
`text_stream_schema` must be null; generic text streaming is unsupported. The
context identity separately binds this exact declaration:

```json
{"schema":"synrex.context.current_frame.v1","history":false,"origin":"per_window_minimum","emit":"current_once"}
```

The descriptor admits that declaration; the generic Session uses the retained
persistent dense backend with one current frame per run and no historical context.
Real GPU parity of this combined path remains unqualified. Callers supply a
32-character lowercase hex session ID and a nonzero uint64 map epoch. Every
control and admitted frame identity binds both and the entire encoded descriptor
digest. New model, taxonomy, context or epoch requires a new startup object.
The library does not generate fresh sessions, authenticate peers or establish
which authority identity a peer should trust.

Encoded descriptors are limited to 65,536 bytes, independently of the taxonomy
artifact's limit. Escaping and envelope overhead count. SAX validation rejects
duplicate decoded keys, more than eight object/array levels and malformed
UTF-8/JSON before DOM construction. Exact fields and versions, integer ranges,
full projections and context/text identities are checked. Controls have their own
1,024-byte limit and exact schema. `Limits::negotiate` validates both inputs and
selects the smaller point, byte and initialization-time limits independently.
The factory also rejects a frame point limit above the admitted Contract.
The companion control transport exchanges and enforces peer advertisements.
The companion codec validates complete bounded frame/output bodies. The transport
also provides typed socket frame/classification I/O with absolute operation
deadlines. The output wrappers add independently reconstructed terminal snapshots
and challenge acknowledgments; see [transport](GENERIC_TRANSPORT.md#terminal-output-admission).
CLI integration remains pending.

Startup proceeds in this order:

1. CPU package/config admission and construction of an expected descriptor.
2. Admission of descriptor fragments and a matching metadata acknowledgment.
3. Permission to begin GPU initialization, followed by explicit caller
   notifications that initialization began and completed.
4. A matching execution-ready acknowledgment, then admission of frame identities.

Peers can parse metadata and encode a metadata acknowledgment without waiting
for a classified frame or GPU initialization. Creating a control string does not
prove it was delivered. `gpu_permission()` checks state and deadlines; it never
creates GPU resources. `accept_frame_identity()` checks order and identity;
it neither decodes nor executes a frame.

Metadata admission has an absolute five-second deadline from construction.
Fragments and polls never reset it. Metadata acknowledgment starts a separate
absolute initialization deadline with the selected 1..120,000 ms timeout.
Initialization notifications and the execution-ready acknowledgment do not extend
that deadline. Equality expires. Once ready, startup deadlines cease; an actual
service still needs execution, timing and resource failure handling.

The default clock is `std::chrono::steady_clock`. Injected time supports exact
boundary tests, rejects reversal and checks deadline-addition overflow. Malformed,
mismatched, expired and out-of-order events permanently fail the gate, preserving
the first error. Recovery requires a new object/session/epoch. The gate has one
owner and does not claim thread safety.

The ordinary CMake/CTest target `synrex-generic-startup-tests` links only
`synrex_core` and defines a throwing `target_info` sentinel. Its model members are
small opaque CPU fixtures; they are not valid ONNX and are never executed. Native
controls cover descriptor identity and parser bounds, startup ordering and absolute
deadlines, and private admission provenance. Compile-time assertions reject raw
Package ingress, admission booleans, copying, reference conversions and public
retained Package/Contract fields. These tests establish CPU metadata behavior.
They do not establish transport delivery, frame/body/output identity propagation,
context execution, GPU initialization order in a real session, numerical parity,
quality or performance. Those integrations and their compatibility checks remain
required.
