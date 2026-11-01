# Explicit point wire4 and common caller

`common::Session` is a closed choice between the existing dense
`generic::Session` and the separate `point::PointSession`. Its input and output
are corresponding two-type variants. An input for the wrong backend rejects
before either implementation runs. `from_dense(input, output, options, trace)`
uses the existing dual-link dense factory without changing dense geometry or
startup. The dense constructor also accepts a session already produced by that
factory. `from_point(artifact, session_id, factory, options)` captures the trusted
point artifact and application worker factory; construction never probes a GPU
or creates the worker. Only the native point authority's dual metadata ACKs can
permit initialization. Applications call `point_authority()` for the existing
source, readiness and completion lifecycle. There is no dense/point fallback.

The public dense CLI retains its existing package parser and refuses a v4 point
package before device initialization. It does not execute package-provided code.
An application supporting point inference supplies its own trusted validator,
worker factory and durable output committer through `from_point`. The public
`point-source` CLI remains an I/O utility, not an inference application.

## Wire selection and binding

`point::wire4` is a separate codec and bounded socket primitive. Construct
`Binding(native_point_session, retained_point_artifact, max_packet_bytes)` from
the local, independently trusted native authority. It captures the exact
descriptor, source roster, contract and worker hash. The default whole-packet
cap is 8MiB; the allowed configured range is 1KiB through 64MiB. This cap is
additional to all retained model/source/representative/input/output bounds.
There is no untrusted-descriptor constructor. A peer must independently hold
the expected artifact and session identity; a received descriptor alone cannot
provide trust or confer readiness.

Every packet starts with the 12-byte header below. SRX3, SRXS, wrong versions,
unknown expected kinds, reserved bits, excessive advertised lengths, truncated
messages and trailing bytes reject. Existing dense codecs do not parse SRX4.

| Field | Bytes | Encoding |
|---|---:|---|
| Magic | 4 | ASCII `SRX4` |
| Version | 1 | 4 |
| Kind | 1 | control 1–7, batch 16, classification 17 |
| Reserved | 2 | zero |
| Payload length | 4 | unsigned big endian, excludes header |

The payload begins with the descriptor's 64-byte lower-case SHA256 text and
36-byte session UUID. Subsequent integers are unsigned little endian, with
signed lattice values restricted to the nonnegative int32 range. Float arrays
are exact IEEE754 little-endian float32 values. No C++ structure padding, JSON
per-point arrays, pickle, Torch object, GPU handle or dense tensor is serialized.

## Typed payloads

A batch payload, after the common identity, is:

1. Source SHA256 text[64], header SHA256 text[64], source rows u64.
2. Cutoff nanoseconds u64, logical batch SHA256 text[64].
3. Current count u64, representative count u64.
4. Current source rows u64[count], inverse u32[count].
5. Representative source rows u64[N], coord f32[N,3], lattice i32[N,3], feat f32[N,4].

Counts, exact remaining byte length and retained limits are checked before
allocating arrays. The packet encodes the exact prepared batch; it performs no
windowing, coordinate conversion, normalization or representative selection.
Decoding validates finite floats, unique/ranged source rows and inverse bounds.
Native `PointSession::run` independently validates self-maps, source state,
replay fences and the retained model/input/output contract before execution.

A classification payload, after the common identity, is:

1. Source SHA256 text[64], header SHA256 text[64], source rows u64.
2. Epoch u64, request u64, logical batch SHA256 text[64], native authority SHA256 text[64].
3. Count u64, original source rows u64[count].
4. Semantic IDs u8[count], validity u8[count], LAS projection u8[count].

Classifications require unique in-range rows, valid semantic IDs, validity 1
and the exact retained semantic-to-LAS projection. Semantic 0 remains valid.
Consumers must independently reconcile each output with its expected input,
logical batch, ordered row roster, epoch/request and native authority digest;
codec success alone is not a source-coverage or preservation receipt.

## Controls and execution order

Control payloads carry common identity followed by at most 64KiB UTF8 JSON.
Duplicate keys, excessive depth/node count, mismatched commitments and unexpected
body kinds reject. The control kinds are:

| Kind | Name | Body |
|---:|---|---|
| 1 | descriptor | Exact locally expected descriptor object |
| 2 | metadata_ack | Exact descriptor SHA256 string |
| 3 | initialization | Actual native initialize() wrapper/receipt |
| 4 | ready_ack | Initialization SHA256 string |
| 5 | source | Object with sha256, header_sha256, unsigned rows |
| 6 | completion | Actual native complete() wrapper/receipt |
| 7 | completion_ack | Completion SHA256 string |

The application sends the descriptor on distinct input/output links, forwards
the two received metadata ACKs to `PointSession`, then initializes the retained
backend. It sends that actual initialization wrapper to both peers and forwards
their ACKs to the native authority. Input and common-session inference start
only after the authority is active. Source opening remains native-owned. After
all input and independently reconciled output, the native durable committer
publishes complete source-order results and creates the actual completion
wrapper. Both peer ACKs must reach the authority before terminal success.

`Connection` borrows a connected SOCK_STREAM descriptor. It neither closes nor
replaces the descriptor. It uses nonblocking send/receive with a single absolute
steady-clock deadline, including codec work. Fragmentation never renews that
deadline. Header, kind and advertised size precede payload allocation; descriptor/session
identity precedes array allocation. Any
I/O, codec or deadline failure poisons and shuts down the endpoint. EOF always
throws; it is not successful completion. Each connection has one caller;
independent endpoints may run concurrently. The socket layer does not fabricate
backend readiness, durable publication or completion ACKs. Its caller must
preserve the native authority lifecycle above.

CPU fixtures exercise actual socketpairs, dual metadata/readiness/completion,
wrong-backend dispatch, exact arrays/source-order reconstruction, valid zero
semantics, malformed headers/lengths, every truncated batch prefix, source and
row refusal, deadline expiry, EOF and poisoned reuse. They make no learned,
GPU, causal-quality, throughput, package-relocation or whole-model-native claim.
