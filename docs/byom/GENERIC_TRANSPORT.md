<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Tuomas Pietilä -->

# Opt-in binary-v3 transport

`generic_transport.hpp` implements a CPU control exchange over a caller-owned,
connected POSIX `SOCK_STREAM` socket. Callers explicitly select
`WireProtocol::binary_v3`. This library is separate from legacy wire-v2: no legacy
CLI, Service, endpoint or codec calls it, and it never falls back to wire-v2.
The [generic runtime library](GENERIC_RUNTIME.md) now supplies the authority session
factory, frame/output codecs, typed socket I/O and separate Service. CLI selection
and terminal/snapshot wire protocols remain pending. Real generic GPU
interchangeability remains **NOT_RUN**.

Every message has a fixed 12-byte header. Control payloads use UTF-8 JSON;
frame/classification payloads use the binary codec described in the runtime guide:

| Byte | Meaning |
| --- | --- |
| 0–3 | ASCII `SRX3` |
| 4 | Binary version 3 |
| 5 | Kind: hello=1, descriptor=2, metadata ACK=3, execution-ready=4, ready ACK=5, frame=16, classification=17 |
| 6–7 | Reserved, must both be zero |
| 8–11 | Unsigned payload length, big endian |

The decoder reads into a fixed header buffer, validates the exact expected kind,
magic, version and reserved bytes, then checks the nonzero payload length before
allocating its payload. Hello is limited to 512 bytes, the complete encoded
descriptor to 65,536 bytes, and each control to 1,024 bytes. These are per-message
bounds; envelope and hello/ACK overhead are additional to the descriptor bound.
Unknown capabilities, extra fields, invalid integer limits and duplicate decoded
hello keys reject. The [descriptor parser](GENERIC_STARTUP.md) separately bounds
depth, duplicate keys, taxonomy identity and encoded size.

The authority and peer each require their own CPU-admitted package pathname,
expected 32-character lowercase hexadecimal session ID and nonzero map epoch.
Peer admission never derives trust solely from metadata received on the socket.
The library checks exact descriptor bytes reconstructed from the independently
admitted expected package, session, epoch and negotiated limits. Manifest digest,
exact taxonomy UTF-8 bytes and digest, full ordered projection, class count,
adapter, context, package schema and binary schema all participate. Same K,
projection or reused taxonomy ID cannot substitute for semantic identity. This
binding supplies integrity and an application expectation, not authentication.

The exchange is:

1. The peer sends `synrex.session.hello.v3` with supported `max_frame_points`,
   `max_frame_bytes` and `startup_timeout_ms`.
2. The authority selects the minimum of its own and the peer's limits in each
   dimension and sends the package-bound descriptor with those chosen values.
3. The peer rejects chosen values above its supported limits, reconstructs and
   validates the full expected descriptor, and sends the metadata ACK immediately.
4. The authority validates the ACK through its privately owned `Startup` gate.
   Only then may `initialization_permission()` return true. Authority
   `create_session(options)` consumes this same gate and retained engine. It
   validates engine kind/capacity before device construction, initializes the
   backend and verifies the loaded contract, manifest, artifacts, target and
   capacity. Public caller-asserted initialization completion is unavailable.
5. After initialization completion the authority generates a fresh random 128-bit
   readiness challenge and sends execution-ready. The peer verifies its full
   descriptor, session and epoch binding and echoes that challenge in the ready ACK.
   A ready ACK pre-sent from knowledge of the descriptor cannot satisfy this gate.
6. Both sides permit `authorize_frame()` only after their readiness exchange.

The hello JSON has exactly `schema` and `limits`; `schema` is
`synrex.session.hello.v3`, and `limits` has exactly the three fields above.
Other controls have `schema: synrex.session.control.v3`, the corresponding
`kind` string (`metadata_ack`, `execution_ready`, `execution_ready_ack`),
`descriptor_sha256`, `session_id` and `map_epoch`. Execution-ready and its ACK also
carry `readiness_challenge`, exactly 32 lowercase hexadecimal characters. The
transport verifies the exact descriptor-generated serialization plus challenge;
it rejects alternate serialization of those two controls. Metadata ACK validation
uses the startup library's strict bounded JSON parser.

`authorize_frame(hash, session, epoch, points, bytes)` checks the admitted identity
and both negotiated sizes before a caller allocates a payload. It does not inspect
payloads, read frames, decode headers, validate labels or prove execution. A caller
must invoke it before its own allocation and supply the actual frame sizes.

Each connection starts a real absolute five-second metadata deadline at
construction, including all hello, descriptor and ACK work. Nonblocking send/recv,
poll retries, EINTR and fragment progress never reset it. The admitted metadata ACK
starts a separate absolute deadline bounded by the negotiated startup timeout
(1–120,000 milliseconds); initialization notifications and readiness controls never
extend it. Equality expires. The startup gate independently checks that deadline.
Typed frame/classification send and receive methods require a caller-supplied
absolute `Startup::Time` operation deadline. The library checks readiness before
socket I/O or encoding, validates the whole-envelope negotiated byte limit before
incoming payload allocation, and checks the deadline after codec work. A partial
transfer, poll retry or new fragment never resets that deadline.

Malformed or out-of-order messages, duplicates at a receive gate, identity failures,
oversize lengths, EOF/truncation and timeouts permanently poison the connection.
Failure shuts down the borrowed socket in both directions so blocked peers wake;
the owner still closes it. Recovery requires a new connection and startup object.
Methods are single-owner and are not thread safe. The connection retains the
admitted artifact independently of the package wrapper. The borrowed socket must
outlive all produced Sessions/Services and must not be reused during that lifetime.
Do not call any transport method concurrently with a produced Session/Service.
The socket is neither reconfigured nor closed here.
Typed receive methods reject control envelopes at the frame/classification gate;
there is no unsolicited background receive loop. Clean EOF throws a diagnostic
distinct from truncation and permanently fails the connection. There is no terminal
success envelope: EOF must never qualify a completed stream or snapshot.

`synrex-generic-transport-tests` / CTest `generic_control_transport` exercises real
TCP loopback, a socketpair proxy that splits every write, K=2/4/non-FRACTAL 7/256
metadata and semantic ID 255, actual limit negotiation, changed meanings/order with
the same K/projection/ID, malformed/oversized/truncated envelopes, duplicate and
out-of-order controls, frame authorization gates, startup stalls and an actual
five-second fragmented metadata stall. A test-only fake backend consumes opaque engine bytes; a throwing `target_info`
sentinel proves no GPU query occurs. The production factory is exercised with
CPU fixtures, but that does not establish valid engine deserialization or neural
inference. The test establishes the CPU control/factory contracts only.

`generic_binary_v3_socket_io` additionally exercises real socketpairs for typed
data transfer in both directions, raw fragmentation and coalescing, forced
backpressure, fixed-header-only oversize rejection, malformed identity/projection,
startup ordering, EOF/truncation and absolute deadlines. It preserves exact f64,
uint64 source fields and K256/ID255 while resetting remote host timestamps. These
checks use the opaque test backend and establish neither GPU execution nor
concurrent transport/Service integration.
