# Retained point-model session seam

The point API in `synrex/point_session.hpp` is a separate native type boundary
beside dense `Contract`, `Package` and `InferenceSession`. Dense parsing,
mathematics, factories and v2/v3 transport remain unchanged. A point package is
never represented by a dummy ONNX file, dense tensor or TensorRT engine.

This implementation provides bounded CPU envelope admission, retained native
source/current/inverse authority and an abstract retained backend interface.
It does **not** implement a learned backend, local worker process launcher,
external wire4 socket codec, GPU execution or application qualification. The
synthetic tests cannot establish any of those results. A production adapter
must supply the application validator, concrete backend and durable output
committer, and qualify their exact combined sources and bytes.

## Artifact and application boundary

`PointArtifact` takes already bounded member bytes, exact manifest bytes and an
externally trusted manifest SHA256. It captures immutable copies, checks strict
JSON/duplicate keys, exact v4 root/roster fields, safe canonical relative names,
member sizes/hashes, singleton roles, aggregate bounds and declared document
schema identities (JSON Schema members use `$id`; typed documents use `schema`). It refuses absent or extra payload members and requires a
trusted local application validator. No schema, import or executable from the
package is invoked. The constructor does not read an archive or extract files;
the ingress that constructs the member map must independently enforce bounded
archive/path admission before reading or allocating those bytes.

The trusted validator checks the complete model-specific schema, trained
parameters, source/profile/precision/dependency identities, schema bundle,
required source fields and qualified capability evidence. Its returned
`PointContract` is a compact retained native view, not a replacement for that
validation. It binds the selected file, buffered-same-batch or causal capability
and exact policy digest independently; file qualification cannot imply causal
qualification. Package-authored claims cannot supply this callback. Manifest and
member hashes bind those exact retained bytes; an application archive digest is
a separate identity. REFERENCE and HYBRID are explicit. There is no whole-model
NATIVE claim or automatic backend fallback.

Ordinary sessions require the validator's qualified capability. An explicit
`qualification_only` mode enables separately accounted qualification work and
marks the descriptor and completion accordingly. `synthetic_fixture` is a
separate explicit CPU fixture mode; fixture readiness cannot initialize a real
production session. Neither exception is a production qualification result.

## Native source and output authority

The exact source roster is retained in `SessionOptions.sources` before either
metadata ACK. It names each source content hash, original header hash and local
row count, inside per-source/aggregate/source-count bounds. Native source
capture and verification are the application's authority for those identities;
a remote hash alone is insufficient for a preserving writer. Duplicate source
hashes, changed headers/counts and undeclared sources reject.

Two distinct peers must ACK the retained descriptor digest before the backend
factory can run. The factory's retained initializer must return an identity-bound
receipt with the exact package manifest, point contract, worker-source and
environment digests, explicit fixture status and bounded details. Only then can
two distinct peers ACK its receipt digest. The separate backend `activate()`
receives that gate only after both ACKs. Neither metadata ACK nor peer READY
JSON implies initialized inference. The concrete launcher must verify actual
worker PID/start identity, peer credentials, nonce, configuration and durable
initialization receipt; the abstract API cannot establish those process facts.

Each source open increases the epoch, beginning at1. Request IDs begin at1 and
never reset. A new source cannot discard incomplete output. Native retains raw
source identity, current local rows, representative rows and inverse indices.
The batch uses tightly bounded finite FP32 coord[N,3]/feat[N,4], nonnegative
int32 grid_coord[N,3], implicit int64 offset=N and uint64 representative rows.
The payload digest uses exact packed little endian array bytes in that order.
It is independent of C++ object padding or host byte order. Zero-row sources
complete without any model forward; an empty model batch itself rejects.

When a representative source row is among current rows, its inverse must select
that same representative index. History-only representatives may remain outside
current rows. Contradictory in-range maps reject before backend execution.

The native authority record additionally binds descriptor, epoch, request,
source/header/row count, exact ordered current/inverse maps, representative count,
input payload, logical batch and policy cutoff. This internal
`synrex.point.native-authority.v1` record is not an external worker-protocol
record; a concrete adapter binds the retained native record into its separately
versioned IPC authority. Likewise `synrex.point.native-session.v1` is an internal
native descriptor requiring wire4, not an implementation of external wire4
framing. Dense wire-v3 peers cannot consume these types.

The backend returns only representative semantics/validity plus the retained
request/authority identity. It never returns source mapping authority. Every
current inverse must select validity1; semantic0 with validity1 remains valid.
Invalid context representatives may remain only if no current inverse uses them.
All results are checked before any current output is emitted. Returned native
classifications retain epoch, request, logical-batch and native-authority digests
alongside source/current rows so the application can bind publication to admission. Wrong epoch,
request, digest, counts, class, validity, partial response or backend exception
fails the session. Accepted/emitted row fences survive source reset. Immediate
backend destruction on failure invokes the concrete backend's RAII retirement;
the concrete backend must own cancellation, deadlines, process-group drainage
and receipts and must never abandon a live worker.

Completion requires the exact declared roster and complete once-only coverage
for every source. A native-owned committer receives full original-order semantic
arrays; it verifies/stages/publishes durable output according to its own writer
contract and returns its byte commitment. Completion binds per-source coverage
and semantic digests, request/epoch counts and that durable commitment. Backend
completion and two distinct peer ACKs of the exact completion digest are required
before terminal success. A failed session never claims successful completion.
The native committer must not be constructed from an untrusted peer callback.

Source I/O in `synrex/point_io.hpp` is separately bounded to the documented
standard compressed LAS1.4 layouts. Model preprocessing, complete learned
initialization/inference, identical-batch transport, causal windows, full worker
lifecycle and deployed dependency identity remain application integration gates.

## CPU verification

Build with CUDA disabled. `point_session_authority` uses only synthetic package
bytes and a clearly marked fixture backend, with a throwing device-probe sentinel.
It checks retained admission, source roster, dual admissions, initialization,
source-order inverse expansion, valid semantic0, stale/partial/invalid results,
negative lattice coordinates, reset/replay and exact completion. The unchanged
dense tests remain required alongside these additive point tests.
