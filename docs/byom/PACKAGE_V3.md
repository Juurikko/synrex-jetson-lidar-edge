<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Tuomas Pietilä -->

# Explicit taxonomy packages

Package-v3 adds a validated [taxonomy artifact](TAXONOMY.md) to the existing dense
adapter envelope. The [generic runtime library](GENERIC_RUNTIME.md) adds explicit control/session/
Service and frame/output contracts. These CPU-qualified APIs do not establish
real GPU execution qualification.

The existing live receiver, sender, subscriber (including source-less validation),
text-stream replay, sustained benchmark and asynchronous streaming exporter are
legacy-only. They reject every v3 envelope on CPU with `package v3 is file-only;
legacy streaming requires package v2`, before Contract/artifact admission or GPU
construction, source decode, exporter work or socket readiness. The retained
legacy session package enforces this policy itself, avoiding a check/reopen gap.
Matching the legacy class count, taxonomy ID or LAS vector grants no exception.
The path-only `create_session` API remains legacy-only. The separate generic
`AuthorityTransport::create_session` API requires the admitted v3 descriptor and
actual backend initialization before readiness. Metadata-only stream fixture
and validate modes remain model-free. Generic file/stream GPU interchangeability
and trained quality are **NOT_RUN**; these guards do not complete E2.
`DENSE_VOXEL_SEGMENTATION_V1`, tensor layouts, features, geometry and numerical
projection remain unchanged.

The root manifest has exactly `schema`, `contract`, `artifacts`, and `target`.
`schema` is `synrex.srxmodel.v3`. Artifacts contain exactly the existing license,
exactly one ONNX or engine artifact, and:

```json
"taxonomy": {"path": "taxonomy.json", "sha256": "<exact artifact SHA-256>"}
```

The validated taxonomy must have the same K and complete ordered LAS projection
as `contract.las_classes`; the Contract already checks its output channel count.
LAS codes may repeat without collapsing semantic IDs. Taxonomy ID or K alone
cannot identify class meanings. Schema versions and package basenames are exact;
extra/missing artifacts and unexpected directory entries are rejected.

The manifest is read through one nonblocking, non-symlink final file descriptor,
checked regular, bounded to 1 MiB before parsing and throughout reading, and
captured in memory. A SAX pass rejects duplicate keys, invalid syntax/UTF-8 and
more than sixteen object/array levels; a normal DOM parser then reads those same
bytes. `Package::manifest_bytes` and `manifest_sha256` describe those exact bytes,
including whitespace and final newlines. The taxonomy independently enforces its
65,536-byte limit through its own safe file reader. The generic 2 GiB model member
budget does not apply to taxonomy. The new path uses the captured manifest digest
for file semantic identity; legacy output behavior is retained.

Existing invocations without `--taxonomy` still create package-v2 with exactly
two artifacts and the same serialized manifest. Existing generic v2 file packages
remain valid. A taxonomy file or third artifact is not silently accepted into v2.

Explicitly create a new package-v3:

```bash
synrex-native pack --spec dense.json --onnx model.onnx --license NOTICE \
  --taxonomy taxonomy.json --out new.srxmodel
```

Engine packing still identifies the current target and therefore requires the
native GPU backend. Taxonomy/Contract admission happens before that target query.
Copy an existing admitted package to a new taxonomy envelope without a GPU:

```bash
synrex-native repack --model old.srxmodel --taxonomy taxonomy.json --out new.srxmodel
```

Repack accepts valid v2 or v3 inputs. It retains the admitted model bytes, license,
Contract and target exactly, without querying the current GPU or retagging an
engine. It can therefore attach changed class meanings only by creating a new
package identity. Existing files and packages are never replaced, and a new
package cannot be written inside its source package, including through a parent
symlink alias.

New v3 creation/repack stages privately beside the destination. A shared reader
opens model/license members with `O_NOFOLLOW | O_NONBLOCK`, checks regular type
and initial size in 1..2 GiB, then reads at most that initial byte count in bounded
chunks. It hashes the exact bytes it copies, without an earlier unbounded hash or
a read-until-EOF loop that could follow a growing source. Before returning the
digest it compares the opened descriptor and final pathname with the initial
device, inode, mode, link count, size, modification time and change time; atime is
excluded because reading can update it. Truncation, observed mutation or pathname
replacement fails. Package admission uses this same reader, including v2 packages
used as repack inputs; stable valid v2 package hashes and serialization are
unchanged. New engine packing queries its target only after the CPU snapshots
have passed.

New packing records the copied-byte hashes. Repacking additionally requires each
copied hash to match its source manifest's admitted hash. The code writes the
exact already validated taxonomy bytes and validates the complete staged Package
before `RENAME_NOREPLACE` publication.
Repack also checks that its source manifest still equals the admitted bytes before
publishing. Failed validation removes the private staging directory. These checks
bind the published snapshot; they do not promise that another process cannot
change an external source after copying. Package/model input immutability remains
required during later inference and session admission.

CPU tests cover v2 preservation, explicit four-class/non-FRACTAL seven-class,
K=2/K=256/class 255 envelopes, repeated LAS projections, exact taxonomy bytes,
repack source preservation, CPU-only engine target retention, changed meanings,
invalid schema/digest/K/projection, final symlinks/FIFOs/oversize metadata,
bounded source admission for both model/license and ONNX/engine pack choices,
replacement refusal and output containment. Source size tests use sparse metadata,
not a generated 2 GiB model. Each new CLI test invocation has a 15-second timeout.
Model-free package fixtures contain
explicit placeholder bytes and are never deserialized. They establish metadata
and publication contracts, not ONNX validity, neural inference, quality, throughput
or generic streaming behavior. Those gates require separately recorded execution.

The original raw v2 `pack` implementation and general-purpose `sha256(path)` helper
remain unchanged; this package-only change does not claim to harden every legacy
caller against concurrent source mutation. The new v3/repack path does not use
that unbounded helper for source model/license admission. The snapshot checks are
not an atomic filesystem snapshot or a promise of immutability after a successful
copy; later inference/session admission must separately bind its loaded inputs.
