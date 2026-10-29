# Bounded CPU semantic-map eviction index

<!-- Copyright (c) 2026 Tuomas Pietilä. SPDX-License-Identifier: CC-BY-4.0. -->

The accepted source `958c418d6799196605155f0223418d6f3b4f5c4d` is frozen at local tag
`streaming-r0-geometry-frozen-958c418`. The [freeze receipt](../../archive/evidence/eviction_baseline_freeze.json)
hashes its archive, accepted geometry/churn reports, all earlier PASS and FAIL receipts and
qualified source identities. Those files are never overwritten by this continuation.

## Verified policy

The original map stores ordered integer XYZ voxel keys. Each accepted observation receives an
incrementing unsigned 64-bit revision. Capacity eviction selects the smallest numeric revision;
if revisions tie, ordered-key traversal selects the lexicographically first key. Before revision
wrap, accepted revisions are unique, so this is recency of accepted observations, irrespective
of timestamp order across distinct cells. It is not timestamp-based eviction or source-ID order.

An older timestamp for an existing cell is rejected without refreshing recency or producing a
delta. An equal timestamp is accepted: a changed semantic ID emits a delta, and an unchanged
semantic ID emits none; both refresh recency. Repeated source IDs have no independent map key.
Map identity comprises sensor, stream, coordinate frame, CRS and taxonomy, exactly as before.
Cell key arithmetic, semantic IDs, timestamps and confidence-null behavior remain unchanged.

There is **no map TTL expiration policy** in the frozen source. Snapshot age is `as_of - time`,
negative ages reject, and age never expires cells. The stream assembler's packet timeout is a
separate policy and remains unchanged. Expiration instrumentation explicitly reports disabled
rather than implementing a new removal rule.

## Implementation

`native/semantic_map.cpp` retains the original revision scan under `MapPolicy::reference` and
adds `MapPolicy::indexed`, selected by `--map-policy reference|indexed` in both replay and the
native sustained benchmark. Indexed is the default; semantic behavior is identical. No model,
TensorRT, CUDA, geometry or source contract is modified.

Each ordered-map cell contains intrusive previous/next links plus its voxel key. A pair of map
roots tracks oldest/newest. Acceptance unlinks and appends an existing cell, or appends a newly
inserted cell; a late rejection leaves links untouched. Eviction reads the oldest key, unlinks
the victim, then removes that map node. Node addresses remain stable in `std::map` until erased.
There are no per-refresh allocations, hash buckets, stale heap entries or deferred tombstones in
the index. The ordered tree continues to define lookup and snapshot order. The benchmark oracle
uses a separately allocated hash table and list, while the reference uses numeric revision scans.

The implementation deliberately preserves unsigned revision overflow. Once the next increment
would wrap, the index ceases to define the numeric-minimum policy: that instance permanently
falls back to the original scan. Tests force this transition. The fallback does not relabel or
renumber revisions, does not change tie-breaking, and needs no unbounded index storage. Normal
sustained tests require no overflow fallback. At millions of updates/s, reaching 2^64 accepted
observations would require many millennia, but correctness is still defined.

Map instances are confined to the existing output-consumer thread. Copy and move are disabled to
prevent accidental cross-object intrusive links. Destruction frees only the resident map nodes;
links own no allocations. `reset()` clears cells, roots, identity, revisions, counters and timings
while retaining constructor configuration, so a new stream identity is accepted afterward.
Cancellation stops service delivery using existing queue behavior; already accepted observations
remain a coherent map. There is no background index worker to cancel or drain.

## Complexity and memory

For N active cells and K observations: reference victim selection costs O(N) per replacement;
indexed victim selection and link maintenance cost O(1). Lookup, tree insertion and key erase
remain O(log N). Updating K observations therefore costs O(K log N), plus output-delta storage.
No full resident-map iteration occurs on the ordinary indexed update path. The overflow fallback
is explicitly exceptional and reported. Snapshot generation remains O(N) time/output memory;
JSON encoding and its final SHA256 are outside the measured streaming loop. Detailed index
validation is O(N log N) and runs in CPU tests, never per production observation.

On this 64-bit build, cell payload grows from 24 to 64 bytes: two pointers plus a 24-byte key add
40 bytes per resident cell (2.5MiB at 65,536). Tree-node allocator overhead and ordered keys are
additional; profile receipts report payload sizes rather than pretending to measure allocator
internals. There are exactly N live index entries, bounded by configured capacity, regardless
of refresh count. Reference mode uses the same enlarged cell layout, so the separately retained
frozen executable is the authoritative before-performance measurement. Native RSS and GPU memory
remain separately gated; an index entry count is not a substitute for measured memory.

## Instrumentation and independent checks

Optional detailed timers separately measure coordinate/key checks, lookup, insertion, existing
updates, victim selection, removal, index maintenance, delta construction and full snapshot
construction. Classification validation has its own timer. Regions are disjoint; untimed loop
and timing-call overhead remain residual. Snapshot JSON string encoding is outside snapshot
construction timing. Detailed per-observation clocks are disabled in accepted sustained trials;
coarse map and end-to-end timers remain active. Instrumented CPU service probes are not neural
throughput or sustained-rate claims.

Differential tests compare all deltas, eviction keys/order, classes, timestamps, snapshots, ages,
late rejection, identity mismatch, reset and capacity bounds against the scan. Capacities include
every value 1–64 plus 257, 1,024, 4,096, 16,384 and 65,536. The independent hash oracle deliberately
forces collisions at capacity 257; production lookup is ordered and has no hash-collision case.
Tests cover negative and adjacent floating-point coordinate boundaries, large offsets, coincident
points, equal timestamps, class changes, delayed input and all seven semantic channels.
[Explicit policy expectations](../../archive/evidence/eviction_temporal_cpu.json) separately assert
that a new cell with an older global timestamp is accepted, equal-time observations refresh
recency, late existing observations do not, and eviction follows the exact expected key order.
These tests cover resolutions 0.001, 0.125, 1, 32.5 and 1,000 metres, full-width ages and no TTL.
They pass both release and ASan/UBSan builds.
A separate 2,097,152-accepted-observation refresh phase advances timestamps and checks RSS,
zero late rejection and exactly 65,536 index entries. The initial fixed-timestamp memory test
mostly exercised late rejection; its receipt is preserved and the strengthened CPU/sanitizer
runs are recorded in [the correction receipt](../../archive/evidence/eviction_test_correction.json).

[CPU parity](../../archive/evidence/eviction_cpu_accepted_refresh.json),
[reference stage profile](../../archive/evidence/eviction_cpu_profile_reference.json) and
[indexed stage profile](../../archive/evidence/eviction_cpu_profile_indexed.json) are model-free receipts.
Frozen/optimized neural comparisons and sustained hardware conclusions are recorded separately
in the qualification report; no model-free number is advertised as freshly classified points/s.

## Provenance and boundaries

This is independently authored MIT-scoped native source and test code, based only on the existing
public MIT source and this task's accepted contract. Standard C++17 containers add no vendored
implementation. Original upstream notices and dependency licenses remain intact. The trained
models and licensed IGN/FRACTAL data remain separate, unchanged artifacts outside Git. Documentation
uses the established CC-BY-4.0 scope. No private engineering code, weights, LiDAR binaries, secrets
or endpoint addresses are added to history. Track B remains local; R4, AGX and Thor are untouched.
