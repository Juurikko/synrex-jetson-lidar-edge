<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Tuomas Pietilä -->
# Legacy snapshot admission

The live receiver and subscriber admit 1–65,536 map cells. Both validate the
same `wire::LegacyLiveLimits` before constructing a GPU session, replay source,
exporter or map. Independent `SemanticMap` callers retain their existing capacity
contract. Wire version 2, its 16,777,216-byte payload budget and supported snapshot
fields remain unchanged.

For the current compact JSON serializer, a conservative snapshot bound is
`8,192 + 224 * cells` bytes. At 65,536 cells this is 14,688,256 bytes.

The fixed allowance covers all field names, punctuation, capacity, resolution,
schema and mode. Each of the five identity strings has at most 128 printable
ASCII bytes on the live wire. A quote or backslash expands to two JSON bytes;
five maximally escaped strings need at most 1,280 content bytes plus delimiters.
The two legacy taxonomy names and required CRS prefix only reduce that bound.
Empty identity is serialized as `null`.

Each cell has exactly six fields. Three signed coordinate keys of magnitude at
most 10^15 need at most 17 bytes each. The live 1-metre map and ±10^9-metre wire
coordinate limit are narrower. Semantic IDs fit within three decimal bytes
(legacy IDs actually need one). Timestamp and nonnegative age each fit in 20
decimal bytes. The two confidence fields are literal `null`. Including field
names, array/object delimiters and the following comma, the compact cell encoding
fits within 224 bytes. No identity, coordinate or timestamp is truncated to fit.

`json_bytes` checks the actual encoded byte count, and `send` independently
enforces the envelope budget. A serialization or required-snapshot send failure
prevents successful receiver qualification. Receiver delivery counters describe
socket sends. Only a subscriber receipt that reconstructs every cell and checks
the complete snapshot digest establishes independent validation. Missing,
corrupt or duplicate snapshots cannot qualify a subscriber run; an empty stream
cannot claim successful neural execution.

The CPU `legacy_snapshot_budget` CTest recreates the former 262,144-cell defect,
checks maximal escaped identities and map keys, uint64 timestamp/age cases, empty
and full snapshots, actual fragmented socket transfer, digest equality,
independent reconstruction, and exact encoded-byte boundaries. CLI tests verify
admission before package/source/export/session access and reject false terminal
success. These are CPU contract tests; inference quality and GPU throughput
require their own qualification receipts.
