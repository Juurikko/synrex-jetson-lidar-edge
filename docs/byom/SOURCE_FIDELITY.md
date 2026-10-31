<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Tuomas Pietilä -->
# Opt-in LAS source attributes transport

Manifest replay accepts `--wire binary-source-v1` on the sender, receiver and
subscriber. Library callers select `WireProtocol::binary_source_v1` explicitly.
The existing binary-v3 and single-file paths retain their wire layout and return
validation. No decoder negotiates a fallback. This transport carries source
provenance in addition to the existing online point features; it does not add
source labels, historical context or new model features.

Every envelope, including startup, snapshots and terminal records, uses the
12-byte [generic transport header](GENERIC_TRANSPORT.md) with `SRXS` magic and
version byte 1. Message kinds, reserved zero bytes, big-endian payload length,
absolute deadlines, fixed-header checks before allocation and permanent failure
on error follow the generic transport contract. The hello schema is
`synrex.session.hello.source.v1`. Its exact two fields remain `schema` and `limits`.

The existing session descriptor envelope is retained, with these exact identities:

| Field | Required value |
| --- | --- |
| `binary_wire_schema` | `synrex.binary.source.v1` |
| `binary_wire_version` | `1` |
| `source_attributes_schema` | `synrex.las14.source_attributes.v1` |

The last field is required only for this transport and forbidden in binary-v3.
Unknown descriptor fields, attribute schema versions and local protocol enum values
reject. Each peer independently reconstructs the complete expected descriptor
from its admitted package, selected wire protocol, session, epoch and negotiated
limits. Both metadata acknowledgments precede backend creation. Readiness challenges,
transcript digests, terminal acknowledgments and snapshots bind the exact descriptor
bytes/digest, so a peer cannot mix protocols after admission.

Frame and classification bodies begin with the unchanged binary-v3 body through
its count and all 48-byte point records. The source variant then appends:

| Size | Field |
| --- | --- |
| 375 bytes | Exact standard LAS 1.4 public header, original little-endian bytes |
| 8 bytes | Declared source record count, unsigned big endian |
| 8 bytes | Source's global wire-ID prefix, unsigned big endian |
| 28 bytes per point, in order | Original zero-based record index u64, raw GPS binary64 bits u64, XYZ raw i32 bit patterns; each integer big endian |

Classification semantic/LAS labels and timings follow this extension. There is
no attribute extension flag or variable attribute map. The complete encoding,
including the 12-byte envelope, counts against the negotiated frame byte limit.
The source variant adds exactly `391 + 28 * points` bytes. Manifest CLI defaults
to 2 MiB for this variant, preserving the 1 MiB binary-v3 default; explicit smaller
budgets reject if worst-case classified output would not fit. Count, fixed header,
whole-envelope budget and minimum remaining payload are checked before allocation.

Supported source layouts are compressed LAS 1.4 PDRF 6/7/8 with their standard
30/36/38-byte records. Unknown extra-record attributes reject. The defined fields
preserve raw signed XYZ integers, exact scale/offset bits within the public header,
uint16 intensity, return number/count, raw GPS bits and source header/row identity.
RGB/NIR and other record fields are outside this attribute schema and make no
transport-preservation claim. VLRs, compressed bytes and complete file preservation
remain represented by the source SHA256, not copied into frames.

The `stream_id` is the exact lowercase source SHA256. Every point ID must equal
`wire_begin + original_record_index`; rows are contiguous and within source count.
LAS header count, record bounds and finite positive scales/finite offsets are
validated. Scaled XYZ must agree bit-for-bit with the ordinary source integer,
scale and offset calculation. Intensity and paired return fields are required.
Paired return number/count `0/0` is representable in this schema. Mixed zero,
number greater than count, and counts above 15 reject. Binary-v3 still rejects
`0/0`. GPS is opaque 64-bit data: signed zero, infinity and NaN payload bits remain
unchanged and are never interpreted as synthetic or acquisition timestamps.

The receiver checks source hash/row/count/metadata coverage and requires the public
header to remain identical across frames. The subscriber independently reopens and
decodes the source, compares every exact public-header byte, raw integer, GPS bit
pattern, row identity and supported point attribute, and independently reconstructs
the map before acknowledging the descriptor-bound terminal snapshot. Its coverage receipt reports
`raw_source_attribute_points`; identity-only receiver coverage does not claim an
independent source decode. The source header is provenance only: bounding boxes,
source labels and later rows never affect current-frame online inference.

This contract does not establish physical capture time, physical EOF beyond header
records, sustainable capacity or neural quality. CPU fixtures qualify source,
protocol and completion mechanics; real GPU qualification remains a separate gate.
