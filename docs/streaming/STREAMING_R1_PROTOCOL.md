# Streaming R1 loopback wire protocol

<!-- Copyright (c) 2026 Tuomas Pietilä. SPDX-License-Identifier: CC-BY-4.0. -->

R1 extends the frozen R0 API with independently authored native transport. This is software
TCP replay, not a physical sensor adapter. The frozen R4 package and neural arithmetic remain
authoritative. One ordered sensor/stream/coordinate identity owns one service and map. Different
sensors use separate sessions; missing poses never cause implicit map fusion.

All integers are little endian, doubles IEEE754 binary64. There is no native structure padding.
The first unqualified prototype used envelope v1; v2 adds per-event model binding and rejects v1.
A 56-byte envelope precedes every payload: magic `SRXWIRE1` (8 bytes), version u16 (=2),
kind u16 (1 input, 2 classified points plus map deltas, 3 snapshot, 4 end), payload bytes u32,
reserved u64 (=0), SHA256 of payload (32 bytes). Limit is 16 MiB, checked before allocation.
Checksums detect corruption and are not authentication. Listener addresses are always IPv4
loopback. Applications must not forward this unauthenticated endpoint to a public interface.

Input payload order:

- Five strings: sensor, stream, coordinate frame, CRS, taxonomy. Each is u16 length followed
  by 1–128 printable ASCII bytes.
- Four u8 values: coordinate domain (1 sensor-local, 2 vehicle, 3 local world/map, 4 projected),
  timestamp reference (=1 UNIX ns), units (=1 metres), completion (=1 complete frame).
- Five u64 values: sequence, frame ID, inclusive measurement start/end, optional same-host
  monotonic replay deadline. The replay clock is separate from measurement time.
- Point count u32, 1–65,536. Frame span at most one second.
- Exactly that many 48-byte source records: original ID u64, measurement UNIX ns u64,
  XYZ 3*f64, intensity u16, return number u8, return count u8, attribute flags u8
  (bit0 intensity present, bit1 both returns present), three zero reserved bytes.
  Absent values must be zero. Returns, if present, satisfy 1 <= number <= count <=15.

Complete-frame transport uses chunk index 0/count 1; fragmented TCP reads do not create new
semantic chunks. Unknown versions, flags, enum values, trailing bytes, nonfinite/out-of-budget
coordinates, duplicate within-frame source IDs and invalid timestamps reject the whole message.
Sequence must increment exactly by one, frame ID must increase and frame time cannot move
backward. Sequence state survives up to 16 clean reconnects. EOF inside an envelope or payload
is a visible FAIL and never produces a partial classification. A complete end marker must
reconcile exact frame/point totals. No unbounded reorder or reconnect history is retained.

Local/vehicle/world coordinates require an explicit `LOCAL:*` CRS; projected coordinates require
an explicit `EPSG:*` declaration. Declaring a CRS performs no reprojection or georeferencing.
`wire::Transform` v1 is an explicitly supplied rigid rotation/translation valid over the entire
measurement interval; unknown versions, invalid rigid matrices or missing timed poses reject.
It preserves IDs, attributes and measurement timestamps. Sensor-local maps remain explicitly
local unless the producer applies an explicit transform. There is no SLAM, pose interpolation,
motion compensation, TTL expiration or implicit GPS-time conversion.

Classified output starts with the complete input payload, then first-envelope-read receiver time, publication
and model execution count (3*u64), model manifest SHA256 (u16 length + 64 lowercase hex ASCII), one semantic ID u8 plus LAS projection u8 per source point,
delta count u32, then that many 64-byte deltas: signed voxel XYZ 3*i64 (two's complement),
semantic ID u8, eviction-present u8, six reserved zeros, measurement time u64, evicted XYZ
3*i64 (zero if absent). Map deltas count observations, not unique new voxels. Model semantic
channels 0–6 remain distinct from the frozen LAS projection `[1,2,200,6,9,17,201]`.
Confidence is unknown; snapshots retain confidence null and source-time age exactly as in R0.
Snapshot/end payloads are bounded UTF-8 JSON, each protected by the same envelope.

Service input/output and delivery queues each hold two frames. Producer overload fails visibly.
Network output has a separate writer, nonblocking bounded enqueue and a 100 ms absolute send
deadline. A stalled subscriber is disconnected and delivery failures are counted separately
from fresh neural classification coverage. There is no silent success with missing deliveries.
Kernel socket buffers are explicitly bounded at 256 KiB requested (Linux may double this).
Input reads have absolute ten-second deadlines, including partial records. Cancellation drains
or explicitly accounts queued work; active GPU kernels finish before resource teardown.

Measurement timestamps, replay deadlines, first-envelope-read receiver time and publication
are distinct. Replay latency is not physical sensor age. End-to-end qualification must also
measure arrival at the independent subscriber; internal publication alone is insufficient.
Canonical SHA256 output excludes host replay/arrival/publication clocks but includes all source
fields, semantic projections, model execution counts and ordered map deltas. The independent
subscriber verifies every source record and delta and the final full snapshot.
