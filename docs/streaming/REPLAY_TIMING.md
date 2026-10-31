<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Tuomas Pietilä -->
# Manifest replay timing

The manifest CLI reports fixed-memory latency histograms **per source** and in
`trial_timings` for the whole attempted trial. Trial histograms merge the actual
sample bins with the same population, weighting and process-local boundary;
they never average per-source quantiles. Failed sources contribute their measured
partial populations, while unattempted sources contribute no samples. All durations use the measuring
process's `steady_clock`; serialized source GPS and synthetic replay timestamps
are never subtracted from a host clock.

Each histogram retains 3,776 uint64 counters and exact count, sum and maximum,
without a sample array. Nearest-rank p50/p95/p99 are intervals containing the
sample quantile. Values below 128 ns are exact; wider intervals have width at
most their lower endpoint divided by 64. Use the upper endpoint for conservative
comparisons. Empty populations have null quantiles/maximum. Counter or sum
overflow fails the trial before mutating that histogram, including when merging
source populations into a trial histogram. Histogram storage is fixed per metric;
the source receipt list still grows with the admitted source count.

The sender measures decoded windows (`Reader::next`, excluding open/hash and
EOF verification), frame assembly plus exact source validation, dispatch lag,
and serialization/socket writes. A write proves input emission only. Queue
wait, storage bounds, discards and maximum age are separately reported in
`decode_queue`.

`synthetic_accumulation` contains one value per successfully emitted point:
scheduled end-exclusive frame release minus that point's synthetic timestamp.
The last point waits one declared period. EOF partial windows use their actual
size. This reports the accumulation policy, without claiming physical scan
order or acquisition timing. At 40k observations/s, a full 16,384-point window
requires 409.6 ms of accumulation before processing.

The receiver's `receiver_service` starts after complete frame read/validation
and ends after coverage checks, inference, mapping, serialization and output
socket writes. Each point in that frame receives the same measured service
duration, so quantiles are point weighted. It excludes frame accumulation,
ingress wait, and subscriber validation. `frame_ingress` separately includes
socket wait/read and codec validation. GPU stage times may overlap and are not
summed into this wall duration.

The subscriber separately measures classification receipt plus independent map
validation, and source attribute checks against its independently decoded
window. Source decode is excluded from those two intervals. Its terminal
acknowledgment and source/coverage receipts establish success separately.

Partial measurements and verified counters remain visible on failure; no failed
trial becomes a passing throughput result. Execution duration, anchored producer
duration, completed frame counts and sender decode/dispatch timing survive faults.
`actual_sender_rate_pps` divides emitted input points by the same reported anchored
producer duration, including a failed prefix; it proves neither classified output
success nor capacity. A full bounded decode queue applies backpressure and can drain
successfully; cancellation and decoder faults discard queued windows, join the
producer and retain a failed trial. Linux process peak RSS is the whole
process lifetime `ru_maxrss` (KiB converted to bytes), including admission and
startup. It is distinct from device memory. Missing power/capture provenance
remains null. `instrumentation_overhead` stays
`NOT_MEASURED_FOR_ACTUAL_PIPELINE`: the standalone helper CPU fixture reports
only its own measurement overhead. Finite replay capacity and timing target
qualification still require the admitted real workload and GPU evidence.
