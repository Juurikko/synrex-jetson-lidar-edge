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
remains null. `process_user_cpu_seconds` and `process_system_cpu_seconds` use
`getrusage(RUSAGE_SELF)` over the same whole process lifetime, excluding child
processes. They include admission; they are not execution-only CPU durations.

Manifest peers accept `--timing-instrumentation enabled|disabled` (default:
`enabled`). Disabled mode skips the optional replay histogram observations and
the clock reads used only by those observations. It reports
`timing_availability=UNAVAILABLE_INSTRUMENTATION_DISABLED`, null source `timings`,
empty `trial_timings`, and null maximum sender scheduling lag. Pacing, deadlines,
source/coverage validation, output transcripts and acknowledgment fences, queue
resource diagnostics, backend timing, whole-process wall/CPU/RSS counters and
per-source histogram arrays remain allocated in both modes. Trial histogram
aggregation and histogram receipt serialization run only when enabled. Thus this
switch isolates the incremental replay measurement/aggregation/reporting cost;
it cannot measure the cost of all runtime instrumentation or per-source
histogram storage. Failure receipts retain that
availability and all verified partial counters. Enabled runs with no recorded
samples report `UNAVAILABLE_NO_SAMPLES`; a zero-sample histogram has null tails.
`replay_configuration` records window/rate/map/queue/I/O limits in both modes so
paired analysis can reject configuration drift.

`tools/replay_overhead.py PAIRS.json COMPARISON.json` compares already recorded
receipts and creates a new output file. It launches no jobs. The owner of the
resource controller must record actual GPU runs. A pair plan has this shape
(paths may be absolute or relative to the pair plan):

```json
{
  "schema": "synrex.replay.overhead_pairs.v1",
  "scope": "opaque_cpu_fixture",
  "executable": {"path": "build/synrex-generic-live-cpu-fixture", "sha256": "..."},
  "source_identity": {"commit": "...", "changed_file_sha256": {}},
  "pairs": [
    {
      "order": ["disabled", "enabled"],
      "disabled": {"send": "d0/send.json", "receive": "d0/receive.json", "subscribe": "d0/subscribe.json"},
      "enabled": {"send": "e0/send.json", "receive": "e0/receive.json", "subscribe": "e0/subscribe.json"}
    }
  ]
}
```

Record 3–100 pairs, alternating disabled/enabled then enabled/disabled order.
Use the same frozen executable, package, manifest, source files, session/epoch,
wire mode, window, rate, queue and map configuration. Keep each process's raw
receipt, exit identity, command, logs and source/environment identity in the
runner evidence. The comparator verifies the current executable hash; source
identity and execution order are runner declarations, not independently attested
by receipt analysis. Reused receipt paths, failed/incomplete trials, missing
resources, inconsistent counters/completion fences/histograms, identity drift
and input/output transcript differences reject the
comparison. JSON rejects duplicate keys and nonfinite constants; receipt reads
are bounded to 16 MiB each and the pair plan to 1 MiB. Larger receipts need a
separate bounded analysis policy. All peers must agree on delivered input and classified output/map
hashes. Random completion/acknowledgment challenges are excluded from identity;
their original values remain in hashed raw receipts.

The result retains each receipt path/hash, semantic identity hash, per-pair
execution seconds, whole-process CPU seconds, peak RSS, enabled histograms and
signed enabled-minus-disabled differences. Disabled tails are unavailable; never
interpret them as zero or claim a tail improvement against them. Median wall
changes summarize finite pairs without a pass threshold or significance claim.
Startup, offered-rate pacing and host noise can dominate; negative differences
remain visible. Per-role execution intervals overlap, so do not sum them into an
end-to-end wall time. CPU fixture measurements establish switch/transcript
mechanics and finite host observations only. `unqualified_native_pipeline` is the
other accepted scope; it still makes no GPU qualification claim.

`instrumentation_overhead` stays `NOT_MEASURED_FOR_ACTUAL_PIPELINE` in individual
receipts: neither a single enabled run nor the standalone histogram helper
establishes overhead. Finite replay capacity and timing target qualification
still require the admitted real workload, repeated paired GPU execution under
the controller, and an independent source-bound evidence review. The comparator
always reports actual GPU instrumentation qualification as
`NOT_ESTABLISHED_BY_THIS_TOOL`.
