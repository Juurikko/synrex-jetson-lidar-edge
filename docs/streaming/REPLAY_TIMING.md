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


## Optional backend phase profiles

Manifest `receive` accepts `--profile-backend enabled|disabled`, default `disabled`.
It is separate from `--timing-instrumentation`: this switch only retains values
already measured by the actual persistent backend. It introduces no GPU event,
context, model, geometry-policy or transport changes. Disabled/default mode adds
no backend-profile fields or accumulators. Other peer modes reject this option.

Enabled source receipts include `backend_profile`; the receiver trial receipt
includes `trial_backend_profile` (`synrex.backend_profile.v1`). Seven named
phase summaries retain sample count, missing-window count, sum, minimum and
maximum seconds. A sample is one successful `Session::run` return, before map,
serialization and output send. Unequal windows remain unweighted window samples;
three integer summaries separately retain completed inference blocks, freshly
classified points and occupied voxels. Trial summaries merge actual sums/counts
and extrema, never source averages. Returned inference results remain counted if
output later fails. A thrown inference contributes no successful-return sample;
prior source/trial populations survive failures. Samples do not prove output
completion; existing coverage and acknowledgment receipts still establish that.

`cpu_geometry_seconds` and `window_seconds` use the receiver's steady clock.
The H2D/zeroing, feature preparation, TensorRT, scatter and D2H durations are
backend CUDA-event measurements with the scopes named in the receipt. Geometry
subphase fields that are uninstrumented under reference/fallback geometry are
not exposed as measured zeros. GPU phases and host wall scopes may overlap;
never sum phase durations into end-to-end wall or subtract them from service.
`window_seconds` includes CPU geometry through synchronized label reconstruction
and backend result creation, and excludes startup, receiver coverage, map and
output. Existing receiver-service and process resource measurements retain their
boundaries.

Workload metadata comes from the actual session identity and contract: manifest
hash, selected/effective geometry, dimensions/voxel size/overlap and resolution
policy, feature/class counts, voxels per block and input/output dtype. Occupied
voxels count block-local occupied cells summed over completed blocks; repeated
points or cells in overlapping blocks remain in this count. A denominator for
block-local occupancy is completed blocks times voxels per block, not distinct
scene volume. No feature/class meaning is inferred from channel count.

Opaque CPU fixtures and unrecognized backends report `UNSUPPORTED_BACKEND` with
null phase/counter/workload values, even if synthetic timing-like fields exist.
For a supported backend, absent fields retain zero samples and null statistics;
explicit `missing_windows` counts distinguish missing values from measured zeros.
Malformed, negative/nonfinite values, noninteger counters, inconsistent point or
voxel counts, workload drift and arithmetic overflow fail profiling rather than
fabricating a measurement. Every completed block must contain at least one
occupied cell, and occupied cells cannot exceed returned points times completed
blocks. Block/occupied-cell/returned-point counts cannot exceed the admitted
membership cap. These checks use only available counter values; missing values
remain missing. Nominal zero overlap alone is not used to infer a strict point
partition: existing geometry permits duplicate rounded block origins when block
extents fall below coordinate precision. Profiling retains those actual counts;
it does not repair or qualify that pre-existing geometry behavior. Updates are transactional, retaining the previous
valid prefix. Profiles support source-bound measurement; they do not attest that
a GPU job ran or establish quality, capacity, optimization or instrumentation
overhead. External frozen source/binary and actual job receipts remain required.

The seven phase and three counter accumulators have fixed-size storage and no
per-frame sample list. Only one source and one trial accumulator are active;
workload metadata is bounded to 2048 compact JSON bytes and each compact serialized
profile to 16 KiB. `storage_bytes` reports the inline C++ object only, excluding
that metadata's JSON heap and serialized receipts; it is not total profile memory.
Outer receipt pretty-printing adds whitespace to the compact profile bound.
Receipt lists still grow with the already admitted manifest
source count (at most 4096); worst-case added compact profile JSON is bounded by
`16384 * (attempted_sources + 1)` bytes. No new admission limit replaces existing
source, frame, codec or output budgets. Actual profiling overhead remains
unmeasured until independently compared under the resource controller.

The paired `replay_overhead.py` comparator normalizes absent `backend_profiling`
to `disabled` for older receipts, and includes the setting in configuration
identity. It rejects drift between a profiling-enabled and profiling-disabled
receipt, so that difference cannot be presented as replay histogram overhead.
