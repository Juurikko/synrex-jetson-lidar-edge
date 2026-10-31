<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Tuomas Pietilä -->

# Generic finite single-file replay

`synrex-generic-live` supplies opt-in binary-v3 `receive`, `send` and `subscribe`
commands. Legacy executables retain their selection. CPU process/socket/export
regressions use an explicitly opaque test backend, including K=256/channel-255
coverage; they provide no neural or performance result. R3 additionally executed
real CUDA/TensorRT full-file, identical-window and independently reconstructed
stream mechanics for explicit K4 and non-FRACTAL K7 packages at `23b8162`.

Trained Q0/Q1 execution at `62a6f81` passed identical-window numerical gates.
Whole-source versus streaming policy quality remains inconclusive on one validation
source with five truth classes absent. Finite manifest replay at `92e2067` preserved
exact semantics but failed its sender scheduling gate and demonstrated
`NO_MATERIAL_OPTIMIZATION`. The [scoped R3 evidence summary](../../ARCHITECTURE.md#quality-comparison-and-evidence-boundaries)
records the measured limits and campaign handoff authority; these source-specific
results do not qualify another candidate, sensor timing or an Orin deployment.

Build with the usual [native dependencies](../byom/QUICKSTART.md). The production
executable uses the configured backend; a CPU-only build rejects inference.
`synrex-generic-live-cpu-fixture` exists only with `BUILD_TESTING`, is never installed
and is never eligible for model qualification. It runs the same CLI entry point.

Every command requires strict key/value options: `--wire binary-v3`, an explicit
package-v3 `--model`, a new `--receipt`, and a caller-selected 32-character lowercase
hexadecimal `--session`. `--epoch` defaults to 1 and must be nonzero. Use a new
session/epoch for each new model or trial. All peers independently admit the
expected package, exact taxonomy and session; there is no fallback.

The receiver opens separate loopback input/output ports and waits for both peers.
Both metadata acknowledgments and identical negotiated descriptors precede one
retained inference backend. Both fresh readiness ACKs complete before Session
return. One caller owns both transports and synchronous receive/inference/output.
Failure poisons both authority links. Socket owners outlive transports and Session.

This CLI supports one compressed LAS 1.4 PDRF6/7/8 source file, at most 1,000,000
rows and 256 MiB compressed. It captures bytes into a sealed anonymous file and
hashes/decodes that same immutable capture. Original f64 XYZ and uint16 intensity
are sent once in source row order. IDs are local zero-based rows. Source labels
are excluded from inference features. Return attributes and original acquisition
times are not supplied by this replay decoder; they remain preserved in optional
LAZ output. The wire declares `source-file` / `UNSPECIFIED:source-file`, with no
coordinate transform or inferred CRS.

The wire stream ID is the exact compressed source SHA256. Receiver and subscriber
require that expected digest through `--source-sha256`. Subscriber `--source` also
compares every XYZ/intensity record against its own sealed capture. Without it,
the receipt records that source record comparison was not performed. The receiver
verifies declared identity and exact row/window/count coverage; it does not read
the original file.

For an admitted production engine and allocated GPU execution, the command shapes
are below. Freeze `POINTS`, `RECORDS`, `FRAMES` and `SHA` from the admitted source
before launch. `FRAMES` is exactly `ceil(RECORDS / POINTS)`; the final partial window
is retained. Launch the receiver first, then subscriber and producer through the
applicable job controller.

```bash
synrex-generic-live receive --wire binary-v3 --model "$MODEL" \
  --session "$SESSION" --receipt receive.json --points "$POINTS" \
  --port 39183 --output-port 39184 --frames "$FRAMES" --records "$RECORDS" \
  --source-sha256 "$SHA" --map-cells 65536
synrex-generic-live subscribe --wire binary-v3 --model "$MODEL" \
  --session "$SESSION" --receipt subscribe.json --points "$POINTS" \
  --port 39184 --frames "$FRAMES" --records "$RECORDS" --source-sha256 "$SHA" \
  --source "$SOURCE" --export-bundle new-export-bundle
synrex-generic-live send --wire binary-v3 --model "$MODEL" \
  --session "$SESSION" --receipt send.json --points "$POINTS" \
  --port 39183 --source "$SOURCE" --rate 1000000
```

Defaults are 16,384 points/window, 1 MiB whole frame/classification envelopes,
30 seconds startup, 10 seconds per I/O operation and 65,536 map cells at 1 m.
`--points` is 1..65,536; the model point budget also applies. Conservative admission
requires `17,676 + 50 * points` envelope bytes, including maximum supported strings,
labels and timing metadata. `--frame-bytes` is capped at 16 MiB; `--startup-ms` and
`--io-ms` are 1..120,000. Map admission precedes GPU or export construction. Peers
must use matching point limits; receiver links must negotiate identical descriptors.

Pacing uses local monotonic time; point timestamps are explicit synthetic nanoseconds
from trial origin zero. Frame availability follows full window accumulation.
`--rate` must be at least the configured points/window and is capped at 20M/s.
Finite sender elapsed time is measured, without a sustained throughput or physical
capture-age claim. No looping, warmup exclusion or cross-host clock subtraction is
performed. [Manifest replay and multi-file distinct accounting](DISTINCT_REPLAY.md)
use a separate contract. Their measured finite capacity limits and the
[Q0/Q1 evaluation method](QUALITY_EVALUATION.md) remain distinct from this
single-file mechanics qualification; neither establishes sustainable capacity or
general policy quality.

After frames, input completion binds exact encoded-frame SHA256, nonempty counts,
descriptor and fresh terminal/ACK challenges, with EOF fences in both directions.
Producer PASS means input acknowledged only. Receiver PASS also requires independent
classification/map snapshot completion. Subscriber attests validation and ACK sent,
not authority ACK receipt. EOF alone never qualifies completion.

Optional [generic export](../../native/include/synrex/generic_export.hpp) is an
offline subscriber consumer. It verifies the same sealed source SHA256 and exact
local rows, complete single-observation coverage, descriptor, taxonomy and projection,
including ID255 and duplicate LAS codes. `observe()` performs no encoding/publication.
Only after independent output terminal validation does `finish()` encode preserved
LAZ and atomically publish a new bundle without replacement. The bundle contains
exact descriptor/taxonomy/model-manifest bytes, semantic IDs, explicit source-clock
uncertainty, provenance and file hashes. Original record bytes except classification
are preserved. Partial, duplicate, transformed or global row IDs fail. Multi-file
preservation and COPC conversion are unsupported.

SIGINT/SIGTERM requests cancellation at the next application boundary. Pending I/O
remains bounded by its deadline; active vendor work is not preempted. Partial
receipts have FAIL status. Receipt paths are never overwritten. A receipt write
failure makes the process fail even after successful transport.

Optional receiver-only [tensor diagnostics](GENERIC_DIAGNOSTIC_TRACE.md) expose
per-window persistent input/logit traces with a finite total session byte budget.
They are disabled by default and do not qualify trained quality or capacity.
