<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Tuomas Pietilä -->
# Checked laz-perf decoder overlay

The native build uses laz-perf 3.4.0 at upstream commit
`b7bbe26109dc986f42d4fc80b8de3d2b6ca634ce`, with the explicit decoder correction in
[`patches/lazperf-b7bbe261`](../../patches/lazperf-b7bbe261). The upstream checkout
is read only. CMake invokes
[`tools/prepare_lazperf_overlay.py`](../../tools/prepare_lazperf_overlay.py) to
generate `lazperf-overlay` inside the binary directory, and compiles the actual
codec sources from that directory. No prebuilt replacement library, overflow
compiler option, sanitizer suppression or global SDK change is involved.

`manifest.json` pins every byte of the 47 codec files and upstream `COPYING`,
plus the patch hash and expected generated hashes. A Git checkout must also
have the pinned HEAD; a source export must have identical checked bytes. Extra,
missing, changed and symlinked codec files are rejected. Generation checks all
inputs, copies them to a private sibling staging directory, applies unified
hunks at their exact line and context, verifies every result and publishes with
one directory rename under an exclusive local file lock. Concurrent configure
processes share that lock. Repeated configuration verifies the current upstream
and generated files before reusing an identical overlay without changing its
timestamps. An incompatible or altered existing overlay fails closed; configure
a fresh binary directory after changing the checked patch or helper.

`LAZPERF_SOURCE_DIR` continues to identify the pristine external upstream tree.
Normal CMake commands and `SYNREX_SANITIZE=ON` apply to the overlay's compiled
codec as well as the callers. A minimal standalone preparation is:

```sh
python3 tools/prepare_lazperf_overlay.py \
  --source /path/to/pinned/laz-perf --output /path/to/build/lazperf-overlay
```

The correction defines modulo-2^32 reconstruction for integer decoder
predictor/corrector addition, Point10 and Point14 X/Y addition, signed range
folding and the k=31 corrector mask/increment. Conversion of a large unsigned
32-bit result into signed int32 uses only representable casts. Shared integer
decoder changes also affect other integer-coded fields, including Z.
`MemoryStream::getByte` and `getBytes` reject exhausted or invalid positions;
aggregate reads use `size_t` and check the entire read before modifying the
output or index.

Upstream laz-perf is Apache-2.0. The derivative patch retains upstream notices;
each changed file has an explicit SynRex modification notice. The generated
tree retains upstream `COPYING` and adds `SYNREX_OVERLAY_NOTICE.txt`. The helper
and regression drivers are original MIT project code; this does not relicense
upstream or the derivative patch as MIT.

The maintained native `checked_lazperf_decoder` CTest covers empty/exhausted
streams, oversized aggregate reads, `SIZE_MAX` indices, defined integer
endpoints and k=31/32 correctors. The Python generator regression can be run
against the admitted upstream tree with:

```sh
SYNREX_TEST_LAZPERF_SOURCE_DIR=/path/to/pinned/laz-perf \
  python -m pytest -q tests/test_lazperf_overlay.py
```

The source-reader tests generate compressed PDRF6/7/8 fixtures with independent
laz-rs and compare observations/digests; preservation tests independently
decode exported LAZ and compare every nonclassification record byte, including
modern formats and extra bytes. The round-15 diagnosis retained actual upstream
ASan/UBSan failures for extreme coordinates and inflated header counts. The
round-16 receipt binds the checked build overlay, actual codec instrumentation,
commands and regression results. These campaign receipts live outside the
source checkout in `reviews/E3-round15-codec` and `reviews/E3-round16-codec`.

This is a bounded decoder correction, not blanket malicious-input hardening or
a physical EOF proof. Arbitrary compressor coordinates remain outside the
qualification: upstream compression still has signed subtraction, negation and
mask arithmetic at extreme values. GPS-time predictor multiplication and signed
64-bit bit-pattern arithmetic need separate adversarial qualification. The
inactive historical `field_xyz.hpp` is unchanged. No new arbitrary-file-size,
allocation-bound or malformed-stream security guarantee follows from these
tests. The replay reader still reports declared-count consumption separately
from unverified physical EOF.
