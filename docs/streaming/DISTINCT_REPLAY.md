<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Tuomas Pietilä -->
# Distinct source admission and bounded decoding

This is the CPU source layer for E3. `synrex-replay inspect` admits an exact
manifest, decodes each file once in its declared order, verifies source identity
after decoding and reports the actual finite duration. It does not run a model
or qualify stream transport, sustainable throughput or sensor latency. The
generic single-file CLI and its preservation exporter remain separate.

Build the ordinary CPU targets, then run:

```sh
build/synrex-replay inspect --manifest replay.json --window-points 16384 --receipt new-receipt.json
```

The receipt path must be new. Failure exits nonzero and cannot publish a PASS
receipt. Manifest and file identities are SHA256 of exact bytes. A serialization
change gives the manifest a new identity; filenames never replace file hashes.
This example uses a placeholder digest;
replace it with the actual source hash and record count before admission:

```json
{
  "schema": "synrex.distinct_replay.v1",
  "workload": "file_order_synthetic",
  "boundary_policy": "reset_each_file",
  "sources": [
    {
      "path": "tile.laz",
      "sha256": "0000000000000000000000000000000000000000000000000000000000000000",
      "records": 12345,
      "coordinate_frame": "tile-local",
      "crs": "UNSPECIFIED:source"
    }
  ]
}
```

Paths resolve against the manifest directory. Only nonempty regular files with
a non-symlink final path component are admitted. Bounds are 1 MiB per manifest,
4,096 sources, 256 MiB per source and 256 printable ASCII bytes per declared
coordinate/CRS identifier. Unknown fields, duplicate JSON keys, raw NULs,
duplicate source digests, invalid counts and uint64 count overflow fail
admission. An alias or byte-for-byte copy of a source cannot add distinct
records. Identical physical points occurring in different source files are not
deduplicated; the distinctness claim is original record identity, not physical
scene uniqueness.

The authoritative key is `(file SHA256, original zero-based record index)`.
Files are assigned prefix offsets in manifest order. The wire ID is the offset
plus the record index, with checked forward and inverse mappings. Each file is
hashed during admission; opening it for decoding checks its original inode,
size, mode, modification/change times and digest. A retained read-only file
descriptor supplies the decoder path. Finishing checks the descriptor and
original pathname again. A source modified during decoding fails qualification.
Readers must call `finish()` after EOF; destruction alone is not success.

Failure receipts retain each attempted source and its actual finite duration.
`raw_records_decoded` counts completed decoder calls; `decoded_records` counts
observations that also pass coordinate validation. `processed_records` counts
observations included in the inspection digests, including unverified prefixes.
An incomplete output window is cleared on failure. `verified_records` counts
only sources that finish with matching identity/hash and complete declared
coverage. Incomplete sources have null digests; a verified first file does not
make a failed second file or the overall trial pass.

The reader supports the existing compressed LAS 1.4 PDRF 6/7/8 layout. It retains
raw signed XYZ integers, scales/offsets, scaled double coordinates, uint16
intensity, return number/count and the exact raw GPS double bits. Original LAS
GPS encoding flags are reported. No timestamp, coordinate transform, CRS or
flight continuity is inferred. GPS bits are not converted into a physical
acquisition clock. The catalog's record counts are declarations until decoding
succeeds. All declared header records are decoded; extra compressed records
beyond the header count and physical EOF are not independently audited.

The label-free observation type never exposes classification. Two diagnostic
digests have a fixed per-record byte layout, concatenated in file order:

| Digest | Little-endian fields per record |
| --- | --- |
| `runtime_features_sha256` | scaled XYZ as three IEEE754 f64, intensity u16 |
| `source_observations_sha256` | raw XYZ as three i32, intensity u16, return number u8, return count u8, raw GPS f64 bits |

Source identity is separate from these digests: erasing labels must change the
file digest while preserving both diagnostics. The tests compare them with an
independent laz-rs decoder, including large translations and negative integer
coordinates. Neither digest is a prediction or semantic-quality measurement.

One source handle, one raw record buffer and one output window of 1–65,536
observations are retained. No full-cloud preload or decode queue exists. LAZ
decoder-internal compressed-chunk buffers are additional allocations and are
not claimed to fit the output window bound. Catalog admission opens and closes
sources serially instead of retaining thousands of descriptors.

Elapsed times use one process's steady clock. Admission hashing is reported
separately; the decode interval includes source reopening/hash checks, decoding,
diagnostic hashing and post-decode verification. This instrumented inspection
has no offered rate, pacing, warmup or inference; its duration is not a 60-second
capacity result. Instrumentation overhead is included and has not been measured
separately. Power and physical capture age are unavailable.

`reset_each_file` declares the required future stream boundary: each unrelated
file must start fresh context/map state. The inspector creates no map. The
existing generic Session binds a single source; dispatch/reset integration,
bounded producer/service queues, independent subscriber coverage and measured
latency/capacity/fault trials remain E3 work. Multi-file preservation is
`NOT_SUPPORTED`: global IDs must never enter the single-file exporter as local
indices. Existing single-file export regression tests remain mandatory.
