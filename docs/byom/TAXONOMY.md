<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Tuomas Pietilä -->

# Explicit taxonomy artifact

`synrex::Taxonomy` validates metadata for the new generic path. This preparatory
object does not change package-v2, wire-v2, legacy taxonomy mappings, inference,
or map behavior. It does not establish generic file/stream execution qualification.
Package and protocol integration must separately bind the validated artifact.

The artifact is one UTF-8 JSON document, at most **65,536 bytes**, with no duplicate
object keys and at most eight nested object/array containers including the root.
The exact root fields are:

```json
{
  "schema": "synrex.taxonomy.v1",
  "taxonomy_id": "example.surfaces.v1",
  "namespace": "example.cpu",
  "classes": [
    {"semantic_id": 0, "key": "surface_a", "name": "Surface A", "las_code": 42},
    {"semantic_id": 1, "key": "surface_b", "name": "Surface B", "las_code": 42}
  ],
  "provenance": {"source": "CPU fixture", "author": "SynRex tests", "license": "MIT"},
  "unknown_semantic_id": null
}
```

Every object has exactly the displayed fields. Schema versions are exact strings;
unknown fields or versions require a separate schema rather than silent acceptance.

| Field | Contract |
| --- | --- |
| `taxonomy_id`, `namespace` | 1–128 ASCII bytes, grammar `[A-Za-z0-9][A-Za-z0-9._:-]*` |
| `classes` | Ordered array of 2–256 classes |
| `semantic_id` | JSON integer equal to its zero-based array position; IDs include 255 at K=256 |
| `key` | Unique stable class key, 1–64 ASCII bytes, same identifier grammar |
| `name` | Nonempty decoded UTF-8 text, at most 256 bytes, no C0 controls or DEL |
| `las_code` | JSON integer in 0–255; duplicate LAS codes are permitted |
| `provenance` | Exactly `source`, `author`, `license`; each nonempty decoded UTF-8 text, at most 512 bytes, no C0 controls or DEL |
| `unknown_semantic_id` | Required field; `null` or an explicitly declared integer in 0..K−1 |

Booleans, strings and floating-point values do not count as integers. String bounds
count UTF-8 bytes after JSON escape decoding, not Unicode characters. The total
artifact limit applies even when each individual field fits. Unicode is neither
normalized nor case-folded. No class meaning or unknown channel is inferred from
K, filenames, identifier reuse or LAS codes. Distinct semantic classes remain
distinct even when their LAS projections match.

`Taxonomy::read(path)` opens the final component without following symlinks,
checks that the opened artifact is regular, and bounds both its initial size and
the bytes actually read. FIFOs are opened without blocking and rejected. This is
not a promise to reject symlinks in parent directories. Validation and hashing use
the same in-memory bytes, rather than rereading the path for its digest.

`validate_channels` requires exact output channel count. `validate_projection`
requires the entire ordered LAS-code vector, including its length.
`validate_digest` requires the exact lowercase 64-character SHA-256 produced for
the artifact. Digest equality is integrity/identity binding, not a signature or
an assertion that the class meanings are scientifically correct.

The digest covers **the original artifact bytes**, including whitespace, escapes,
field order and any trailing newline. Callers must preserve those bytes. Equivalent
JSON documents can have different identities; implementations must not hash a
parse-and-reserialize result or assume cross-language canonical JSON. Changing a
name or class order while reusing taxonomy ID, K and LAS projection changes the
identity and must fail against an earlier expected artifact digest.

The native CPU regression includes a fixed 329-byte ASCII two-class document with
Python `hashlib.sha256` digest
`a83b5890af6f255159d4764377704a0d98097d0dcc57c5ea4b90455ee23ef241`.
It checks OpenSSL against that independent value, newline identity differences,
four-class and deliberately non-FRACTAL seven-class metadata, complete projection,
changed meaning/order at the same ID/K/projection, K=256/class 255, invalid schema,
duplicate keys, UTF-8 failures, size/depth limits and regular-file admission.
These are metadata contract fixtures; their classifications and names provide no
neural inference, model-quality, GPU or generic-stream qualification evidence.
