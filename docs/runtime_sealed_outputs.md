<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Tuomas Pietilä -->
# Rescoring retained semantic outputs

`tools/runtime_sealed_outputs.py` consumes sealed `synrex.point_semantics.v1`
vectors from two retained runs and decodes original LAS/LAZ classification on CPU.
It shares the explicit taxonomy, metric and ground-proxy rules of
[runtime comparison](runtime_comparison.md). It executes no model or engine.

```sh
python tools/runtime_sealed_outputs.py --spec retained-outputs.json --out rescored.json
```

The specification uses `synrex.runtime_sealed_outputs.v1`, with exactly these fields:

| Field | Contents |
| --- | --- |
| `schema` | `synrex.runtime_sealed_outputs.v1` |
| `comparison` | The complete `runtime_comparison.v1` specification, with track `model_policy_quality` |
| `source_las_to_label` | Explicit `[{las_code, label}, ...]` mapping, including ignored labels |
| `runs` | `reference` and `candidate`, each containing `package_manifest: {path, sha256}` and `channel_to_label: [scoring ID, ...]` |
| `sources` | Same order as `comparison.sources`; each entry has original `path` and `reference`/`candidate` sidecar artifacts `{path, sha256}` |

Paths resolve from the process working directory. A package's declared engine hash
must equal that run's declared engine identity. This is artifact binding, not
package admission, engine deserialization or weight verification. Each sidecar must
match its exact source bytes, count, package-manifest bytes and LAS projection;
its mapped-byte digest is independently recomputed from its semantic channels.
Channel-to-scoring-ID mappings are explicit and unique, so LAS projection is never
used to infer semantic meaning. Different projections can retain identical channels.

The adapter hashes each raw source before and after decoding. Classification is
decoded in 65,536-point chunks with explicit `laspy.LazBackend.Lazrs`. The decoded
count must match the manifest and header; every raw label needs an explicit mapping.
Raw classification is used only by the scorer. Coverage and channel disagreement
include ignored truth; confusion and quality metrics exclude it. Sidecar and
package-manifest bytes are rehashed after each source. Per-source reports retain
raw-classification and semantic-vector digests, matrices and ignored support.

Memory is bounded by one source's two sidecars, their channel vectors, chunks,
class matrices and the admitted source/report roster. Each sidecar is capped at
64 MiB and each source at two million points; the specification is capped at 4 MiB.
This is a bounded adapter for retained files, not the O(K²) streaming JSONL scorer.
Malformed UTF-8, duplicate keys, invalid channels, digest/count/order declarations,
unmapped truth and resource excess fail. Reports are exclusively created and never
replace existing artifacts.

Results say `CPU_REEVALUATION_RETAINED_SIDECARS_NO_INFERENCE`, with scientific
qualification `NOT_ASSESSED`. Their inference timing fields remain unavailable.
The tool does not verify output-LAZ preservation, physical EOF beyond declared
records, logits, causal execution, source split independence or historical engine
execution. Those need separate upstream evidence. A consumed historical test set
remains regression data; rescoring it creates no fresh holdout or new quality gate.
In particular, this adapter cannot establish current-source Q0/Q1 or GPU parity.
