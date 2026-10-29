# Independent validation and known numerical differences

<!-- Copyright (c) 2026 Tuomas Pietilä. SPDX-License-Identifier: CC-BY-4.0. Original expression only; upstream material and facts excluded. -->

**Developer and project lead: Tuomas Pietilä ([@Juurikko](https://github.com/Juurikko)).**

Two questions are checked separately: **did native deployment reproduce the qualified model result, and did file output preserve the original data apart from classification?** Exact file preservation is not a model-accuracy metric.

## Historical four-file integrity checks

The native writer used laz-perf; an independent path explicitly selected **laz-rs through laspy** to decode source and output. Source records were compared in source order in bounded chunks. For each 42-byte PDRF-7 record, classification at byte 16 was extracted and hashed, then byte 16 alone was masked before comparing every remaining byte.

This verifies the ordered sequence of coordinates and other attributes, not merely a sorted point set or approximate XYZ equality. Records identical in every other byte are inherently indistinguishable by a byte comparator, so the proof is ordered-record equality plus expected source-order labels, not an external identity tag for duplicate records.

Across all four qualified files:

- Point count and source-order nonclassification record bytes matched exactly.
- Classification vectors matched frozen SHA256 sentinels and native execution receipts.
- Public LAS headers, non-LASzip VLRs and metadata padding matched byte-for-byte.
- Extra Bytes descriptors and values were preserved: `Amplitude`, `Reflectance`, `Deviation`, all int16.
- EVLR count was zero; preservation of nonempty EVLRs was not tested.
- Final classified-LAZ digests matched the accepted frozen results.

LAZ compression metadata is expected to change: re-encoding regenerates the LASzip VLR and compressed stream/chunk table. The qualification does not demand source and output LAZ files have the same whole-file hash. Frozen *output* hashes are checked against the expected classified output, while decompressed nonclassification bytes are compared to the source.

The benchmark writer contract is LAS 1.4 / PDRF 7 / 42-byte records / 375-byte header / point offset 1111 / two VLRs / zero EVLRs. Broader metadata/point-format preservation is not established. [Output digests](../evidence/benchmark.json) are included without restricted files or identifying operational details.

## Numerical model-conversion attribution

A characterized 886,928-point scene was compared against the frozen TreeAIBox/PyTorch oracle. These counts concern point class-ID differences, not general accuracy and not every scene:

| Comparison | Different point class IDs |
|---|---:|
| Pinned TreeAIBox/PyTorch vs historical PyTorch oracle | 0 |
| ONNX Runtime FP32 vs PyTorch | 27 |
| TensorRT FP32 vs ONNX Runtime FP32 | 0 |
| TensorRT FP16 vs PyTorch | 89 |
| TensorRT FP16 vs ONNX Runtime FP32 | 90 |

The first characterized numerical boundary is PyTorch → ONNX/FP32, not an additional TensorRT-FP32 discrepancy. Pairwise mismatch sets overlap; 27, 89 and 90 are not additive. “Exact native parity” refers to the frozen characterized TensorRT-FP16 deployment outputs, not universal PyTorch bit parity. No quality metric was recomputed for the accepted timing runs. [Contributions and method](../../CONTRIBUTIONS_AND_METHOD.md) explains this evidence boundary.

## Runnable synthetic checks in this release

`src/verify_laz.py` is a new public-focused offline utility. It independently parses raw metadata, explicitly decodes with laz-rs and checks ordered nonclassification bytes and expected classification digests. It does not load the C++ writer's implementation or trust its PASS message.

`src/synthetic_laz.py` invents coordinates, GPS times, RGB, return/flag fields and three Extra Bytes attributes. A simple height rule assigns labels. These files are not geospatial observations, ECLAIR derivatives or neural predictions. The 50,013-point case crosses the native writer's 50,000-point compression chunk boundary and uses a different verifier read chunk size (7,001 points).

The 19 tests cover one-point and small/multichunk round trips; exact input immutability; whole uint8 class range independent of flags; detection of attribute/order/classification corruption; raw header/VLR/padding changes; wrong label counts; existing outputs; malformed/unsupported format, EVLR, offset, count and truncation fields; missing output parent; and scratch cleanup.

The new utility and its tests are useful independently of a classifier, but they are not retroactive evidence that a changed production binary or arbitrary LAZ layout is qualified. Test results, tool versions and limitations are recorded in [REPRODUCIBILITY_REPORT.md](REPRODUCIBILITY_REPORT.md).
