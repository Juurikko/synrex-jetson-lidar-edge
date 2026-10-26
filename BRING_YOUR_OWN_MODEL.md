# Bring your own model: from point labels to validated classified LAZ

<!-- Copyright (c) 2026 Tuomas Pietilä. SPDX-License-Identifier: CC-BY-4.0. Original expression only; upstream material and facts excluded. -->

**Developer/project lead:** Tuomas Pietilä ([@Juurikko](https://github.com/Juurikko)) · AI-assisted software engineering using OpenAI Codex.

**Native BYOM R0:** [native build, packaging and classification quickstart](docs/byom/QUICKSTART.md) and [adapter contract](docs/byom/ADAPTER.md) cover the generic CUDA/TensorRT runtime on this local branch. [Physical RTX4090 qualification passes on 50,013 synthetic points](docs/byom/RTX4090_QUALIFICATION.md); physical AGX qualification remains NOT_RUN. This does not expose the private reference implementation or restricted pretrained weights.

The guide below retains the independently usable label-writer workflow. Its integration contract applies to `synrex-laz-labels`; the new `synrex-native` runtime performs geometry, inference, reconstruction and explicit class mapping itself within its declared dense-voxel adapter.

The [second-model U-Net quickstart](docs/byom/UNET_QUICKSTART.md) now demonstrates
trained real-data segmentation through the same binary, with [measured quality,
limitations and preservation evidence](docs/byom/UNET_QUALIFICATION.md).

## The integration contract

| You supply | This repository supplies |
|---|---|
| A compressed, supported `INPUT.laz` | Native C++ LAZ point-record writer based on pinned [laz-perf 3.4.0](https://github.com/hobuinc/laz-perf) |
| `LABELS.u8`: exactly one **uint8 LAS classification code** per source point, **in original point order** | Substitutes the classification byte of every point record without changing other decompressed record bytes |
| Your own model/algorithm and any necessary point-index reconstruction and class mapping | Independent laz-rs validation, synthetic fixtures and negative/integrity tests |

The writer does **not** infer classes, train a model, calculate confidence, solve georeferencing, reorder predictions or interpret your model's ontology. If preprocessing downsamples, reorders, voxelizes or duplicates points, **your integration must reconstruct one prediction for each original point** before writing `LABELS.u8`.

### Supported file boundary (deliberately narrow)

The current utility accepts compressed **LAS 1.4, point formats 6/7/8, at least
30/36/38 record bytes respectively**, including extra bytes, with no EVLRs,
waveform data or COPC indexing. It rejects unsupported formats, refuses output
replacement and operates on files rather than a live stream. Check each actual
survey against this contract; LAS 1.2 and legacy/waveform point formats remain unsupported.

For supported inputs, it preserves point count and source order, public LAS header, non-LASzip VLRs and metadata padding, and all decompressed point-record bytes except classification. The LAZ compression VLR is regenerated, so the whole compressed file is **not** byte-identical. Its original qualified production corpus consisted of four LAS 1.4/PDRF7 files with 42-byte records and no EVLRs; the public tool and tests are a separate implementation.

## First: reproduce the model-free example

Requires Linux, Git, Python 3.12 with venv, CMake ≥ 3.20 and a C++17 compiler. A Jetson, TensorRT, CUDA, real LiDAR dataset and neural-network weights are **not** needed for this demonstration.

```bash
git clone https://github.com/Juurikko/synrex-jetson-lidar-edge.git
cd synrex-jetson-lidar-edge

python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.lock
mkdir -p .deps
git clone --branch 3.4.0 --depth 1 https://github.com/hobuinc/laz-perf.git .deps/laz-perf
git -C .deps/laz-perf rev-parse HEAD
# Pinned source: b7bbe26109dc986f42d4fc80b8de3d2b6ca634ce

cmake -S . -B build -DLAZPERF_SOURCE_DIR="$PWD/.deps/laz-perf" -DCMAKE_BUILD_TYPE=Release
cmake --build build --target synrex-laz-labels -j 4
.venv/bin/python tools/synthetic_demo.py --points 50013
.venv/bin/python -m pytest -q --basetemp=build/pytest
```

The `synthetic_demo.py` script generates invented LAS records, creates example labels using a simple height rule, runs the **actual native writer**, and independently verifies the output. That height rule is **not a trained classifier**. The 50,013-point result is a model-free functional demo, **not** the Jetson 210,318 points/s measurement. The example refuses to overwrite its output; use a fresh `--out` directory if repeating it.

## Next: integrate the labels from your own model

1. **Inspect the input file**. Confirm LAS 1.4 / compressed PDRF 6, 7 or 8 / valid record length / no unsupported EVLR, waveform or COPC structure. Preserve the original file; don't perform an implicit laspy rewrite just to make the example work.
2. **Run your own segmentation/classification method**. Maintain a stable mapping from its working representation back to the **original source point index** (0 to N−1).
3. **Translate model class IDs to intended LAS/ASPRS classifications**. Do not write raw neural IDs directly unless they are already exactly the desired output codes. Example conventional codes include 2 for ground and 6 for buildings; LAS's low/medium/high vegetation codes denote defined categories and should not be assumed equivalent to an arbitrary model's single "vegetation" output.
4. **Export one unsigned byte per source point** to `labels.u8`: no header, separators or indices; binary file length must equal the original point count. Your code must reject missing, duplicate, non-mapped or out-of-range predictions rather than silently reshuffling or truncating.
5. **Apply labels and verify separately**:

```bash
build/synrex-laz-labels input.laz labels.u8 classified.laz
.venv/bin/python src/verify_laz.py input.laz classified.laz labels.u8
```

The independent verifier reads both LAZ files through **laz-rs** (via laspy), compares all nonclassification point-record bytes in source order, validates the new classification bytes and compares qualified metadata/header regions. Treat a verifier failure as a **failed output**, not a warning.

For a byte stream you already have in Python, the export step can be as simple as `Path("labels.u8").write_bytes(bytes(mapped_source_order_las_codes))`, but only **after** checking length, uint8 range, semantic mapping and source-order identity. Python in model/test tooling does **not** contradict the lack of Python dependencies in the separately demonstrated native classifier.

## Practical application example

**Surveying or forestry:** A developer has their own ground/building/vegetation model and a supported LAZ tile. Their model generates classifications and maps its output to appropriate LAS codes. This tool produces a classified LAZ while preserving existing coordinates, intensity, GPS time, RGB, return information and Extra Bytes; the independent verifier supplies an integrity check before downstream GIS/QGIS workflows.

This is a **file-delivery integration**, not a validated field service, automatic confidence calibration, model-accuracy assessment or QGIS plugin. A third-party model remains responsible for its predictions.

## Common mistakes to avoid

- **Wrong point order:** A voxel prediction array is not automatically ordered like the input LAZ.
- **Wrong class vocabulary:** A model's "class 2" can mean something completely different from LAS ground class 2.
- **Loss of prior classification:** The utility **replaces** the classification byte. If authoritative existing classes must survive, design a new versioned product or extra-dimension/sidecar approach first; that is **not implemented here**.
- **Unsupported input:** LAS 1.2, varying point formats/record lengths, EVLRs, waveform packets, and COPC require other tools or new qualification.
- **Confused benchmark attribution:** The public utility is **not** the frozen private product writer. No publicly reproducible Jetson model benchmark is claimed.
- **Publishing restricted imagery/data:** Follow dataset licenses, sensitive-feature restrictions, model-weight rights and applicable institutional agreements.

See [architecture](ARCHITECTURE.md), [verification evidence](VALIDATION.md), [performance protocol](BENCHMARK.md) and [third-party notices](THIRD_PARTY_NOTICES.md).

---

**License:** Original guide text CC BY 4.0, copyright © 2026 Tuomas Pietilä for protectable original expression. Original code/tests in the repository are MIT where indicated. Upstream components keep their own notices.
