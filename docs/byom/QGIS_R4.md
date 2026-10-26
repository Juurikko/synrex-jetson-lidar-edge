# QGIS export and the 48-byte reader failure — R4

<!-- Copyright (c) 2026 Tuomas Pietilä. SPDX-License-Identifier: CC-BY-4.0. Original expression only; upstream material and facts excluded. -->

The R4 export is an optional postprocessing tool. It consumes already classified,
source/model-bound examples; it never changes model weights, features or predictions,
and never runs neural inference. The original source and classified LAZ remain unchanged.

For a convenient existing showcase, open `qgis/ready/synrex-fractal.qgz` from the release
archive in QGIS. Relative layer paths make the directory movable as a unit. The project
contains all seven labelled categories and custom LAS codes 200/201. The authoritative
full ontology remains semantic IDs 0–6 in the sidecars and `SynRexSemanticID` Extra Byte.
LAS 17 is a bridge-deck projection of IGN bridge, not an exact ontology equivalence.

To reconstruct from the archive's qualified input packages, install the pinned verification
requirements in the source venv, plus external PDAL with `writers.copc` and a QGIS-enabled
Python. With Linux installations exposing those commands:

```bash
.venv/bin/python tools/export_qgis.py \
  --input "$RELEASE/qgis/direct" "$RELEASE/qgis/copc" \
  --out "$PWD/build/qgis-export" \
  --pdal pdal --qgis-python /usr/bin/python3 \
  --fallback-crs EPSG:2154
```

The Linux invocation is supplied for equivalent configured environments; the executed
qualification used WSL Python and the existing Windows QGIS 3.44.14 installation:

```bash
.venv/bin/python tools/export_qgis.py \
  --input "$RELEASE/qgis/direct" "$RELEASE/qgis/copc" \
  --out "$WINDOWS_EXPORT" \
  --pdal 'C:\Program Files\QGIS 3.44.14\bin\pdal.exe' \
  --qgis-python 'C:\Program Files\QGIS 3.44.14\bin\python-qgis-ltr.bat' \
  --windows-tools --fallback-crs EPSG:2154
```

Replace the example Windows output location with your own. The WSL tool uses `wslpath`
and generates a quoted batch wrapper; QGIS's adjacent `o4w_env.bat` initializes PDAL.
It rejects unsupported Windows shell metacharacters and refuses an existing output folder.
The explicit EPSG:2154 fallback applies **only to the viewing project** for two source
files lacking CRS VLRs, supported by official IGN dataset context. No vertical datum is
inferred, no coordinate transform or source-header edit occurs. Eight embedded CRSs are
parsed by QGIS and compared after conversion. Failed exports retain `EXPORT_FAILED.json`
and never produce a PASS receipt.

The command verifies original model/source/output/taxonomy bindings, stages an uncompressed
LAS with uint32 `SynRexSourceIndex` and uint8 `SynRexSemanticID`, invokes PDAL COPC conversion
with `forward=all, extra_dims=all`, independently decodes with laz-rs, and restores source
order to compare **all original decompressed point bytes**. Scales, offsets and semantic
IDs must match exactly. Index coverage must be a permutation. COPC metadata is rewritten
for indexing; full byte-preserving metadata claims apply to the separate native LAZ,
not this viewing derivative. Existing Reflectance/Deviation Extra Bytes survive as values.
The QGIS worker creates and reopens a real project, validates provider point counts,
seven-class style and CRS, and records SHA256 identities.

Actual desktop qualification additionally loads that exact project, reads all 1,062,717
points through QGIS's COPC provider, checks each source index, semantic ID and LAS code,
and captures genuine screenshots. See `evidence/unet_r4_qgis_export.json` and the packaged
`desktop-qualification.json`; this is application validation, not static QML validation.
To repeat that desktop stage, launch QGIS with an isolated profile, `--noplugins`,
`--code tools/qualify_qgis_export.py`, and set `SYNREX_QGIS_ROOT` to the exported folder.
Use QGIS's supported environment launcher; keep the export project/data tree together.

The three direct-LAZ failures are `TEST-0737_6243-006721286` (42,377 points),
`TEST-0737_6255-006722653` (72,691) and `TEST-0741_6244-006630086` (85,222).
Both originals and native classified outputs fail with:

> readers.las: Extra byte specification exceeds point length beyond base format length.

Raw inspection finds LAS 1.4, compressed PDRF 8, 48-byte records, one 384-byte Extra Bytes
VLR: Reflectance type 10/double (8 bytes), Deviation type 3/uint16 (2 bytes), options zero.
38 + 8 + 2 = 48. LASzip's three item sizes are 30, 8 and 10, compressor 3, version 3.4.3;
chunk size 50,000. Both WKT VLR copies describe EPSG:2154. The original/noncompression
metadata and all unrelated decoded point bytes match after native classification.
All six files also decode with laz-rs and standalone PDAL 2.10 `info --stats`.
Raw layouts/hashes are recorded in `evidence/unet_r4_laz_layout.json`.

QGIS [#63906](https://github.com/qgis/QGIS/issues/63906) reports the same error with
version/platform-dependent results; [#64612](https://github.com/qgis/QGIS/issues/64612)
was closed as its duplicate. Their matching symptoms support investigating the reader;
we do not assume every upstream example has exactly our cause. Removing VLRs, suggested
in a report, would discard metadata and is not our remedy.

The installed QGIS provider and standalone CLI use the same PDAL DLL SHA256
`11bc301a68c7d405d0685e2bdb7e173ee0599ecfc7fad83c05ac9d1aa5d63d14`.
A separately compiled original C++ diagnostic, `tools/probe_pdal_reader.cpp`, links that
external installed library. Across all six files, prepare alone and preview alone pass;
prepare then preview, and preview twice on the same reader, fail with the exact message.
This identifies a reader reuse failure independently of native LAZ writing and QGIS GUI.

The [QGIS 3.44.14 provider source](https://github.com/qgis/QGIS/blob/final-3_44_14/src/providers/pdal/qgspdalprovider.cpp)
uses prepare followed by preview. In the inspected
[PDAL 2.10 LAS reader](https://github.com/PDAL/PDAL/blob/2.10.0/io/LasReader.cpp),
initialization appends VLRs without clearing the retained list; preview initializes again.
The Extra Bytes reader concatenates the retained descriptors, making the 10 declared
bytes appear as 20 and triggering its record-length guard. This static mechanism explains
the executed lifecycle probe. We have not patched/rebuilt upstream binaries or established
which later upstream releases fix it. Other QGIS/OS versions remain NOT_RUN here.
Diagnostic SDK headers/import library were unpacked locally, not installed or redistributed;
probe/library/source identities and its 24 results are in `evidence/unet_r4_reader_probe.json`.
The existing native source-preserving writer is unchanged.

To reproduce the optional lifecycle diagnostic with installed external PDAL development
headers/library on Linux, build using its advertised `pkg-config` flags, then run each
of `preview`, `prepare`, `prepare-preview` and `preview-preview` on an unmodified rejected
file. The executed Windows build used isolated PDAL 2.10 SDK headers/import library,
Visual Studio 2022 x64, `/EHsc /MD /std:c++17`, `pdalcpp.lib` and `ws2_32.lib`; runtime PATH
was initialized with the existing QGIS `o4w_env.bat`. Example developer-shell build:

```bat
cl /EHsc /MD /std:c++17 /I "%PDAL_SDK%\include" tools\probe_pdal_reader.cpp /link "%PDAL_SDK%\lib\pdalcpp.lib" ws2_32.lib
probe_pdal_reader.exe INPUT.laz prepare-preview
```

The diagnostic's failing exit is expected for the two reuse modes on the qualified files;
it is separate from model execution. All SDK/compiler artifacts remain outside normal Git.
