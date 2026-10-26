# Clean RC2 reconstruction without private repositories

<!-- Copyright (c) 2026 Tuomas Pietilä. SPDX-License-Identifier: CC-BY-4.0. -->

[Executed receipt](../../evidence/unet_r3_clean_reconstruction.json): **PASS**.
An independent directory on i5radeon was reconstructed from the accepted local
RC2 archive/bundle, without using a private repository, cached native binary,
existing development venv or existing dependency checkout. Public pinned
laz-perf/json dependencies were freshly cloned and a new venv installed the
published lockfiles (ordinary package-download caching is allowed).

The source is detached at `f8d3ab81804e0c427684ac08f1b055e75206b861` and remains
Git-clean after building/testing. Exact dependency identities are recorded.
Native C++ CPU build and CTest1/1 pass. Python tests:69 PASS,5 GPU-only NOT_RUN.
Ruff, release SPDX/link/path checks, package inspect and relocated CPU install
pass. Original Conv3D ONNX reconstruction is byte-identical to the qualified
artifact: `e9a55b97e1c38e9be057f7f23f395be00418c80d6e160736af53e7a64d43067c`.
All source-provenance and242 archive-member hashes match. Frozen weights and both
ONNX hashes match their freeze. All four real archived examples and the original
synthetic output pass fresh laz-rs checks, semantic/source/model/output bindings,
point order, unrelated record bytes and qualified metadata preservation.

This is independent reconstruction/verification code execution, not independent
human or legal certification. It does not retrain the model, reopen a fresh test,
repeat an expensive GPU campaign or claim a new GPU measurement. RTX execution
remains the accepted R2 receipts. No GPU fallback is hidden in the CPU build.

Commands follow the public [native quickstart](QUICKSTART.md) and
[frozen model quickstart](UNET_R2_QUICKSTART.md). A minimal repetition is:

```bash
sha256sum showcase-rc2.tar.gz
# c72a762fa14de87033e82eff7bc653a906dc2ff117fb14f847b72e0269250092
mkdir reconstruction-directory
tar -xzf showcase-rc2.tar.gz -C reconstruction-directory
cd reconstruction-directory
git clone --branch codex/byom-native-mit-r0 showcase-rc2/source.bundle checkout
cd checkout
git checkout --detach f8d3ab81804e0c427684ac08f1b055e75206b861
mkdir -p .deps
git clone --depth 1 --branch 3.4.0 https://github.com/hobuinc/laz-perf .deps/laz-perf
git clone --depth 1 --branch v3.11.3 https://github.com/nlohmann/json .deps/json
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.lock -r requirements-byom.lock
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DLAZPERF_SOURCE_DIR="$PWD/.deps/laz-perf" \
  -DNLOHMANN_JSON_INCLUDE_DIR="$PWD/.deps/json/single_include"
cmake --build build -j 4
ctest --test-dir build --output-on-failure
.venv/bin/python -m pytest -q
.venv/bin/ruff check .
.venv/bin/python tools/release_checks.py --check-only
.venv/bin/python examples/export_reference.py --out build/reconstructed.onnx
build/synrex-native pack --spec examples/dense_voxel.json \
  --onnx build/reconstructed.onnx --license LICENSES/MIT.txt \
  --out build/reconstructed.srxmodel
build/synrex-native inspect --model ../showcase-rc2/models/unet.srxmodel
cmake --install build --prefix ../install-stage
mv ../install-stage ../relocated
../relocated/bin/synrex-native version
```

Check the dependency SHA256/Git identities against the receipt; do not assume a
tag name alone is immutable. Restore the qualified weights rather than retrain
them for byte-exact inference reproduction. The full R2 publication manifest
binds every archive member by name/size/SHA256; verify it before consuming assets.
Run `verify_semantics` from `src/verify_semantics.py` against each real example
with its binding/taxonomy; use `verify_laz` for the synthetic expected vector.

The source notices preserve Adrian Wolny's MIT architecture, the imported MIT
squeeze/excitation notice, IGN Etalab2.0 source/latest-update attribution, NVIDIA
plan terms, Apache2 codec and the scoped MIT/CC-BY4.0 declarations. The documented
engineering rights assessment permits redistribution of newly trained parameters
with source/date attribution; it is not a general clearance of arbitrary trained
weights. No upstream pretrained checkpoint or restricted private model is used.
