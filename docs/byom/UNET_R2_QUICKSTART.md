# Reproduce the frozen seven-class R2 workflow

<!-- Copyright (c) 2026 Tuomas Pietilä. SPDX-License-Identifier: CC-BY-4.0. -->

Use the [native build instructions](QUICKSTART.md) and the pinned Python/export
[environment setup](UNET_QUICKSTART.md). Python/PyTorch is for training and
verification only. The classification executable needs compatible NVIDIA shared
libraries, no interpreter. Check GPU compute processes/free memory before every
job; do not use AGX resources without a verified nonconflicting allocation.
No public push or upload is part of these commands.

The accepted weights and test are frozen. For inference reproduction, use the
preserved `best.pt` and packages whose hashes are in the release receipt. The
following training commands document how the selected model was obtained; do not
rerun them to select against the consumed test. Fresh independently retrained
weights may differ because MaxPool3d backward is not fully deterministic.

```bash
python tools/fetch_fractal.py --out build/unet-r2/data \
  --train 1000 --val 256 --splits train val
python tools/fractal_inventory.py --data build/unet-r2/data \
  --splits train val --out build/unet-r2/inventory.json
python tools/train_unet.py --data build/unet-r2/data \
  --upstream .deps/pytorch-3dunet --out build/unet-r2/trained \
  --epochs 40 --balanced-blocks --resource-allocation-verified
python tools/evaluate_unet_points.py --data build/unet-r2/data \
  --trained build/unet-r2/trained --upstream .deps/pytorch-3dunet \
  --inventory build/unet-r2/inventory.json --split val \
  --out build/unet-r2/val-points.json --resource-allocation-verified
```

Record the validation decision, weights/ONNX/configuration/protocol hashes and
freeze before test access, following the checked-in
[freeze record](../../evidence/unet_r2_model_freeze.json). The accepted selected
checkpoint is epoch28. Test retrieval was then:

```bash
python tools/fetch_fractal.py --out build/unet-r2/data \
  --test 256 --test-archive 1 --spread --splits test
python tools/fractal_inventory.py --data build/unet-r2/data \
  --splits train val test --out build/unet-r2/full-inventory.json
python tools/audit_fractal_duplicates.py --data build/unet-r2/data \
  --out build/unet-r2/duplicate-audit.json
```

To classify with the final LAS projection, compile the original exported FP16
ONNX into a compatible TensorRT10 engine on the NVIDIA target. Use a fresh output
directory for each package; never overwrite a qualified engine/checkpoint.

```bash
build-gpu/synrex-native pack --spec examples/unet/dense_voxel_asprs_safe.json \
  --onnx build/unet-r2/trained/unet-fp16.onnx \
  --license examples/unet/MODEL_NOTICE.txt --out build/unet-r2/onnx.srxmodel
build-gpu/synrex-native build-engine --model build/unet-r2/onnx.srxmodel \
  --precision fp16 --out build/unet-r2/unet.engine
build-gpu/synrex-native pack --spec examples/unet/dense_voxel_asprs_safe.json \
  --engine build/unet-r2/unet.engine --license examples/unet/MODEL_NOTICE.txt \
  --out build/unet-r2/unet.srxmodel
build-gpu/synrex-native classify --model build/unet-r2/unet.srxmodel \
  --input build/unet-r2/data/test/TEST-0737_6255-006722653.laz \
  --out build/unet-r2/classified.laz --semantics build/unet-r2/semantics.json
python tools/seal_semantics.py \
  --source build/unet-r2/data/test/TEST-0737_6255-006722653.laz \
  --output build/unet-r2/classified.laz --sidecar build/unet-r2/semantics.json \
  --package build/unet-r2/unet.srxmodel --taxonomy examples/unet/semantic_classes.json \
  --destination build/unet-r2/binding.json
```

Keep the source separately. `seal_semantics.py` independently decodes via laz-rs
and produces an output-hash envelope without modifying neural outputs. On another
host, `verify_semantics(source, output, sidecar, package, binding, taxonomy)` from
`src/verify_semantics.py` verifies every binding and point record. No Python
validation helper is called from the native inference executable.

For exact reproduction of the accepted sequence, `qualify_unet_r2.py` builds and
traces the original IGN projection using the frozen model, then
`qualify_las_projection.py` repackages that same engine with the final projection:

```bash
python tools/qualify_unet_r2.py --native build-gpu/synrex-native \
  --data build/unet-r2/data --trained build/unet-r2/trained \
  --upstream .deps/pytorch-3dunet --freeze build/unet-r2/model-freeze.json \
  --inventory build/unet-r2/full-inventory.json --synthetic build-gpu/dense-50013 \
  --out build/unet-r2/qualified --resource-allocation-verified
python tools/qualify_las_projection.py --native build-gpu/synrex-native \
  --data build/unet-r2/data --qualified build/unet-r2/qualified \
  --out build/unet-r2/safe-qualified --resource-allocation-verified
python tools/audit_unet_r2.py --data build/unet-r2/data \
  --trained build/unet-r2/trained --upstream .deps/pytorch-3dunet \
  --qualified build/unet-r2/safe-qualified --traces build/unet-r2/qualified \
  --freeze build/unet-r2/model-freeze.json --out build/unet-r2/independent-metrics.json \
  --resource-allocation-verified
python tools/benchmark_unet_scenes.py --native build-gpu/synrex-native \
  --data build/unet-r2/data --qualified build/unet-r2/safe-qualified \
  --out build/unet-r2/benchmarks --resource-allocation-verified
```

Use the original Conv3D runner's `dense-50013` directory from
[RTX qualification](RTX4090_QUALIFICATION.md) as `--synthetic`.
No runtime recompilation occurs between the two models. Model input/output
names/shapes/types are validated from each `.srxmodel` package. FP16 engine
byte identity is target/build dependent; exact artifact reuse is needed to
reproduce the receipt byte hashes.

For each source, seal the `*-classified.laz`/`*-semantics.json` pair into
`*-binding.json`, then run `tools/verify_unet_outputs.py --data build/unet-r2/data
--qualified build/unet-r2/safe-qualified --taxonomy examples/unet/semantic_classes.json
--out build/unet-r2/final-output-validation.json`. The helper uses the independent
release mapping table, not solely the model manifest lookup.

In QGIS3.44, add the classified LAZ as a **point cloud**. In Layer Properties →
Symbology → Style → Load Style select `examples/unet/fractal-classification.qml`.
Choose the `Classification` attribute. RGB remains preserved original survey
information and is not a neural input; QGIS may initially select RGB by default.
The style labels200/201 as user-defined and documents the bridge projection.
Style XML/taxonomy checks are executed, but actual QGIS application loading has
not been executed on these hosts. A viewer may create its own display index;
that is separate from the original source-order semantic sidecar.

The genuine four-scene comparison can be reproduced using Matplotlib3.10.8:

```bash
python tools/plot_unet_r2.py --data build/unet-r2/data \
  --qualified build/unet-r2/safe-qualified --out media/unet-r2/heldout-semantics.png
python tools/prepare_unet_showcase.py --data build/unet-r2/data \
  --trained build/unet-r2/trained --qualified build/unet-r2/safe-qualified \
  --engine-package build/unet-r2/safe-qualified/unet.srxmodel \
  --synthetic build-gpu/dense-50013 --media-manifest evidence/unet_r2_showcase_media.json \
  --out build/unet-r2/showcase-rc2
```

Archive creation requires committed source, verified output bindings, model
notices and the exact displayed artifact hashes. It copies original/new models,
new weights, ONNX, real/synthetic source-output pairs, seven-class sidecars,
bindings, QGIS style, licences, source export and Git bundle into ignored storage.
It does not authorize publication. See [results/limitations](UNET_R2_RELEASE.md)
and the immutable local release receipt before any owner-approved distribution.
