# Reproduce the trained dense U-Net pilot

<!-- Copyright (c) 2026 Tuomas Pietilä. SPDX-License-Identifier: CC-BY-4.0. -->

Start from [native build/install instructions](QUICKSTART.md) with a compatible
NVIDIA GPU, CUDA toolkit and TensorRT 10.x. Source and CPU work can run without a
GPU. Training, engine compilation and qualification require an independently
verified nonconflicting allocation. Do not use an active AGX autopilot's resources.
This is the fixed dense adapter, not arbitrary ONNX or sparse transformer support.

Use separate development environments for training/oracles and optional plotting.
No Python/PyTorch executable, package or interpreter is needed by native classification.
The executed reference environment reused existing PyTorch packages read-only;
the following gives a fresh-install equivalent without modifying another environment.

```bash
python3.11 -m venv build/unet/env
build/unet/env/bin/pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
build/unet/env/bin/pip install -r requirements-unet.lock \
  --extra-index-url https://download.pytorch.org/whl/cu128
build/unet/env/bin/pip install --no-deps -r requirements-unet-validation.lock
# The validation bindings use the already installed matching TensorRT SDK libraries,
# including libnvinfer_plugin.so.10; --no-deps avoids downloading a second SDK.
git clone https://github.com/wolny/pytorch-3dunet .deps/pytorch-3dunet
git -C .deps/pytorch-3dunet checkout --detach 7d047d2802561d075d6ce44679d7e696138c47df
build/unet/env/bin/python tools/fetch_fractal.py --out build/unet/data
```

The default download is 256 train/32 val/32 test LAZ tiles, about 199 MiB extracted,
using HTTP ranges rather than whole 0.8 GB archives. Ranges must return 206.
Existing downloaded tiles are rechecked against ZIP size and CRC. Retain the exact
`dataset.json`, source revision/date, original data and SHA256s. Use this fixed
selection for the reported pilot, not a randomly altered subset. Real source
information remains Etalab-2.0; see [rights and attribution](UNET_PROVENANCE.md).

After verifying GPU availability, train and export:

```bash
build/unet/env/bin/python tools/train_unet.py \
  --data build/unet/data --upstream .deps/pytorch-3dunet \
  --out build/unet/trained --epochs 60 --resource-allocation-verified
```

This runs the original upstream architecture and produces seeded new weights,
FP32 and FP16 ONNX files, training history and a receipt. Output directories must
be new. The test split is not opened. PyTorch warns about nondeterministic
MaxPool3d backward; retraining can produce slightly different weights. Exact
inference reproduction uses the preserved qualified model hashes, not an assertion
that independent training reproduces identical bytes.

Package and compile on the target, with vendor library paths configured as in the
native quickstart:

```bash
build-gpu/synrex-native pack --spec examples/unet/dense_voxel.json \
  --onnx build/unet/trained/unet-fp16.onnx --license examples/unet/MODEL_NOTICE.txt \
  --out build/unet/onnx.srxmodel
build-gpu/synrex-native build-engine --model build/unet/onnx.srxmodel \
  --precision fp16 --out build/unet/unet.engine
build-gpu/synrex-native pack --spec examples/unet/dense_voxel.json \
  --engine build/unet/unet.engine --license examples/unet/MODEL_NOTICE.txt \
  --out build/unet/unet.srxmodel
build-gpu/synrex-native classify --model build/unet/unet.srxmodel \
  --input build/unet/data/test/TEST-0436_6384-002325248.laz \
  --out build/unet/classified.laz
```

Inspect the engine-build JSON for Half tensor/layer formats and exact I/O shapes.
An FP16 permission flag alone is not evidence of FP16 execution. Plans/packages
are target-specific and must be rebuilt on a different GPU or incompatible stack.
The runtime is compiled once: switching `--model` between the original Conv3D
engine package and this U-Net package requires no rebuild. Record binary hashes
before/after the switch. Preserve model notices with every exported artifact.

To repeat the full numerical/native/independent qualification, first create the
original synthetic package using `tools/qualify_dense_gpu.py` from the main
quickstart, then use its output directory as `--synthetic`:

```bash
build/unet/env/bin/python tools/qualify_unet.py \
  --native build-gpu/synrex-native --data build/unet/data \
  --trained build/unet/trained --upstream .deps/pytorch-3dunet \
  --synthetic build-gpu/dense-50013 --out build/unet/qualified \
  --resource-allocation-verified
SYNREX_RUN_GPU=1 SYNREX_TEST_NATIVE=build-gpu/synrex-native \
  SYNREX_TEST_WRITER=build-gpu/synrex-laz-labels .venv/bin/python -m pytest -q
build/unet/env/bin/python tools/benchmark_native.py \
  --native build-gpu/synrex-native --model build/unet/qualified/unet.srxmodel \
  --input build/unet/data/test/TEST-0436_6384-002325248.laz \
  --out build/unet/benchmark --resource-allocation-verified
```

The numerical oracle needs TensorRT Python bindings matching the native SDK;
it uses PyTorch CUDA buffers only in its own validation process. Native commands
run with PATH=/nonexistent and loader diagnostics. Compare every label to the
independently prepared tensor oracle, then use explicit laz-rs verification on
another host as well. Keep all logs, outputs and failures. The reported gates
and rare-class limitations are in [the qualification report](UNET_QUALIFICATION.md).

For the figure, install matplotlib 3.10.8 with laspy/lazrs/NumPy in a separate
plotting environment and run `tools/plot_unet_showcase.py --data build/unet/data
--classified build/unet/qualified --out media/unet/pilot-heldout.png`.
This plots actual classified outputs; its source selection is disclosed.
The checked-in figure is review material. Model, LAZ examples and release archives
stay in ignored build directories pending explicit owner publication approval.

To prepare the same local showcase layout after qualification, supply the exact
qualified packages and synthetic runner directory. This copies both genuine
models, the synthetic pair and two disclosed real-source/classified pairs,
all notices, a complete source export and Git bundle. It performs no network write:

```bash
.venv/bin/python tools/prepare_unet_showcase.py \
  --data build/unet/data --trained build/unet/trained \
  --qualified build/unet/qualified --engine-package build/unet/qualified/unet.srxmodel \
  --synthetic build-gpu/dense-50013 --out build/unet/showcase-rc1
```

Use committed, reviewed source and retain the resulting archive SHA256 and
`publication_candidate.json`. The archive manifest separates real Etalab-2.0
data, original MIT source/parameters and vendor plan terms. Source/compiled-model
hash checks are mandatory; archive creation does not authorize publication.

The expanded frozen seven-class campaign is recorded separately in
[the R2 release qualification](UNET_R2_RELEASE.md) and
[its reproducible quickstart](UNET_R2_QUICKSTART.md). The results above remain the historical R1 pilot.
