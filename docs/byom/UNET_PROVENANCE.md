<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Tuomas Pietilä -->
# U-Net source and data assessment

The second dense adapter model uses authentic `UNet3D` from
[pytorch-3dunet](https://github.com/wolny/pytorch-3dunet), pinned to
`7d047d2802561d075d6ce44679d7e696138c47df`. Its
[MIT licence](https://github.com/wolny/pytorch-3dunet/blob/7d047d2802561d075d6ce44679d7e696138c47df/LICENSE)
permits reuse and redistribution with the copyright and licence retained.
The dependency stays external. No upstream checkpoint is downloaded. All learned
parameters start from seeded random initialization. Architecture provenance is
attributed to Adrian Wolny; independent authorship of that architecture is not claimed.

The imported `se.py` credits Anne-Marie Rickmann and relates to
[ai-med squeeze/excitation](https://github.com/ai-med/squeeze_and_excitation),
also MIT (copyright Abhijit Guha Roy); its notice is preserved separately.
No squeeze/excitation module is instantiated or exported in this DoubleConv U-Net.

[FRACTAL's primary dataset card](https://huggingface.co/datasets/IGNF/FRACTAL)
identifies Etalab-2.0 and separate train, validation and test splits. The pilot
pins dataset revision `36e00245ef92fd180f655f6ee410f2e935693cc2`, selecting the
first 256/32/32 sorted LAZ entries from split archive 00. These are real survey
points and labels. Selection is bounded and reproducible, not representative of
the entire benchmark. ZIP CRCs and per-file SHA256s are recorded.
The pinned source repository reports its latest update as 2025-04-05T09:37:48Z;
retrieval occurred on 2026-10-26. Both are carried with redistributable artifacts.

The [official Etalab Open Licence 2.0](https://github.com/etalab/licence-ouverte/blob/master/open-licence.md)
allows transformation, derived information and commercial redistribution, subject
to source and latest-update attribution. The reviewed terms contain no restriction
against training or redistribution of newly trained parameters. Accordingly this
project offers its new weights under MIT while retaining IGN attribution with
weights and exported artifacts. This is a documented engineering rights assessment,
not a legal certification. Real LAZ examples retain Etalab-2.0 separately. Source
licences do not establish rights in arbitrary pretrained weights; none are used.

The pilot maps model channels to LAS codes `[1,2,5,6,9,17,64]`. IGN's
[published configuration](https://huggingface.co/IGNF/FRACTAL-LidarHD_7cl_randlanet/blob/main/FRACTAL-LidarHD_7cl_randlanet-inference-Myria3DV3.8.yaml)
merges classes 3/4 into vegetation 5 and treats sensor artifacts 65/66 separately
from the seven semantic classes. Unknown and artifact labels are ignored during
training and scoring; their point records still receive predictions in output.
Code 64 is the IGN application-specific permanent-structure code, not a universal
ASPRS semantic assignment. No RandLA-Net implementation or weights are incorporated.

New `src/dense_training.py` is a NumPy implementation of the public regular-grid
contract, independently written for training/validation. It uses coordinates,
occupancy, counts and measured intensity only. No ground-truth class or height
above a ground-truth terrain model enters input features. The native regular-grid
implementation retains its existing design-informed provenance boundary in
[PROVENANCE_REVIEW_R1.md](PROVENANCE_REVIEW_R1.md); no clean-room claim is added.
The generic LAZ extension retains the original MIT writer and Apache-2.0 laz-perf.
No private source, restricted model or restricted survey data is incorporated.

Publication remains subject to owner approval. Training and qualification artifacts
are local and ignored by Git; redistributable source and notices can be reviewed
without publishing a model or dataset.

The expanded R2 campaign preserves the same architecture/dataset licence pins
and attribution. Its frozen weights, tensor contract and held-out outputs are
recorded in [R2 qualification](UNET_R2_RELEASE.md). Additional verification and
packaging helpers are original MIT source, not imported preprocessing code.
The QGIS style is original declarative XML using documented renderer fields;
QGIS GPL renderer implementation is not copied or linked into the MIT runtime.
The seven-class taxonomy retains separate IGN Etalab attribution. Neither the
new code nor the expanded tests changes the explicit non-clean-room boundary in
[the preprocessing provenance review](PROVENANCE_REVIEW_R1.md).
