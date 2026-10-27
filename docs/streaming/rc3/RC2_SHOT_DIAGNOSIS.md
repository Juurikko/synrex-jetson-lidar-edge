# RC2 shot diagnosis

Actual RC2 H.264 was decoded at all listed timestamps, including every second from 39–48 and 48.9 s. The contact sheet was visually inspected. Historical automated PASS did not establish editorial quality.

Pitch is atan2(eye.z − target.z, horizontal distance), positive downward. All vectors are illustrative local metres, not sensor poses. Exact endpoints appear in evidence/RC2_CAMERA_AUDIT.json.

| Seconds | Scene | Start eye → target | End eye → target | Actual downward pitch |
|---|---|---|---|---|
| [0, 4] | urban | [56.0, -12.0, 19.0] → [19.0, 31.0, 11.0] | [39.141666666666666, -4.066666666666666, 15.033333333333333] → [18.008333333333333, 33.975, 11.991666666666667] | 4.00–8.03° |
| [4, 8] | urban | [7.0, -1.0, 13.0] → [14.0, 34.0, 13.0] | [26.833333333333336, 5.941666666666666, 14.983333333333333] → [19.950000000000003, 38.958333333333336, 13.991666666666665] | -0.00–1.68° |
| [8, 11] | forest | [-4.0, -8.0, 19.0] → [22.0, 27.0, 18.0] | [13.8, -3.0555555555555554, 20.977777777777774] → [26.944444444444446, 28.977777777777774, 19.977777777777778] | 1.31–1.65° |
| [11, 14] | tower | [46.893126107235986, -13.166621926136749, 19.0] → [25.0, 29.0, 20.0] | [63.484000301257616, 3.6696994673585515, 19.0] → [25.0, 29.0, 20.0] | -1.24–-1.21° |
| [14, 25] | urban | [45.0, -9.0, 18.0] → [18.0, 33.0, 12.0] | [9.109090909090913, 0.9696969696969684, 14.012121212121212] → [18.0, 33.99696969696969, 12.996969696969696] | 1.70–6.85° |
| [25, 28] | forest | [-12.766666666666667, -15.966666666666667, 28.977777777777778] → [25.0, 27.0, 19.0] | [7.9999999999999885, -13.000000000000002, 27.000000000000004] → [25.0, 27.0, 19.0] | 9.89–10.43° |
| [28, 33] | forest | [24.867488379654635, -20.13230337787169, 30.978666666666626] → [15.0, 5.0, 20.991999999999962] | [35.65073905668119, -12.393877555417653, 28.991999999999962] → [15.0, 5.0, 20.991999999999962] | 16.50–20.30° |
| [33, 38] | forest | [34.86748837965463, -0.13230337787168978, 31.978666666666626] → [25.0, 25.0, 21.991999999999962] | [45.65073905668119, 7.606122444582347, 29.991999999999962] → [25.0, 25.0, 21.991999999999962] | 16.50–20.30° |
| [38, 45] | urban | [54.0, 6.0, 21.0] → [18.0, 32.0, 12.0] | [30.11428571428571, -3.9523809523809548, 17.01904761904762] → [18.0, 32.0, 12.0] | 7.54–11.46° |
| [45, 49] | urban | [2.0, -9.0, 14.0] → [18.0, 34.0, 13.0] | [17.866666666666674, -2.05833333333333, 14.0] → [18.0, 34.0, 13.0] | 1.25–1.59° |

## Decoded visual findings

0–8 s: near-horizontal church views expose the walls but compress roofs and ground into narrow strips. 8–11 s: tree crowns merge into a forward-facing wall. 11–14 s: slightly upward-looking crane view leaves a large empty centre and flattens ground. 14–25 s: synchronized reveal is authentic but repeats the low pitch. 25–38 s: coarse cubes obscure canopy structure and form an oversized green block wall. 38–45 s: large metric card hides much of the same flattened urban view. 45–49 s: actual closing frames are point splats, not cubes, but the camera is only about 1.25–1.59° down and retains the flattened look. The owner-described ending defect is removed across the late sequence, not relocated.

## Exact block-effect cause

`tools/streaming_r2_rc2.py:voxel_raster` constructs eight corners at ±0.42 m around every actual 1 m map cell centre, projects six faces at focal 850 px, fills depth-buffered triangles and shades cube faces. `Film.draw` dispatches to that function whenever `mapstate is not None`. `Film.frame` supplies mapstate only at 25–38 s, including low 27 m-radius detail orbits. Large projected cube faces plus dense occupancy create the observed wall/pixel motif. It is not H.264 corruption or a pixel-dissolve transition. No block effect is visible in the decoded final ten seconds (39–49); their separate defects are low camera pitch and overlaid metric-card occlusion.

## Inspected evidence

`media/rc2-contact-sheet.jpg`; actual decoded PNGs in `media/rc2-decoded/`; reference MP4 SHA256 `3ac6c0bb98b4f7bc993b347051e2933ec4875dc0c34163de7bce19eda922534d`. RC2 camera metadata was independently recomputed from its exact eye/target vectors.

<!-- Copyright (c) 2026 Tuomas Pietilä. SPDX-License-Identifier: CC-BY-4.0. Original expression only; facts and upstream material excluded. -->
