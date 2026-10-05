# Spacing defect in `Data/preprocessed_256`

Found 2026-10-04 during Phase 0 recon of the lesion-detection experiment.
**This branch does not fix it.** The preprocessing pipeline and all existing metrics are
unchanged. This experiment works around the defect (see DECISIONS.md, D4) and reports
bridging numbers at the old assumption.

## What is wrong

`scripts/preprocessing/preprocess_256.py` resizes every volume to 256×256×256 with
`scipy.ndimage.zoom`. It does not update the geometry. Every output NIfTI under
`Data/preprocessed_256/` has an identity affine and a pixdim of 1.0 mm on all three axes.
The native volumes it was built from are not 256³ at 1 mm. As a result, a voxel in the
256³ grid has a true physical size that differs per case and per axis:

    true_spacing_256[axis] = native_zoom[axis] × native_shape[axis] / 256

Across all 566 development cases (native headers read from `Data/train/Mets_*` for BMS and
`Raw Data/UCSF_BrainMetastases/train/*` for UCSF):

| | min | median | max |
|---|---|---|---|
| effective in-plane spacing (mm) | 0.70 | 0.94 | 1.23 |
| effective through-plane spacing (mm) | 0.54 | 0.62 | 1.21 |
| true voxel volume (mm³), vs. assumed 1.0 | 0.40 | 0.51 | 1.26 |
| anisotropy (max/min axis) | n/a | 1.44 | 2.08 |

The most common native geometries are 256×256×150 at 0.9375×0.9375×1.0 mm (all 105 BMS),
256×256×126 at 0.859×0.859×1.5 mm, 400×400×90 at 0.75×0.75×2.0 mm, and
320×320×104 at 0.6875×0.6875×1.5 mm.

## The 2× example

`BMS_Mets_005`: the native GT (`Data/train/Mets_005/seg.nii.gz`, 256×256×150 at 0.9375 ×
0.9375 × 1.0 mm) has a physical lesion volume of **820.9 mm³**. The same GT in
`preprocessed_256` (256³, header says 1 mm isotropic) gives **1640.0 mm³** if the header
is believed, a ratio of **2.00**. The through-plane axis was stretched from 150 to 256
slices, so each 256³ voxel is really 0.9375 × 0.9375 × 0.586 mm ≈ 0.515 mm³.

## How it was found

The task brief required reading spacing from image headers and never assuming isotropy.
Checking the headers showed 1.0/1.0/1.0 with identity affines on every preprocessed
volume, while the native volume for the same case was 256×256×150 at 0.9375 mm. Comparing
GT physical volumes between the two grids exposed the factor of 2. Surveying every
native header then confirmed that the effective spacing varies per case (table above).

## Which prior metrics it affects

Everything computed on the 256³ grid that converts voxels to mm assumed 1 mm isotropic.
That includes `jannus.evaluation.metrics` called with `spacing=1.0`, as in
`seven_model_stacker.py`, `eval_stratified_by_size.py` and `eval_measurable_disease.py`.

| Metric | Affected? | Direction / nature |
|---|---|---|
| Voxel Dice, voxel sensitivity/precision | No | Unitless voxel counts. |
| Lesion-wise sensitivity, FP/case, lesion-wise Dice (all lesions) | No | IoU matching on voxel counts. |
| Size bins (`<3`, `3–5`, `5–10`, `10–20`, `>20` mm longest axis), including **sub-3 mm sensitivity 0.061** and **>20 mm 0.986** | **Yes** | The longest bounding-box axis is overstated in mm, by up to ~1.85× along the through-plane axis (typically 1.6×). Lesions are placed in larger bins than they belong in, so the true sub-3 mm population is larger and different from the one reported. |
| RANO-BM measurable-disease scope (≥10 mm longest axis): **sensitivity 0.943** (≥10 mm lesions only), matched Dice 0.835, κ 0.867 | **Yes** | The ≥10 mm filter admits lesions whose true longest axis is below 10 mm, so the scoped population is wrong. |
| SoLD relative error (1.84 %) | Partly | A ratio of predicted to GT diameters. Common scaling largely cancels, but anisotropic scaling does not. |
| Small-lesion sensitivity (<100 voxels, described as "~<5 mm") | Label only | The voxel threshold is consistent, but the "~<5 mm" interpretation is wrong. 100 voxels is ~51 mm³ at the median, ~4.6 mm ESD rather than 5.8 mm. |
| HD95 (**17.26 mm**), MSD | **Yes** | Surface distances are computed in voxel units labelled as mm. With true spacing the values change, typically downward along the through-plane axis. |

**Not affected:** `jannus evaluate` and `jannus predict` read spacing from the input's own
header. At an external site with correct headers they are correct. The defect lies only
in the `preprocessed_256` derived dataset and in research scripts that hard-code
`spacing=1.0` on it.

## How this experiment handles it

- Primary numbers use the true spacing derived from native headers. That covers ESD, size bins, centroid distances, FP distances and HD95.
- Bridging columns repeat the key numbers under the old 1 mm assumption, so they can be compared with prior reports.
- The fix belongs in a separate change. It would either write correct affines in `preprocess_256.py` or carry native geometry alongside the 256³ volumes, and then re-issue the affected figures. That is out of scope here.
