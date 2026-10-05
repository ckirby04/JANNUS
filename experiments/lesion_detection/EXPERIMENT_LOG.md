# Experiment log

Append-only. Each entry is dated and names the **split** that every number comes from.
Nothing here is edited after the fact; corrections are made in new entries.

Reference baseline: overall lesion sensitivity **0.810 at 1.68 FP/case** (84-case cohort,
threshold 0.55, prior JANNUS metrics). 0.943 is sensitivity for **lesions ≥10 mm only**.

---

## 2026-10-04: Phase 0 recon and plan

- Split: none. No model outputs were computed.
- Wrote PLAN.md. Found that `preprocessed_256` misstates spacing (SPACING_DEFECT.md), that no probability maps or split ID lists exist in 1.50, and that the 0.55 threshold was chosen on the 84-case cohort.
- Baseline environment: pytest 238 passed; `ruff check .` shows 8 pre-existing errors, all in `run_demo.py`.

## 2026-10-04: Phase 0 approved; setup

- Tagged `v1.50.0` (annotated) on `d43dc97`. Branch `exp/lesion-detection-eval` is based there.
- **One-time read (Q2 exception).** Read only these two files under `G:\BrainMetShare\1.40`, and copied only IDs:
  - `model/evaluation_results/stacking_v5_results_per_case.json`
  - `nnUNet/nnUNet_preprocessed/Dataset001_BrainMets/splits_final.json`
  - Sources and hashes are in `configs/splits/SOURCES.md`. Equivalent paths under 1.30 and 1.1 were existence-tested only, and none exist. The stacker internal-validation list was never logged.
- **Contamination check.**
  - Split: 84-case cohort IDs vs nnU-Net folds. This is ID set arithmetic only; no labels or predictions were used.
  - The stacker-validation IDs are unavailable, so the requested check cannot be done directly. Decision: use the 2-fold fallback within the 84 (DECISIONS D5).
  - 24/84 scans are in nnU-Net fold-0 **training**, the fold behind nnU-Net 3D/2D and SwinUNETR.
  - UCSF letter suffixes are timepoints of one patient, per TableS1 in `Raw Data` (read-only). The 84 scans are 77 patients, and 33 of those have other scans in the stacker training pool.
  - Clean subset: 29 scans / 28 patients, as an upper bound (patch-model training sets and synthetic-derivative sources cannot be checked).
  - Q8 raised with the owner.
- **Spacing survey.** Split: all 566 development cases, native headers only. The median true voxel volume on the 256³ grid is 0.515 mm³, against the assumed 1.0. Through-plane spacing is 0.54–1.21 mm. Details in SPACING_DEFECT.md.
- **Observed bug, not fixed.** `metrics.compute_case_stratified` compares against the scipy `label` function instead of `bin_label`, so the per-bin FP column has always been 0.
- Created a nested `.gitignore` that ignores `cache/`.

## 2026-10-04: Phase 1, lesion extraction and matching

- Split: none. Synthetic volumes only.
- Added `src/jannus/evaluation/detection/{__init__,geometry,matching}.py` and `tests/unit/test_detection_matching.py` (24 tests).
- **Bug caught by the tests before any real data was used.** `MatchResult.detected(tau=-inf)` treated unhit lesions (score −inf) as detected, because `-inf >= -inf`. Fixed by requiring a hit explicitly. The empty-prediction, simple-miss, strict-fragment and both centroid tests failed before the fix and pass after it.

## 2026-10-04: Q5 UCSF ground-truth comparison

- Split: all 461 UCSF development scans. This compares GT files only; no predictions were used.
- Script: `tools/compare_ucsf_gt.py`. Output: `results/q5_ucsf_gt/ucsf_gt_comparison.csv`.
- **A = `preprocessed_256/.../seg.nii.gz` is exactly the original `<id>_seg.nii.gz` (B), nearest-neighbour zoomed to 256³.** Dice is 1.0000 on all 461 scans. The existing evaluation therefore used the original 2020 release annotations.
- **C = `<id>_BraTS-seg.nii.gz`** is the 2023 BraTS-METS neuroradiologist revision. It exists for 324/461 scans (54 of the 72 eval-cohort UCSF scans). It uses labels 1, 2 and 3; label 3, enhancing tumour, is the counterpart of B's label 1.
  - Dice of B vs C label 3: median 0.961, quartile 1 0.908, minimum 0.167.
  - Dice of B vs all C labels: median 0.454. Including oedema is not comparable.
  - Lesion count (26-connectivity, native grid) differs in 91/324 scans: 3,278 lesions in B vs 3,350 in C label 3.
- Decision pending the owner (DECISIONS D7).
