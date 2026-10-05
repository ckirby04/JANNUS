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

## 2026-10-04: Owner decisions on Q5, Q7 and Q8; Phase 1 committed

- Phase 1 committed as `9fee455` on `exp/lesion-detection-eval`.
- **Q5.** The original `<id>_seg.nii.gz` with corrected spacing is used for every main number. The BraTS-revision side analysis on 54 eval scans is approved (D7).
- **Q8.** Split design changed (D5.1). PRIMARY: tune on the 55 contaminated scans, evaluate once on the 29 clean scans. SECONDARY: the patient-grouped 2-fold over all 84.
  - Split: 84-case cohort, using ID arithmetic only. `tools/build_leakage_splits.py` wrote `clean_ids.txt` (29 scans / 28 patients) and `contaminated_ids.txt` (55 / 49). No patient appears on both sides.
  - The two sources that cannot be checked (patch-model training sets and synthetic BMS provenance) are recorded in D5.1. The owner will supply what is known.
- **Q7.** No per-case production-stacker file can exist; the scripts write aggregates only. The aggregate gate is in use (D8). No frozen file was opened.
- Wrote LEAKAGE_FINDING.md and PRIOR_DEFECTS.md (an index of 6 defects in existing code, data and docs). No existing doc was edited.

## 2026-10-04: Phases 2–4 tooling; inference; REPRODUCTION GATE FAILED (stopped)

- Phase 2 committed as `6a1d251`: scorers, provenance helper, inference wrapper, gate script.
- Written and unit-tested, but **not yet run on real data**:
  - `detection/metrics.py`, `bootstrap.py`, `report.py` (13 tests);
  - experiment drivers: `build_components.py`, `select_operating_point.py`, `evaluate.py`, `pool_2fold.py`, `brats_side_analysis.py`, `make_report.py`, `tools/make_2fold.py`;
  - DECISIONS D9–D15;
  - PRIOR_DEFECTS #7: `jannus predict` provenance omits the nnU-Net hashes.
- **Inference.**
  - Split: all 84; contaminated first, then clean.
  - `run_inference.py` ran the unchanged `BrainMetPipeline.from_config(verify=True)` on `Data/preprocessed_256/train` inputs through `load_case`, the same path as `jannus predict`.
  - 84/84 cases at about 55 s each, 4–7 MiB per map.
  - The manifest and provenance are in `results/inference/`.
- **Reproduction gate.**
  - Split: all 84. Labels were used only to compare against the logged aggregates; nothing was selected.
  - Conventions: the original `seven_model_stacker.py` ones (`p > 0.55`, 1 mm spacing, 6-connectivity, IoU ≥ 0.1).
  - Result: **FAILED** on every check.

  | Check | Logged | Observed | Δ |
  |---|---|---|---|
  | mean voxel Dice | 0.7858 | 0.6093 | −0.1765 |
  | mean lesion sensitivity | 0.810 | 0.5630 | −0.2470 |
  | total FPs | 141 | 223 | +82 |
  | matched lesion Dice | 0.791 | 0.7143 | −0.0767 |

  - Per the pre-declared protocol (D8): stop and report. **No debugging against these labels.** The contaminated-split component build was stopped. No detection results were computed or read.
- **Label-free diagnostic** (tuning split only, predictions only, no GT). Per-model median voxels ≥ 0.5 per scan:

  | Model | Median voxels ≥ 0.5 | Dice vs majority of the other 6 |
  |---|---|---|
  | nnunet_3d | 2,245 | 0.685 |
  | nnunet_2d | 870 | 0.296 |
  | patch_8 | **141,189** | **0.022** |
  | patch_12 | **40,486** | **0.123** |
  | patch_24 | 3,427 | 0.533 |
  | patch_36 | 2,371 | 0.717 |
  | swin_unetr | 3,440 | 0.679 |

  `patch_8` and `patch_12` flag one to two orders of magnitude more voxels than the other models, and agree with almost none of them. This is consistent with the regenerated base maps differing from the cache the stacker was trained on (1.30 "cache-build protocol", `model/stacking_cache_v5*`), most visibly for the small-patch models. Whether that is their normal behaviour (and the stacker learned to discount it) cannot be decided without the original cache. Not investigated further, pending the owner.

## 2026-10-04: Reproduction-failure diagnosis (option 1; diagnosis only)

- **Owner exception.** One-time read-only access to `1.40/model/stacking_cache_v5/` and `stacking_cache_v5_256/`.
  - Used for 5 TUNING cases only: `BMS_Mets_039`, `BMS_Mets_051`, `UCSF_100102A`, `UCSF_100110A`, `UCSF_100319A`. The script enforces that they are tuning cases and not clean ones.
  - Only exact file paths were tested and opened. No directory was listed.
  - The `mask` key was never read. No labels were used. Nothing was run on clean cases.
  - Ten cache files were read; their SHA-256 hashes are in `results/reproduction_diagnosis/provenance.json`.
- **Scripts.** `tools/diagnose_reproduction.py` and `tools/diagnose_nnunet_cache_variants.py`. They use runtime-only forward pre-hooks and change no pipeline code, config, weight or weights.lock.
- **Findings** (full write-up: REPRODUCTION_FAILURE.md):
  - SwinUNETR reproduces its cache exactly (r = 1.000).
  - nnU-Net 3D/2D reproduce **exactly** the `v5_256` nnU-Net variant, which ran on the resampled 256³ grid and was documented in 2026-04 as calibration-broken. They do **not** reproduce the `v5` "native" variant the stacker was trained on (fresh-vs-cached Dice 0.23–0.95).
  - The cached patch_8/12 maps behave like the other models. Fresh v3 patch_8/12 flag 3–56× more voxels.
  - Stacker swap test: restoring cached nnU-Net 3D is the largest single correction in 4/5 cases. Cached nnU-Net + Swin with fresh patches agrees with the all-cached output at 0.97–0.98 in 4/5 cases, and the patch models dominate in 1/5.
  - No sigmoid/logit mismatch. Stacker windowing matches training. No checkpoint hash changed across weights.lock's git history, and no cache-era hash is recorded anywhere readable.
- **Ranked hypotheses.**
  1. nnU-Net input path (confirmed).
  2. Patch-model generation, v3 vs legacy (strong but circumstantial; protocol not excluded).
  3. Shared normalisation, sigmoid/logit, stacker config and storage were ruled out or found negligible.
- **Correction.** SPACING_DEFECT.md said every `preprocessed_256` volume had an identity affine. A survey of all 566 headers shows that holds only for the 105 BMS volumes. The 461 UCSF volumes keep their native affine and zooms on the resized grid. The file and PRIOR_DEFECTS #1 were corrected. The experiment's true-spacing derivation is unaffected.
- PRIOR_DEFECTS #8 was added (status: under diagnosis).
- Per the owner's instruction: stopped after the write-up. Option 2 (native inputs) has not been started.

## 2026-10-05: PRE-COMMITMENT for the label-based cached-map check (written and committed before running)

Purpose: measure whether the unchanged production stacker reproduces the published 84-case
figures when fed the base maps it was trained on. This is a measurement of the published
system. **Nothing about the operating point is chosen from it.**

- **Owner grant.** Read-only access to `G:\BrainMetShare\1.40\model\stacking_cache_v5\` for all 84 eval cases, for this one run. Model keys only: `nnunet`, `nnunet_2d`, `patch_8`, `patch_12`, `patch_24`, `patch_36`. The `mask` key is never read.
- **Frozen code.** `experiments/lesion_detection/cached_map_reproduction.py` and `configs/50_cached_map_reproduction.yaml`, as committed together with this entry. The production stacker checkpoint, the adapters, models.yaml and weights.lock are unchanged.
- **Settings.** Those of the original `seven_model_stacker.py:evaluate()`:
  - stacker sliding window 32³, overlap 0.5, 9-channel features;
  - `postprocess_prediction((p > 0.55).astype(float32), 0)`;
  - that script's `voxel_dice`;
  - existing `compute_case_metrics(spacing=1.0)`, which uses scipy's default 6-connectivity and IoU ≥ 0.1 greedy matching.
- **Gate**, judged on all 84 scans together:
  - mean voxel Dice within ±1e-3 of 0.7858;
  - total FPs within ±2 of 141;
  - mean lesion sensitivity within ±0.005 of 0.810.
- **Strata.** The same metrics are reported separately for clean (29) and contaminated (55). They are reported only, never used to decide anything.
- **Outcome rule.** The result is reported as-is. If it misses: stop and report. No debugging against labels, and none against clean cases under any circumstances.
- **Two departures from the literal instruction, decided now, before any result exists:**
  1. *SwinUNETR source.* `stacking_cache_v5` has no Swin key; its keys are `mask, nnunet, nnunet_2d, patch_8/12/24/36`. The stacker was trained with Swin from `stacking_cache_v5_256`, which this grant does not cover. Swin is therefore generated fresh, in float32, by the unchanged `SwinUNETRAdapter` on the same `load_case` input. On all 5 diagnosis cases, fresh Swin matched cached `v5_256` Swin at r = 1.000 with max abs diff ≤ 0.005 (REPRODUCTION_FAILURE.md §1). Any residual Swin difference is small and is noted as a possible contributor if the gate misses narrowly.
  2. *Ground truth.* `Data/preprocessed_256/train/<case>/seg.nii.gz`, not the cache's `mask` key, which the original script used and this grant excludes. Q5 established that this file is the original annotation resized to 256³. It is assumed, not verified, to equal the cache `mask`.
- **Output.** `results/cached_map_reproduction/`: `metrics.json`, `per_case.csv`, `provenance.json` (with the SHA-256 of every cache file read).

## 2026-10-05: Owner report (patch-model training overlap) and cached-map check result

- **Owner report, from 1.60 recipe recovery; not independently verified here.** The legacy patch models had 77 of the 84 eval cases in their training split.
  - Added to LEAKAGE_FINDING.md (Finding 3), PRIOR_DEFECTS #2 and DECISIONS D5.2. Commit `fed8e12`.
  - Every 29-scan result is now labelled "clean with respect to nnU-Net and stacker training only; NOT clean for the patch models".
- **Cached-map check: PASSED.**
  - It ran exactly as pre-committed (`dbfe176`): split all 84, labels used only for the pre-declared aggregates, nothing selected.
  - All 84: mean Dice 0.7858 (Δ 0.0000), total FPs 141 (Δ 0), mean sensitivity 0.8095 (Δ −0.0005). Matched lesion Dice 0.7910; GT lesions 636.
  - Strata, reported only:

    | Stratum | Dice | Sensitivity | FPs |
    |---|---|---|---|
    | "clean" (29, nnU-Net/stacker-clean only) | 0.8049 | 0.8530 | 36 |
    | contaminated (55) | 0.7758 | 0.7866 | 105 |

  - The `stacking_cache_v5` files read (84) have their SHA-256 hashes in `results/cached_map_reproduction/provenance.json`. The `mask` key was never read.
  - Conclusion: the stacker is confirmed, and the gate failure lies entirely in today's base-map generation (REPRODUCTION_FAILURE.md, PRIOR_DEFECTS #8 updated).
- Stopped per instruction. Option 2 not started. No pipeline, adapter, models.yaml or weights.lock change.

## 2026-10-05: CLOSED OUT (superseded by JANNUS 1.60)

- **Owner decision:** retrain, not restore. JANNUS 1.60 is being rebuilt from scratch in `G:\BrainMetShare\1.60` with patient-level locked splits, a single input path and out-of-fold stacker training.
- This experiment was not completed. No detection result was produced on real data.
- Wrote CLOSEOUT.md: the findings, the cached-map result, and the modules ready to port.
- Marked README.md, PLAN.md and the REPORT template (`make_report.py`) as SUPERSEDED.
- This branch is tagged `audit/1.50-defects` as an audit record and is not merged into main.
