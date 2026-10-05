> **STATUS: SUPERSEDED (2026-10-05).** This experiment was not completed. JANNUS 1.50's published figures are invalid as performance estimates (see CLOSEOUT.md, tag `audit/1.50-defects`). Successor: JANNUS 1.60 (`G:\BrainMetShare\1.60`).

# Lesion-level detection evaluation: Phase 0 plan

Status: **draft, awaiting approval.** Nothing beyond this file has been written.
Branch: `exp/lesion-detection-eval`, created from `main` @ `d43dc97` (see Q1).
Date: 2026-10-04

---

## 1. Recon findings

### 1.1 How test-set inference runs today

There are two separate paths, and the published numbers come from the second one.

| Path | Entry point | What it saves |
|---|---|---|
| Production CLI | `jannus predict`: `cli/commands.py:218` → `BrainMetPipeline.predict_volume` (`segmentation/pipeline.py`) | `{case}_seg.nii.gz` = `prob >= threshold`, with no morphology (`commands.py:322`). With `--save-probabilities` it also writes `{case}_prob.npy`, the **stacker map only**, as float16. Base-model maps are never saved. |
| Research evals that produced the 84-case figures | `scripts/evaluation/seven_model_stacker.py`, `eval_stratified_by_size.py`, `eval_measurable_disease.py` | They read cached base predictions from `model/stacking_cache_v5/` + `model/stacking_cache_v5_256/`, run the stacker in memory, and save only JSON metrics. |

- **No per-voxel probability maps exist in 1.50.** The `model/` directory holds only the six `.pth` files. The `stacking_cache_v5*` caches and `model/evaluation_results/*.json` lived in older trees (1.30/1.40), which are off limits.
- **The clean way to regenerate them** needs no change to existing code. `BrainMetPipeline.predict_volume` already returns `PipelineResult.probability_map` (stacker) **and** `PipelineResult.base_probs` (all 7 base maps). A wrapper script, `experiments/lesion_detection/run_inference.py`, will build the pipeline with `BrainMetPipeline.from_config(..., verify=True)`, load each case through `jannus.data.loading.load_case`, and write both map types to `cache/`. That script is the only new inference code.

### 1.2 Dataset splits

| Pool | Location | n | GT? | Notes |
|---|---|---|---|---|
| Stanford BrainMetShare "test" | `Data/preprocessed_256/test`, `Data/test` | 51 | **No** `seg.nii.gz` | Unusable for this experiment. |
| Development pool | `Data/preprocessed_256/train` | 566 (105 `BMS_Mets_*`, 461 `UCSF_*`) | Yes | Every documented metric comes from this pool. |
| "84-case held-out eval" | IDs = keys of `model/evaluation_results/stacking_v5_results_per_case.json` | 84 | Yes | **The ID list is not in 1.50.** It exists only in a frozen tree. |
| Stacker internal-val | last 10% of the sorted non-eval IDs that had valid caches (`seven_model_stacker.py:200`) | ~59 | Yes | Depends on which cache files existed at training time, so it **cannot be reconstructed** from 1.50. |
| Base-model training | nnU-Net `splits_final.json` fold 0 on the expanded 2,917-case Dataset001 (also used by SwinUNETR) | 2,333 train / ? val | n/a | **`splits_final.json` is not in 1.50.** |

Three facts about these splits change the methodology.

1. **The 84-case set is a development cohort, not a pristine test set.** `seven_model_stacker.py:evaluate()` sweeps the voxel threshold over 0.05–0.95 on those 84 cases and keeps the Dice-optimal value. That is where 0.55 came from. The same 84 cases were also used to choose between stacker variants (`docs/research/stacking_v4_vs_v5_writeup.md`). `PRODUCTION.md` itself calls it the "dev (84)" cohort, separate from a "pivotal (70)" cohort.
2. **Base-model exposure is unknown.** If the stacker's internal-val cases were in nnU-Net/SwinUNETR fold-0 *training*, the base predictions on them are in-sample and over-confident. They would then be a poor place to choose a detection operating point.
3. **The `preprocessed_256` headers misstate geometry.** Every volume has an identity affine and 1.0 mm pixdim. They were made by `scipy.ndimage.zoom` from native grids such as 256×256×150 @ 0.938×0.938×1.0 mm (BMS) and 256×256×112 @ 0.859×0.859×1.5 mm (UCSF sample). For `BMS_Mets_005`, GT volume computed from the 256³ header is **2.0×** the native physical volume. All earlier mm-based numbers, including the size bins behind "sub-3 mm sensitivity 0.061", treated these grids as 1 mm isotropic. Native headers exist for all 566 cases (`Data/train/Mets_*`, `Raw Data/UCSF_BrainMetastases/train/*`). The true per-axis spacing of each 256³ volume is therefore recoverable as `native_zoom × native_shape / 256`. That is what this experiment will use (D4).

### 1.3 Per-model predictions

Yes, these are available after regeneration. `PipelineResult.base_probs` gives all 7 maps in `STACKING_MODEL_NAMES` order: `nnunet_3d, nnunet_2d, patch_8, patch_12, patch_24, patch_36, swin_unetr`. Agreement count and inter-model variance can be computed per component. The stacker's own 9-channel input (7 preds + variance + range) is rebuilt by `build_stacking_features_from_preds`, and the scoring code will reuse that function.

### 1.4 Baseline environment state (recorded before any change)

- `pytest`: **238 passed** (53 s).
- `ruff check .`: **8 pre-existing errors, all in `run_demo.py`** (F401/I001). I will not touch that file. "Ruff clean" will mean: no new errors, and `ruff check src/jannus/evaluation/detection experiments/lesion_detection tests/unit/test_detection_*.py` is clean.
- Python 3.13.15, torch 2.10 + CUDA (RTX 3070 Ti, 8 GB), numpy 2.3.4, scipy 1.16.3, scikit-image 0.25.2, nibabel 5.3.3, matplotlib 3.10.8 (a declared `demo` extra), pandas 2.3.3, PyYAML. **No new dependencies are needed.**
- Free disk on G: 606 GB.

### 1.5 A discrepancy in the task brief

The brief gives "overall lesion sensitivity ~0.943 with ~1.61 FP/case". In the repo docs, **0.943 is the measurable-disease (≥10 mm) sensitivity**. Overall all-lesion sensitivity is **0.810 at 1.68 FP/case**, and 1.61 FP/case is the row with a 100-voxel minimum-size filter (`stacking_v4_vs_v5_writeup.md:375`). The report will use the repo figures as the reference and state which scope each number belongs to.

---

## 2. File layout

```
src/jannus/evaluation/detection/          # new subpackage; imports existing code, modifies none
    __init__.py
    geometry.py      # true spacing for 256^3 grids from native headers; ESD; centroids in mm
    matching.py      # CC labelling, overlap & centroid rules, merge/split/fragment bookkeeping
    scoring.py       # pluggable per-component scores, registry keyed by config name
    metrics.py       # FROC, fixed-FP operating points, size strata, patient-level, FP characterisation
    bootstrap.py     # patient-level resampling of whole FROC curves
    provenance.py    # thin wrapper over jannus.core.provenance / checksums
    report.py        # REPORT.md + plots (matplotlib, Agg backend)

experiments/lesion_detection/
    PLAN.md  README.md  EXPERIMENT_LOG.md  DECISIONS.md
    run_inference.py         # wrapper: BrainMetPipeline -> cache/ (stacker + 7 base maps)
    build_components.py      # cache maps -> per-threshold component tables (parquet/CSV)
    run_eval.py              # component tables + config -> results/<run_id>/
    select_operating_point.py# tuning split only -> writes a frozen operating-point YAML
    configs/
        splits/              # ID lists only (tune.txt, test.txt), with source + sha256 (see Q2)
        geometry.yaml        # case-id -> native file mapping rule
        00_repro_gate.yaml   # baseline reproduction check
        10_tune_sweep.yaml   # tuning split: all voxel thresholds x all scores
        20_test_baseline.yaml# test: 0.55, no per-lesion filtering
        21_test_selected.yaml# test: frozen choice from 10_*
        22_test_filtering.yaml # optional min-size/min-score filters (flag)
    cache/                   # gitignored
        maps/{case}.npz      # stacker fp16 + 7 base uint8 (p*255), compressed
        components/{config_hash}/{case}.parquet
    results/<run_id>/        # metrics.json, froc_*.csv, *.png, REPORT.md, provenance.json

tests/unit/test_detection_matching.py, test_detection_scoring.py, test_detection_metrics.py
```

`.gitignore` gets one added line, `experiments/lesion_detection/cache/`. That edit touches a repo-level file, so I'm flagging it here (Q6).

## 3. Data flow

```
Data/preprocessed_256/train/{case}/  (read-only)
   └─ run_inference.py ── BrainMetPipeline.predict_volume ──► cache/maps/{case}.npz
                                                               stacker  float16 256^3
                                                               base[7]  uint8   256^3
native headers (read-only) ── geometry.py ──► true spacing per case (logged per case)
cache/maps + GT ── build_components.py ──► per (case, voxel_thr): component table
       columns: comp_id, n_vox, volume_mm3, esd_mm, longest_axis_mm, centroid_mm,
                score_{max,mean,agree,invvar}, overlapping GT ids, overlap voxels,
                dist_to_nearest_GT_mm (surface), bbox_iou_best (supplementary)
     + GT table: lesion_id, esd_mm, size bin, centroid_mm
component tables ── run_eval.py ──► FROC / operating points / strata / bootstrap ──► results/
```

Postprocessing is defined the way `jannus predict` writes masks: `prob >= thr`, with no opening, closing or size filter. (`PipelineResult.binary_mask` applies opening/closing, but the CLI ignores it.)

## 4. Methodological proposals (go to DECISIONS.md once approved)

**D1 Connectivity.** 26-connectivity (`scipy.ndimage.label` with `generate_binary_structure(3,3)`), configurable. The existing `evaluation/metrics.py` uses scipy's default **6-connectivity**. That is part of why our counts will differ from the published ones, and the report will say so. 6-connectivity will be run once as a sensitivity analysis.

**D2 Primary hit rule (overlap ≥1 voxel).**
- A GT lesion is *detected* at a score threshold τ if any surviving predicted component overlaps it by ≥1 voxel.
- A predicted component is a *TP-component* if it overlaps ≥1 GT lesion, otherwise it is an *FP*.
- **Split** (k components on one GT lesion): the GT lesion counts once. Its FROC score is the max score among the overlapping components, which is the standard one-hit-per-lesion convention. The extra components are neither TPs nor FPs; they are counted in a separate `redundant` column.
- **Merge** (one component on k GT lesions): each covered GT lesion counts as detected. The component counts as one non-FP.
- **Tiny fragment**: a 1-voxel overlap counts as a hit, because that is the rule. Each hit stores `overlap_vox` and `overlap_frac_of_gt`. A supplementary sensitivity also reports a stricter rule (overlap ≥10% of the GT volume, or ≥ min(3 vox, GT size)), so fragment-only detections are visible.

**D3 Secondary rule (centroid distance ≤ D mm, D ∈ {3, 5, 10}).** Distances are measured in physical mm. A GT lesion is detected if any component centroid lies within D of the GT centroid. A component is an FP if its centroid lies within D of no GT centroid. Splits follow the same once-per-lesion rule as D2. A merged component's centroid can fall between two lesions and miss both. That is reported as a known weakness of the rule, not patched.

**D4 Geometry and size.** Spacing per axis = `native_zoom × native_shape / 256`, taken from the native header. The 256³ header is not used. **Size = equivalent spherical diameter**, `ESD = (6V/π)^(1/3)` with `V = n_vox × Πspacing`. Bins: `<3, 3–5, 5–10, >10 mm`, half-open `[lo, hi)`. Bins are always assigned by the **GT** lesion's size. ESD is smaller than the longest-axis measure behind the 0.061 figure, so a longest-axis column is also reported for continuity. It is labelled as not comparable.

**D5 Scores.** Each score is computed per component and maps to [0, 1] with higher = more confident. They are registered by name in `scoring.SCORERS`.
- `max_prob`: max stacker probability in the component.
- `mean_prob`: mean stacker probability.
- `agreement`: count of base models whose map has ≥1 voxel ≥0.5 inside the component, divided by 7. The base threshold of 0.5 matches `configs/models.yaml` and is configurable. Ties are frequent (8 levels), so its FROC has at most 8 points per voxel threshold. **There is no hidden tie-break.**
- `inv_variance`: `1 − mean_voxel_var / 0.25`, where 0.25 is the maximum variance of values in [0,1]. Caveat for the report: variance is also low when every base model is confidently *negative*. A component the stacker alone switches on can score as "confident". `agreement` is the cross-check.

**D6 FROC.**
- x = mean FP components per scan, with all scans in the denominator, including any scans with no lesions.
- y = pooled lesion sensitivity (detected GT lesions / total GT lesions).
- The curve sweeps τ over every unique component score, separately for each voxel threshold in `0.20, 0.25, …, 0.70`.
- **Sensitivity at a fixed FP rate** (0.5, 1, 2, 3 FP/scan) is the sensitivity at the most permissive τ whose FP/scan ≤ target. This is a step function with no interpolation and is conservative. If the curve never reaches a target, the result is "not reached", never extrapolated.
- **Summary scalar** for selection: CPM = mean sensitivity over {0.5, 1, 2, 3} FP/scan.

**D7 Patient-level metrics.** Computed among scans with ≥1 GT lesion, at each reported operating point: the fraction of scans with ≥1 lesion flagged, and the fraction with all lesions flagged. The number of lesion-free scans is reported, with their FP/scan.

**D8 FP characterisation.** For each FP component: ESD, and surface-to-surface distance to the nearest GT lesion (EDT with true spacing).
- The FP distance histogram is reported with bands of 0–5, 5–20, 20–50 and >50 mm.
- **Direct HD95 test**: recompute HD95 with existing `jannus.evaluation.metrics.hd95` after deleting FP components farther than X mm (X = 10, 20) from any GT. If HD95 falls most of the way toward the matched-only value, the "isolated far FPs" explanation is confirmed. If not, it is refuted.

**D9 Confidence intervals.** Patient-level bootstrap: 2000 resamples of cases with replacement, seeded and recorded. Each resample recomputes the whole FROC and every derived number, with pooled sensitivity recomputed from the resampled lesion counts. CIs are percentile 95%. Every sensitivity is printed as `0.xx [lo, hi] (k/n lesions, N scans)`.

**D10 Secondary segmentation metrics.** Voxel Dice and HD95 for each reported config come from the existing `jannus.evaluation.metrics` functions, imported unchanged and called with true spacing. Because spacing is corrected, HD95 will **not** equal the published 17.26 mm. The 1-mm-assumption value is also computed for that config, to bridge to prior numbers.

**D11 Operating-point selection.** This rule is pre-declared and runs on the tuning split only.
1. Choose `(voxel_threshold, scorer)` by maximum CPM.
2. Tie-break: higher sub-5 mm sensitivity at 1 FP/scan.
3. Further tie-break: higher voxel threshold.

`select_operating_point.py` writes the result to `configs/21_test_selected.yaml`, with the tuning run_id and sha256, and it refuses to read any test ID. `run_eval.py` checks every case ID against the split file named in its config and refuses to mix splits. The test configs run once. Every run appends an entry to EXPERIMENT_LOG.md naming its split. (Q3 asks whether the objective should weight small lesions more heavily.)

**D12 Optional filtering**, behind the flag `filtering.enabled`: drop components with ESD < s mm or score < q. The effect is reported as ΔFP/scan vs Δ sensitivity in the <3 and 3–5 mm bins. s and q are chosen on the tuning split.

**D13 Bounding-box IoU** appears only as a supplementary `bbox_iou_best` column. It is never a hit criterion.

## 5. Splits proposal (blocked on the user, see Q2)

- **Option A (preferred if valid).** *Tune* on the stacker's internal-val cases. *Test* on the 84-case cohort. This needs the two ID lists, plus confirmation that none of the tuning cases were in base-model fold-0 training (from `splits_final.json`).
- **Option B (fallback).** Use this if the tuning cases are in-sample for any base model, or the lists can't be recovered. Do patient-level 2-fold cross-fitting **within the 84**, stratified by lesion count and presence of sub-5 mm lesions with a fixed seed: select on fold A and evaluate on B, select on B and evaluate on A, and report pooled out-of-fold results. This is not a sequestered test either, and the report will say so plainly.

With either option the report states two things: the 84-case cohort already informed the choice of 0.55 and the choice of stacker, and no pristine test cohort exists in this repo. If the 70-case "pivotal" cohort mentioned in PRODUCTION.md exists with GT, it would be the right single-use test set.

## 6. Reproduction gate (before any detection analysis)

Regenerate maps for the test cohort. At `prob >= 0.55`, using the **existing** metrics with the **old** 1 mm assumption and 6-connectivity, compare against the published numbers: Dice 0.7858, lesion sensitivity 0.810, FP/case 1.68. Pre-declared tolerance: |ΔDice| ≤ 0.01, |Δsens| ≤ 0.02, |ΔFP| ≤ 0.15. **If the run falls outside tolerance, I stop and report, with no debugging against test labels.** Debugging would go through the tuning split instead. This step only checks that regenerated maps match the pipeline that produced the published numbers. It tunes nothing, and the log will say so.

Known risk: the published cache ran nnU-Net "native-preprocessed at 256³ legacy format", while today's adapter runs on whatever grid it is given. The gate exists to catch exactly that.

## 7. Compute and disk estimates

| Item | Estimate |
|---|---|
| Inference | 3–8 min/case (per docs) on the 3070 Ti. Option A ≈ 84 + ~59 = ~143 cases → **~7–19 h** GPU, resumable per case. Option B = 84 cases → ~4–11 h. |
| Map cache | stacker fp16 32 MiB + 7 × uint8 16 MiB = 144 MiB raw/case. Compressed npz is likely 20–50 MiB (to be measured on the first case). **~3–7 GB compressed, ≤21 GB worst case.** |
| Component tables | 11 voxel thresholds × cases; CC labelling + EDT ≈ 2–5 s per (case, thr) → ~1–2 h CPU once, then cached. Tables are small (MB). |
| Bootstrap | 2000 × FROC recompute on tables only: minutes. |
| Determinism | Inference runs once and is cached, with the map sha256s recorded. "Re-run from scratch reproduces identical metrics" will be verified from the cache stage onward, plus one case re-inferred to check the map hash. GPU nondeterminism is logged via `jannus.core.determinism` if present. |

## 8. Provenance (`results/<run_id>/provenance.json`)

- git commit + dirty flag (`jannus.core.provenance._git_revision`)
- sha256 of the canonicalised config YAML (`config_hash`, also the run_id suffix)
- `jannus.core.checksums.hash_loaded_checkpoints` over the paths actually loaded, cross-checked against `weights.lock.json` (read-only)
- `PIPELINE_REVISION`
- split name + split file sha256 + the full case ID list
- sha256 of every cache map file consumed
- package versions, hardware
- UTC timestamp

## 9. Proposed CLI exposure (not implemented in this task)

`jannus detect-eval --config experiments/lesion_detection/configs/<x>.yaml` would be a thin dispatcher to `run_eval.py`, writing to a separate output directory with no change to `evaluate`. To be decided after the results are in.

## 10. Open questions

- **Q1 Base commit.** There is **no `v1.50` tag** in this repo. I branched from `main` HEAD `d43dc97`, which has `__version__ = 1.50.0` and includes the post-release fixes (einops, trainer sidecars in weights.lock) needed to run the pipeline. Is that acceptable? The alternative is `e4c1cee`, the "JANNUS v1.50" commit.
- **Q2 Split IDs.** I'm not allowed to read 1.30/1.40/`1.50_internal_docs`, which is where the ID lists are. Could you copy **ID lists only** into `experiments/lesion_detection/configs/splits/`? Needed: (a) the 84 keys of `stacking_v5_results_per_case.json`, (b) the stacker internal-val IDs, if a log of them exists, and (c) `splits_final.json` for Dataset001. Alternatively, grant a one-off read of those specific files. Without (a), not even the reproduction gate can run.
- **Q3 Selection objective.** Plain CPM over {0.5, 1, 2, 3} FP/scan (D11), or a size-weighted variant, such as CPM computed on lesions <5 mm, given that small lesions are the stated priority? I recommend plain CPM as primary, with small-lesion CPM reported but not used for selection. That keeps the selection from over-fitting the bins with the smallest n.
- **Q4 Geometry correction.** Do you agree that spacing should be corrected from native headers (D4), even though this makes our size bins and HD95 incomparable to every prior JANNUS number? Bridging columns at the old 1 mm assumption are included either way.
- **Q5 UCSF ground truth.** Raw UCSF cases ship both `*_seg.nii.gz` and `*_BraTS-seg.nii.gz`. I'll use the `preprocessed_256` `seg.nii.gz`, the same file the 84-case numbers used, and only read native headers for geometry. Please confirm.
- **Q6 `.gitignore`.** May I append `experiments/lesion_detection/cache/` to the root `.gitignore`? The alternative is a nested `experiments/lesion_detection/.gitignore`, which touches no existing file. I'd default to the nested one.
- **Q7 Tolerance for the reproduction gate.** Are the §6 tolerances acceptable?
