# CLOSEOUT: lesion-detection experiment on JANNUS 1.50 (audit record)

**Status: SUPERSEDED (2026-10-05).** The experiment was not completed. Successor: **JANNUS 1.60**
(`G:\BrainMetShare\1.60`), rebuilt from scratch with patient-level locked splits, a single input
path and out-of-fold stacker training. This branch is preserved as an audit record under tag
`audit/1.50-defects`. It is not merged into main.

## Bottom line

The published 1.50 figures were **computed correctly**. Fed the cached base maps it was trained
on, the unchanged production stacker reproduces them exactly: Dice 0.7858, 141 FPs, sensitivity
0.8095 (the logged figure is 0.810), on all 84 scans. They are nonetheless **invalid as performance
estimates**, for three reasons:

- **(a)** the evaluation cohort overlapped training for **every** base-model family, and for the stacker;
- **(b)** the shipped pipeline produces **different** base maps and does not reproduce the figures (Dice 0.609);
- **(c)** every mm-based metric used **wrong voxel spacing**.

**Decision (owner, 2026-10-05): retrain, not restore.** The detection modules will be ported to 1.60.

## Findings (index: PRIOR_DEFECTS.md)

| # | Finding | Effect on published figures | Detail |
|---|---|---|---|
| 1 | `preprocessed_256` headers misstate spacing. BMS volumes carry an identity affine at 1 mm; UCSF volumes carry their native affine on a resized grid. Median true voxel volume is 0.515 mm³, and BMS_Mets_005's GT volume is overstated 2.0×. | Size bins (incl. <3 mm 0.061), the ≥10 mm-only scope (sensitivity 0.943, Dice 0.835, κ, SoLD), HD95 17.26 mm, MSD | SPACING_DEFECT.md |
| 2 | The "held-out, patient-disjoint" 84-scan cohort overlaps training. 24 scans are in nnU-Net/SwinUNETR fold-0 training. 33 of 77 patients have other timepoints in the stacker pool. **The legacy patch models trained on 77 of the 84** (owner, 1.60 recipe recovery). | Every figure | LEAKAGE_FINDING.md |
| 3 | The 0.55 threshold and the stacker variant were selected on the reporting cohort. | Every figure at 0.55 is in-sample | LEAKAGE_FINDING.md |
| 4 | The per-size-bin FP count is always 0, because it compares against the scipy `label` function (`metrics.py:425`). | Every stratified FP/case column | PRIOR_DEFECTS.md |
| 5 | INTENDED_USE says all 84 scans are "from UCSF"; 12 are Stanford BMS. | Cohort description | LEAKAGE_FINDING.md |
| 6 | 6-connectivity in lesion matching is undocumented. | Lesion and FP counts not comparable across conventions | DECISIONS.md D1 |
| 7 | `jannus predict` provenance omits the nnU-Net checkpoint hashes. | Cross-site same-weights check is incomplete | PRIOR_DEFECTS.md |
| 8 | The shipped pipeline does not reproduce the figures: 0.609 vs 0.786 Dice, 223 vs 141 FPs. Causes: nnU-Net runs on the resampled 256³ grid with spacing forced to 1 mm, reproducing exactly the calibration-broken `v5_256` variant rather than the `v5` "native" one the stacker learned; and the v3 patch weights replaced the legacy ones after stacker training. The stacker itself is confirmed correct. | Every figure vs every shipped output | REPRODUCTION_FAILURE.md |

## Cached-map reproduction (pre-committed, `dbfe176` → result `965c09a`)

| All 84 | Logged | Observed |
|---|---|---|
| Mean Dice | 0.7858 | 0.7858 |
| Total FPs | 141 | 141 |
| Sensitivity | 0.810 | 0.8095 |
| Matched lesion Dice | 0.791 | 0.7910 |

Strata are descriptive only. **"Clean" (29) means clean with respect to nnU-Net and stacker training
only; NOT clean for the patch models.**

| Stratum | Dice | Sensitivity | FPs |
|---|---|---|---|
| "Clean" (29) | 0.805 | 0.853 | 36 |
| Contaminated (55) | 0.776 | 0.787 | 105 |

## Ready to port to 1.60

All of these are pure numpy/scipy and import nothing from the 1.50 pipeline. Tests are synthetic,
with hand-derived expectations; all 48 pass, and ruff is clean.

| Module | Path | Tests | Status |
|---|---|---|---|
| geometry | `src/jannus/evaluation/detection/geometry.py` | 4 (in `tests/unit/test_detection_matching.py`) | Ready. True spacing from native headers, ESD, contiguous size bins. |
| matching | `src/jannus/evaluation/detection/matching.py` | 20 (`tests/unit/test_detection_matching.py`; file total 24) | Ready. Overlap, strict and centroid rules; merge/split/fragment; anisotropic; empty cases. |
| scoring | `src/jannus/evaluation/detection/scoring.py` | 11 (`tests/unit/test_detection_scoring.py`) | Ready. Assumes 7 base maps + stacker map, so 1.60 must re-check the base-model list. |
| metrics | `src/jannus/evaluation/detection/metrics.py` | 10 (`tests/unit/test_detection_metrics.py`) | Ready. FROC, fixed-FP step readout, strata, scan-level rates, case weights. |
| bootstrap | `src/jannus/evaluation/detection/bootstrap.py` | 2 (same file) | Ready. Patient-level, via case weights. |
| provenance | `src/jannus/evaluation/detection/provenance.py` | 1 (same file) | Port with changes: hard-codes the 1.50 base tag and commit. |
| report (plots, tables) | `src/jannus/evaluation/detection/report.py` | 0 | Port with tests. CVD-validated palette; never run on real results. |
| report generator | `experiments/lesion_detection/make_report.py` | 0 | Port and adapt. Never run on real results; its caveat text is 1.50-specific. |
| drivers | `experiments/lesion_detection/{tables,build_components,select_operating_point,evaluate,pool_2fold,segmentation_secondary,brats_side_analysis,common}.py` | 0 | Reference designs. `build_components` ran on 1 case; the rest never ran on real data. Run-once lock and split-disjointness checks worth keeping. |

Not to port: `run_inference.py`, `repro_gate.py`, `cached_map_reproduction.py` and `tools/diagnose_*`.
They are specific to the 1.50 pipeline and caches.

## What was never produced

No FROC, operating point or detection result was computed on real data. No result on the clean
split was ever looked at. Nothing from this branch supports any performance claim.
