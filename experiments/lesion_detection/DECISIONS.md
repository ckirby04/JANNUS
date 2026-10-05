# Methodological decisions

Each entry gives the choice, the rationale, and the alternatives considered. **Status**
is one of *locked* (approved and implemented), *proposed* (in PLAN.md, not yet
implemented), or *pending* (waiting on the project owner). Do not edit locked entries
in place. To change one, add a new entry that supersedes it, and log the change in
EXPERIMENT_LOG.md.

Reference baseline throughout: **overall lesion sensitivity 0.810 at 1.68 FP/case**
(84-case cohort, threshold 0.55). The often-quoted 0.943 is sensitivity for **lesions
≥10 mm only** (RANO-BM measurable disease), and it is labelled that way wherever it
appears.

---

## D1 Connectivity: 26-connectivity, configurable. *Locked (Phase 1).*

- **Choice.** Components come from `scipy.ndimage.label` with a full 3×3×3 structuring element (26-connectivity). Config key: `matching.connectivity ∈ {6, 18, 26}`.
- **Rationale.** For detection, a lesion is one object. Voxels that touch only at a corner on a ~1 mm grid are almost always the same lesion, and splitting them would invent extra FPs or split hits.
- **Alternatives.**
  - 6-connectivity, the scipy default and what the existing `jannus.evaluation.metrics.match_lesions` uses. It produces more and smaller components, so counts under it are not comparable to ours. It is run once as a sensitivity analysis.
  - 18-connectivity is a compromise with no particular justification.
- **Consequence.** Lesion counts differ from prior JANNUS reports even before any other change.

## D2 Primary hit rule: overlap of ≥1 voxel. *Locked (Phase 1).*

`matching.match_by_overlap`.

- A predicted component **hits** a GT lesion if they share at least `min(min_overlap_vox, lesion_voxels)` voxels and at least `min_overlap_frac` of the lesion. The defaults are 1 voxel and 0 %, so any touch counts.
- **A GT lesion is detected** at score threshold τ if at least one surviving component (score ≥ τ) hits it.
- **A predicted component is an FP** if it hits no lesion. Under the overlap rule, a component that touches a lesion without meeting a stricter criterion is a **fragment**: it detects nothing and is **not** an FP.

Edge cases, each pinned by a unit test in `tests/unit/test_detection_matching.py`:

| Case | How it is counted | Test |
|---|---|---|
| **Split**: k components hit one lesion | The lesion is detected **once**. Its detection score is the **max** score over its hitting components, the standard one-hit-per-lesion FROC convention. The other k−1 components are neither TP nor FP; they are reported as `n_redundant`, and the lesion counts in `n_split_lesions`. | `test_split_counts_lesion_once_and_never_as_fp` |
| **Merge**: one component hits k lesions | **Each** of the k lesions is detected. The component contributes 0 FPs and is counted in `n_merge_predictions`. | `test_merge_detects_every_covered_lesion_with_no_fp` |
| **Tiny fragment**: a 1-voxel touch on a large lesion | Under the primary rule it **is** a hit. The best overlap is recorded per lesion as `gt_overlap_vox` and `gt_overlap_frac`, so fragment-only detections are visible. A supplementary strict rule (≥3 voxels capped at lesion size, and ≥10 % of the lesion) is reported alongside. Under it, the fragment detects nothing and is not an FP. | `test_tiny_fragment_is_hit_by_default_and_reported`, `test_strict_rule_fragment_neither_detects_nor_counts_as_fp`, `test_strict_voxel_criterion_capped_at_lesion_size` |
| Empty GT | Every component is an FP. Sensitivity is undefined for that scan, which is excluded from sensitivity denominators but included in FP/scan. | `test_empty_gt_all_predictions_are_fp` |
| Empty prediction | Every lesion is missed, with 0 FPs. | `test_empty_prediction_misses_everything` |
| Both empty | 0 lesions, 0 FPs. | `test_both_empty` |

- **Rationale.** CADe grading asks whether the reader was pointed to the lesion, and any overlap answers that. Treating split pieces as FPs would penalise good localisation. Treating them as extra TPs would inflate sensitivity beyond one per lesion. Crediting each lesion in a merge matches what a reader sees, since a flag sits on both lesions.
- **Alternatives.**
  - IoU ≥ 0.1, greedy one-to-one, as in the existing harness. It under-credits merges and splits and is harsh on small lesions.
  - Bounding-box IoU: rejected by the brief, and too harsh on small lesions. It is reported only as the supplementary column `gt_best_bbox_iou`.
  - Hungarian one-to-one assignment, which would turn a correct merge into a miss.
- **FP status does not depend on τ.** A component's FP status is fixed by geometry, so one match per (case, voxel threshold) supports the whole FROC sweep.

## D3 Secondary hit rule: centroid within D mm. *Locked (Phase 1).*

`matching.match_by_centroid`.

- **Rule.** A component hits every lesion whose centroid lies within D mm (inclusive) of the component's centroid. Distances are Euclidean in physical mm, using the true per-axis spacing (D4). D ∈ {3, 5, 10} is reported.
- **Edge cases.** Splits and merges are counted as in D2.
- **Known weakness.** A merged component's centroid can fall between two lesions and miss both, and it then counts as an FP even though it overlaps them. This is not patched; the rule is reported as specified (`test_centroid_rule_merge_can_miss_both_lesions`). An elongated or irregular lesion has the same problem.
- **Rationale.** This is a localisation criterion that ignores boundaries, and it is common in CADe literature.
- **Alternative.** Pointing-game style ("prediction centroid inside the GT mask"). It is strictly harsher on small lesions, so it was not chosen.

## D4 Geometry and lesion size. *Locked (Phase 1).*

- **True spacing.** `preprocessed_256` headers are wrong (SPACING_DEFECT.md). True spacing comes from the native header: `native_zoom × native_shape / 256` (`geometry.resampled_spacing`). This is approved in Q4.
- **Size.** Equivalent spherical diameter, `ESD = (6V/π)^(1/3)`, with `V = n_vox × Π spacing`.
- **Bins.** `<3, 3–5, 5–10, >10` mm, half-open `[lo, hi)`, validated to tile [0, ∞). Every lesion is binned by its **GT** size. FPs have no GT size and are characterised separately (D8).
- **Bridging.** Key tables are repeated with the old assumptions (1 mm isotropic, longest bounding-box axis) and labelled as not comparable.
- **Alternatives.**
  - Longest axis, the RANO-style measure used in prior reports. It is kept as a bridging column only. It is orientation-dependent and on these grids badly distorted.
  - Resampling predictions back to the native grid and evaluating there would be most faithful. But it re-interpolates probability maps the stacker never saw, so it is not used.

## D5 Splits and tuning protocol. *Locked: 2-fold fallback. Patient grouping and subset reporting pending (Q8).*

**Contamination check** (required by the Phase 0 approval):

- **Stacker internal-validation IDs were never logged.** The checkpoint stores only epoch 22 and val_loss 0.1789, and the training script prints counts only. A direct check against `splits_final.json` is therefore **impossible**.
- A rough reconstruction (last 10 % of the sorted non-eval pool) gives 49 cases, not the documented 59. So it is not the real list, and it cannot settle the question.
- **Decision.** Per the approval, use the fallback: a fixed-seed **2-fold split of the 84**. Tune on A and evaluate on B, then swap, and report both directions plus pooled.

**Additional findings** that change how the 84 must be described (see EXPERIMENT_LOG 2026-10-04):

1. **Scan level.** 24 of the 84 scans (8 BMS, 16 UCSF) are in nnU-Net Dataset001 **fold-0 training**. That fold trained nnU-Net 3D, nnU-Net 2D and SwinUNETR, so for these scans 3 of the 7 base models saw the GT.
2. **Patient level.** UCSF subject IDs ending A/B/C… are longitudinal scans of one patient. This is confirmed by `TableS1_UCSF_BrainMetastases_SubjectInfo.xlsx`: same sex and cancer type in all 101 multi-scan patients, and BraTS IDs `-000`, `-001`, …. The 84 scans are **77 patients**, and **33 of those patients have other scans in the stacker's training pool**. The 84-case cohort is therefore **not** disjoint from training at the patient level, contrary to MODEL_CARD.md.
3. **Clean subset.** Only **29 scans (28 patients)** have no patient-level overlap with fold-0 base training or the stacker pool. Even this is an upper bound, because two things cannot be checked from 1.50:
   - the training sets of `patch_8/12/24/36`, trained in 1.30;
   - the source patients of the 2,652 synthetic BMS derivatives (`BMS_CPINTER`, `BMS_CPINTRA`, `BMS_SYNTH`) in Dataset001.
4. The 84 were also previously used to choose the 0.55 threshold (Dice sweep) and the stacker variant. REPORT.md must say so, and must say that **this repo has no untouched test set**.

**Decisions for the split:**

- **Grouping.** Folds and bootstrap resamples are **patient-grouped**: all scans of a patient go to the same fold, and are resampled together. The brief requires patient-level CIs, so grouping by scan would be wrong.
- **Fold assignment.** Fixed seed, stratified by (any GT lesion with ESD < 5 mm) × (≥2 GT lesions). The aim is balanced small-lesion and multi-lesion content.
- **Pending (Q8).** Should results also be reported separately for the 29-scan clean subset and the 55 contaminated scans? The proposal is yes, as a stratum, never used for selection.

## D6 Selection objective. *Locked (Q3).*

- **Objective.** Plain average sensitivity at 0.5, 1, 2 and 3 FP/scan (CPM).
- **Tie-breaks.** First the higher voxel threshold, then the scorer listed first in config order.
- **Reporting.** Small-lesion results are reported separately and are not used for selection.

## D7 UCSF ground-truth file. *Proposed; awaiting the owner's confirmation (Q5).*

Candidates and how they differ (EXPERIMENT_LOG, 2026-10-04 Q5 entry):

| | File | Coverage | Labels | Relation |
|---|---|---|---|---|
| A | `preprocessed_256/UCSF_*/seg.nii.gz` | 461/461 | 0/1 | identical to zoom(B) on all 461 scans (Dice 1.0) |
| B | `Raw Data/.../<id>_seg.nii.gz` | 461/461 | 0/1 | original 2020 release |
| C | `Raw Data/.../<id>_BraTS-seg.nii.gz` | 324/461 (54/72 eval UCSF scans) | 1 NETC, 2 oedema, 3 enhancing | 2023 neuroradiologist revision; label 3 vs B: median Dice 0.961, lesion count differs in 91/324 scans |

- **Proposal.** Use **A** for every primary number. It is what every prior JANNUS figure used, and it covers every scan.
- **Why C is not the default.** C is a revision, but not a clean drop-in correction. It covers only 75 % of the eval UCSF scans, so the cohort would mix annotation protocols. It uses a different label scheme, and the release notes say label 3 "may be slightly different" from the original label 1.
- **Supplementary analysis.** Re-score the 54 eval-cohort scans that have C, using C label 3 as GT, at the frozen operating points. Lesion counts differ in about 28 % of scans, which matters more for detection than for Dice. This shows how sensitive the detection numbers are to annotation convention. It is reported as a supplementary analysis and never used for selection.

## D8 Reproduction gate tolerances. *Locked (Q7), with the limitation noted.*

- **Target.** Per-case Dice within 1e-3 and the logged aggregates matched.
- **Limitation.** Under the Q2 exception only **IDs** were copied, so no per-case Dice values are available to compare against. The only logged figures readable in 1.50 are the aggregates in docs: Dice 0.7858, sensitivity 0.810, 1.68 FP/case (= 141 FPs / 84).
- **Gate as implemented:**
  - mean Dice within ±1e-3 of 0.7858;
  - total FP count within ±2 of 141;
  - lesion sensitivity within ±0.005 of 0.810.
  - All three are computed exactly as the original script did: `postprocess_prediction(p > 0.55, 0)`, existing metrics, 1 mm spacing.
- **What a per-case check would need.** Permission to read the per-case production Dice, if it was ever logged. `stacking_v5_results_per_case.json` holds the *older* 6-model stacker's values, not the production one's.

## Observed, not acted on

`jannus.evaluation.metrics.compute_case_stratified` compares unmatched predictions against `label`, the imported scipy function, instead of `bin_label`. As a result **`n_fp_in_bin` is always 0**, and every prior "FPs/case" column in the stratified-by-size tables reads 0. The bug is not fixed on this branch (existing metrics are frozen). It should be fixed separately.
