# Training overlap in the "84-case held-out" cohort

Found 2026-10-04 during setup of the lesion-detection experiment. **This branch does not
edit any existing document.** MODEL_CARD.md, INTENDED_USE.md, PRODUCTION.md and README.md
all still carry the claims quoted below. Correcting them is a separate decision for the
project owner.

Reproduce with:

```bash
python experiments/lesion_detection/tools/build_leakage_splits.py --data-root G:/BrainMetShare/Data
```

That command writes `configs/splits/eval84_leakage.csv` (one row per scan with every
overlap flag), `clean_ids.txt` and `contaminated_ids.txt`. Inputs are ID lists only; see
`configs/splits/SOURCES.md`.

## The claim

`docs/MODEL_CARD.md:70`:

> 84-case held-out cohort, disjoint from training at the patient level.

Related claims elsewhere:

- `docs/INTENDED_USE.md:133`: "Held-out evaluation | 84 cases (subject-level split)".
- `docs/INTENDED_USE.md:101`: "84 held-out cases from UCSF Brain Metastases". In fact 12 of the 84 are Stanford BrainMetShare (`BMS_Mets_*`) scans.
- `README.md:70`, `docs/research/PRODUCTION.md:21`, `configs/models.yaml:18` and `src/jannus/segmentation/pipeline.py:38` each call the cohort "held-out".

## Finding 1: scan-level overlap with base-model training (fold 0)

nnU-Net Dataset001 `splits_final.json` fold 0 is the split behind three of the seven base
models:

- **nnU-Net 3D** and **nnU-Net 2D**: `configs/models.yaml` sets `folds: [0]`.
- **SwinUNETR**: `scripts/training/train_swin_unetr.py` uses `load_nnunet_split(..., fold=0)`.

**24 of the 84 eval scans (22 patients) are themselves in fold-0 *training*.** That is 8 of
the 12 BMS scans and 16 of the 72 UCSF scans. For these scans, three of the seven inputs to
the stacker came from networks that were trained on that exact image and its label.
Another 13 eval scans are in fold-0 *validation*, where nnU-Net's checkpoint selection may
have seen them but no gradient step did.

## Finding 2: patient-level overlap via UCSF timepoints

UCSF subject IDs end in a letter that marks the **timepoint**. `100238A` through `100238G`
are seven scans of one patient. This is confirmed by
`Raw Data/UCSF_BrainMetastases/TableS1_UCSF_BrainMetastases_SubjectInfo.xlsx`. In all 101
multi-scan patients, sex and cancer type agree across scans, and the BraTS IDs share a
patient number with suffixes `-000`, `-001`, …. A patient-level split therefore has to
group by the ID with the letter removed. The original split grouped by full scan ID.

- The 84 eval scans are **77 patients**; six patients have 2–3 scans inside the cohort.
- **33 of the 77 eval patients (38 scans) have other timepoints in the stacker's training pool.** The stacker pool is every development scan not in the 84; `seven_model_stacker.py` splits it 90/10 into train and internal-val.
- **27 eval patients (30 scans) have some timepoint in fold-0 base-model training.** That is Finding 1 widened to the patient level.
- 11 patients (13 scans) fall under both.

## Finding 3: the legacy patch models trained on 77 of the 84 eval cases

*Reported by the project owner on 2026-10-05, from recipe recovery in the 1.60 workspace. This
experiment cannot see that workspace and has not verified the finding independently.*

**The legacy patch models (`patch_8/12/24/36`) had 77 of the 84 eval cases in their training split.**
These are the models whose outputs fill `stacking_cache_v5` and on which the production stacker was
trained. Consequences:

- At most 7 of the 84 scans are outside patch-model training. Which 7 is not recorded here.
- At least 22 of the 29 "clean" scans were therefore patch-model training cases.
- Four of the seven base models feeding the stacker had seen most of the cohort, labels included.
- The training split of the **v3** patch weights loaded by today's pipeline is not covered by this report.

## The clean subset, and why it is only an upper bound

> **Required label (owner, 2026-10-05).** Every result on the 29-scan subset must be labelled:
> **"clean with respect to nnU-Net and stacker training only; NOT clean for the patch models."**
> The file name `clean_ids.txt` and the short word "clean" are kept for continuity, and always
> mean this restricted sense.


**29 scans / 28 patients** have no timepoint in fold-0 training and none in the stacker
pool. They are listed in `configs/splits/clean_ids.txt`: 4 BMS and 25 UCSF. The remaining
**55 scans / 49 patients** are in `contaminated_ids.txt`.

"Clean" means clean *with respect to nnU-Net and stacker training only*. Two training sources
could not be checked from 1.50 when this was written:

1. **The patch models** (`patch_8/12/24/36`). The legacy generation, whose outputs the stacker was trained on, is now known to be **contaminated**: 77 of the 84 eval cases were in its training split (Finding 3). For the v3 generation loaded today (trained in 1.30 by `train_native_v3.py`), the training case list is still not in 1.50.
2. **The synthetic BMS cases in Dataset001**: 1,560 `BMS_CPINTER_*`, 892 `BMS_CPINTRA_*` and 200 `BMS_SYNTH_*`. The names suggest copy-paste and synthesis augmentation built from real BMS patients. If any were built from an eval BMS patient, that patient's lesions were in fold-0 training. This matters most for the 4 "clean" BMS scans.

The project owner will supply what is known about both. Until then, every result on the
clean subset is labelled an upper bound on cleanliness.

## Which prior metrics are affected

Every figure computed on the 84-case cohort, which is every performance number the repo
publishes:

| Figure | Where | Effect |
|---|---|---|
| Voxel Dice 0.7858; lesion sensitivity 0.810 at 1.68 FP/case | MODEL_CARD, README, PRODUCTION, models.yaml, pipeline.py docstring | Likely optimistic. For 24 scans, three base models were trained on the image itself, and 33 patients were seen by the stacker at another timepoint. The size of the effect is unknown and is not estimated here. |
| Measurable-disease (≥10 mm lesions only) sensitivity 0.943, matched Dice 0.835, κ 0.867, SoLD error 1.84 % | MODEL_CARD, INTENDED_USE, PRODUCTION | Same leakage, *and* the ≥10 mm scope is wrong (SPACING_DEFECT.md). |
| Size-stratified sensitivity (<3 mm: 0.061 … >20 mm: 0.986) | MODEL_CARD, CROSS_VALIDATION | Same leakage, plus the spacing defect and the per-bin FP bug (PRIOR_DEFECTS.md). |
| HD95 17.26 mm, MSD | MODEL_CARD, CROSS_VALIDATION | Same leakage, plus the spacing defect. |
| Threshold 0.55 | MODEL_CARD, models.yaml | Chosen by sweeping Dice on these same 84 scans, so it is tuned on the reporting cohort independently of the leakage. |
| Stacker variant selection (v4 vs v5, 6- vs 7-model, hard-negative) | `docs/research/stacking_v4_vs_v5_writeup.md` | Model selection on the reporting cohort. |
| External-site comparisons in CROSS_VALIDATION.md ("compare against MODEL_CARD") | CROSS_VALIDATION | Sites compare honest numbers against a reference that is probably optimistic. |

## How this experiment handles it

Approved by the owner; see DECISIONS.md D5.

- **Primary.** The operating point is chosen on the 55 contaminated scans only, then evaluated **once** on the 29 clean scans (28 patients). The full clean-set FROC is shown across all thresholds, because a threshold tuned on leaked cases may be miscalibrated for clean data.
- **Secondary.** A patient-grouped 2-fold split over all 84, reported as a sensitivity analysis.
- **Confidence intervals.** All bootstrap CIs resample patients.
- **Small-n caveat.** Clean-set per-size-bin lesion counts are printed beside every number, and the bins are described as underpowered.
