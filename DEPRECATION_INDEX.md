# DEPRECATION_INDEX: JANNUS 1.50 published figures and claims

Generated 2026-10-05 on branch `docs/deprecation-notice`. Details: tag `audit/1.50-defects`,
`experiments/lesion_detection/CLOSEOUT.md` and `PRIOR_DEFECTS.md`. **Line numbers refer to the files as of
v1.50.0 (`d43dc97`), before banners were inserted.** A banner adds 6 lines at the top of a doc.

**Searched** (all git-tracked text files): the requested figures and terms `0.786/0.7858, 0.810, 0.814,
0.835, 0.791, 0.943, 141, 1.61, 1.68, 17.26, 3.89, 0.061, held-out, held out, disjoint, external`; the other
headline metrics found on the same tables `18.36, 3.53, 0.806, 0.867, 1.84 %, 0.986, 0.903, 0.773, 0.439,
78.2/78.6/80.6/40.3 %, 94.3 %, 0.951, 0.7778, 0.758, 1.26–2.12, subject-level split`; and claims of
predicate-competitiveness without a listed number (added by hand, marked `(manual)`).

**Classes:**

- **CLAIM:** a published performance figure or cohort claim. An inline marker was added next to it, and the original text is unchanged.
- **RUNTIME-UNMARKED:** a claim inside a string the program displays. It is **not** marked, because the only code change allowed is the CLI warning. Follow-up needed.
- **UNAUDITED:** figures of earlier model versions that this audit did not examine. The doc gets a banner, but there are no inline markers. See `README_STATUS.md` (one level above the repo, `G:\BrainMetShare\README_STATUS.md`).
- **INCIDENTAL:** the term means something else, e.g. *external* sites or folders, per-fold *held-out* splits, a test named `disjoint`, or an unrelated number. No action.

**PRIOR_DEFECTS key:**

| # | Defect |
|---|---|
| #1 | `preprocessed_256` voxel spacing wrong, so mm-based metrics and size bins are wrong |
| #2 | evaluation cohort overlaps training for every base-model family and the stacker; not "held-out" or "disjoint" |
| #3 | threshold 0.55 and stacker variant chosen on the reporting cohort |
| #5 | cohort composition misdescribed (it is BMS + UCSF, not UCSF only or single-institution) |
| #8 | shipped pipeline does not reproduce the figures |

**Bannered docs (15):** `CHANGELOG.md`, `README.md`, `docs/CROSS_VALIDATION.md`, `docs/INTENDED_USE.md`, `docs/MODEL_CARD.md`, `docs/PHI_AND_DEIDENTIFICATION.md`, `docs/TROUBLESHOOTING.md`, `docs/research/BrainMetScan_improvements.md`, `docs/research/PRODUCTION.md`, `docs/research/ROADMAP_FDA_95_SENSITIVITY.md`, `docs/research/atlas_experiment_postmortem.md`, `docs/research/external_validation_report.md`, `docs/research/stacking_architectures_explored.md`, `docs/research/stacking_v4_vs_v5_writeup.md`, `docs/risk_analysis.md`.

**Not bannered** (incidental hits only, no performance figures): `CONTRIBUTING.md`, `docs/research/dead_configs/README.md`.

## CLAIM (184)

| file:line | defects | text (original) |
|---|---|---|
| `CHANGELOG.md:62` | #1, #2, #3, #8 | published figure — voxel Dice 0.7858, measurable-disease sensitivity 0.943 — was |
| `CHANGELOG.md:238` | #1, #2, #3, #8 | - Sub-3 mm sensitivity (0.061) is at the floor of MRI resolution and annotator agreement. |
| `README.md:70` | #2 | ## Performance (internal 84-case held-out cohort) |
| `README.md:79` | #2, #3, #8 | \| Voxel Dice \| 0.7858 \| |
| `README.md:80` | #2, #3, #8 | \| Lesion-wise sensitivity \| 0.810 (95% CI 0.758–0.858) \| |
| `README.md:81` | #2, #3, #8 | \| Lesion-wise Dice (matched) \| 0.791 \| |
| `README.md:82` | #2, #3, #8 | \| False positives / case \| 1.68 (95% CI 1.26–2.12) \| |
| `README.md:83` | #1, #2, #3, #8 | \| HD95 \| 18.36 mm (95% CI 12.5–25.0) \| |
| `README.md:91` | #1, #2, #3, #8 | \| Lesion-wise sensitivity \| **0.943** (0.90–0.98) \| 0.90 \| ≥ 0.85 — pass \| |
| `README.md:92` | #1, #2, #3, #8 | \| Lesion-wise Dice (matched) \| 0.835 \| 0.86 \| ≥ 0.70 — pass \| |
| `README.md:93` | #1, #2, #3, #8 | \| False positives / case \| 1.61 \| 0.57 \| ≤ 5 — pass \| |
| `README.md:94` | #1, #2, #3, #8 | \| HD95 \| 17.26 mm \| 1.78 mm \| ≤ 2.94 — **fail** \| |
| `README.md:95` | #1, #2, #3, #8 | \| Mean surface distance \| 3.89 mm \| 0.36 mm \| ≤ 0.66 — **fail** \| |
| `README.md:105` | #1, #2, #3, #8 | \| < 3 mm \| 33 \| 0.061 \| 0.54 \| 0.53 \| |
| `README.md:106` | #1, #2, #3, #8 | \| 3–5 mm \| 82 \| 0.439 \| 0.54 \| 0.64 \| |
| `README.md:107` | #1, #2, #3, #8 | \| 5–10 mm \| 304 \| 0.773 \| 0.46 \| 0.72 \| |
| `README.md:108` | #1, #2, #3, #8 | \| 10–20 mm \| 144 \| **0.903** \| 0.12 \| 0.81 \| |
| `README.md:109` | #1, #2, #3, #8 | \| > 20 mm \| 73 \| **0.986** \| 0.02 \| 0.87 \| |
| `configs/models.yaml:18` | #2, #3, #8 | # Held-out 84-case eval: Dice 0.7858, lesion-wise sens 0.810, FPs/case 1.68. |
| `configs/models.yaml:19` | #1, #2, #3, #8 | # At RANO-BM measurable-disease scope (>=10mm): lesion-wise sens 0.943, |
| `configs/models.yaml:20` | #1, #2, #3, #8 | # matched Dice 0.835. See docs/PRODUCTION.md. |
| `configs/models.yaml:151` | #2, #3, #8 | # Stacking meta-learner — production StackingClassifierV2 (Dice 0.7858) |
| `configs/models.yaml:166` | #2, #3, #8 | # hardcoded 0.5 while every published metric (Dice 0.7858, measurable-disease |
| `configs/models.yaml:167` | #1, #2, #3, #8 | # sensitivity 0.943) was produced by the evaluation scripts at 0.55. A site |
| `demo/app.py:786` | #2, #3, #8 | _model_info['val_dice'] = 0.7858 |
| `demo/app.py:791` | #1, #2, #3, #8 | print(f"Stacking Dice: 0.7858 (RANO-BM @≥10mm: Sens 0.943 vs Neosoma 0.90)") |
| `docs/CROSS_VALIDATION.md:140` | #2, #3, #8 |  |
| `docs/CROSS_VALIDATION.md:143` | #1, #2, #3, #8 | Indication for Use, and the scope on which JANNUS is competitive with the cleared |
| `docs/CROSS_VALIDATION.md:152` | #1, #2, #3, #8 | - **Smaller lesions.** Sensitivity is strongly size-dependent — 0.061 below 3 mm versus |
| `docs/CROSS_VALIDATION.md:153` | #1, #2, #3, #8 | 0.986 above 20 mm. Check the stratified table in your report before comparing headline |
| `docs/CROSS_VALIDATION.md:174` | #1, #2, #3, #8 | These are far above the FDA thresholds in our own results (17.26 mm vs ≤ 2.94 mm) and |
| `docs/INTENDED_USE.md:101` | #2, #5 | All metrics evaluated on **84 held-out cases** from UCSF Brain Metastases. Production model: **JANNUS v1.4 — tuned 7-model V2 stacking classifier** (4 LightweightUNet3D patch varia |
| `docs/INTENDED_USE.md:107` | #2, #3, #8 | \| Voxel-wise Dice \| 78.6 % \| Threshold 0.55, std 0.154 \| |
| `docs/INTENDED_USE.md:108` | #2, #3, #8 | \| Voxel-wise Sensitivity \| 78.2 % \| Whole-volume \| |
| `docs/INTENDED_USE.md:109` | #2, #3, #8 | \| Voxel-wise Precision \| 80.6 % \| Whole-volume \| |
| `docs/INTENDED_USE.md:111` | #2, #3, #8 | \| FPs / case \| 1.68 \| Lesion-level \| |
| `docs/INTENDED_USE.md:112` | #2, #3, #8 | \| Tiny-lesion Dice (<100 vox) \| 40.3 % \| +8.5 pt over v1.3 baseline \| |
| `docs/INTENDED_USE.md:118` | #1, #2, #3, #8 | \| Lesion-wise Sensitivity \| **0.943** (0.900 – 0.977) \| 0.90 (0.87 – 0.94) \| ≥ 0.85 \| ✓ exceeds predicate \| |
| `docs/INTENDED_USE.md:119` | #1, #2, #3, #8 | \| Lesion-wise Dice (matched) \| 0.835 (0.814 – 0.853) \| 0.86 (0.83 – 0.89) \| ≥ 0.70 \| ≈ near predicate \| |
| `docs/INTENDED_USE.md:120` | #1, #2, #3, #8 | \| FPs / case \| 1.61 (1.19 – 2.05) \| 0.57 (0.35 – 0.80) \| ≤ 5 \| ✓ passes FDA \| |
| `docs/INTENDED_USE.md:121` | #1, #2, #3, #8 | \| HD95 (mm) \| 17.26 (10.4 – 25.1) \| 1.78 (1.02 – 2.54) \| ≤ 2.94 \| ✗ structural gap (§8.3) \| |
| `docs/INTENDED_USE.md:122` | #1, #2, #3, #8 | \| MSD (mm) \| 3.89 (2.34 – 5.65) \| 0.36 (0.16 – 0.56) \| ≤ 0.66 \| ✗ structural gap (§8.3) \| |
| `docs/INTENDED_USE.md:123` | #1, #2, #3, #8 | \| Cohen's κ (measurable classification) \| **0.867** \| — \| ≥ 0.85 (RANO-BM) \| ✓ passes RANO-BM \| |
| `docs/INTENDED_USE.md:124` | #1, #2, #3, #8 | \| SoLD relative error \| **1.84 %** \| — \| ≤ 10 % (RANO-BM) \| ✓ 5× below threshold \| |
| `docs/INTENDED_USE.md:126` | #2 | All confidence intervals computed via 1000-resample bootstrap on the 84-case held-out cohort. |
| `docs/INTENDED_USE.md:133` | #2 | \| Held-out evaluation \| 84 cases (subject-level split) \| |
| `docs/INTENDED_USE.md:145` | #1, #2, #3, #8 | - **Substantial-equivalence argument:** Same intended use (CADe/CADx for brain metastases on multi-sequence MRI), same imaging modality, same patient population. Differs in scope ( |
| `docs/INTENDED_USE.md:179` | #2, #5 | **DISCLAIMER:** This document describes the intended future use of BrainMetScan. The software is currently designated **Research Use Only (RUO)** and must not be used for clinical  |
| `docs/MODEL_CARD.md:70` | #2 | 84-case held-out cohort, disjoint from training at the patient level. |
| `docs/MODEL_CARD.md:85` | #2, #3, #8 | \| Voxel Dice \| 0.7858 \| |
| `docs/MODEL_CARD.md:86` | #2, #3, #8 | \| Lesion-wise sensitivity \| 0.810 (0.758–0.858) \| |
| `docs/MODEL_CARD.md:87` | #2, #3, #8 | \| Lesion-wise Dice (matched) \| 0.791 \| |
| `docs/MODEL_CARD.md:88` | #2, #3, #8 | \| Precision \| 0.806 \| |
| `docs/MODEL_CARD.md:89` | #2, #3, #8 | \| False positives / case \| 1.68 (1.26–2.12) \| |
| `docs/MODEL_CARD.md:90` | #1, #2, #3, #8 | \| HD95 \| 18.36 mm (12.5–25.0) \| |
| `docs/MODEL_CARD.md:91` | #1, #2, #3, #8 | \| Mean surface distance \| 3.53 mm (2.5–4.9) \| |
| `docs/MODEL_CARD.md:97` | #1, #2, #3, #8 | \| Lesion-wise sensitivity \| **0.943** (0.90–0.98) \| 0.90 \| ≥ 0.85 \| pass \| |
| `docs/MODEL_CARD.md:98` | #1, #2, #3, #8 | \| Lesion-wise Dice (matched) \| 0.835 \| 0.86 \| ≥ 0.70 \| pass \| |
| `docs/MODEL_CARD.md:99` | #1, #2, #3, #8 | \| False positives / case \| 1.61 \| 0.57 \| ≤ 5 \| pass \| |
| `docs/MODEL_CARD.md:100` | #1, #2, #3, #8 | \| HD95 \| 17.26 mm \| 1.78 mm \| ≤ 2.94 mm \| **fail** \| |
| `docs/MODEL_CARD.md:101` | #1, #2, #3, #8 | \| Mean surface distance \| 3.89 mm \| 0.36 mm \| ≤ 0.66 mm \| **fail** \| |
| `docs/MODEL_CARD.md:102` | #1, #2, #3, #8 | \| Cohen's κ (measurable classification) \| **0.867** \| n/r \| ≥ 0.85 \| pass \| |
| `docs/MODEL_CARD.md:103` | #1, #2, #3, #8 | \| SoLD relative error \| **1.84 %** \| n/r \| ≤ 10 % \| pass \| |
| `docs/MODEL_CARD.md:112` | #1, #2, #3, #8 | \| < 3 mm \| 33 \| 0.061 \| 0.54 \| 0.53 \| |
| `docs/MODEL_CARD.md:113` | #1, #2, #3, #8 | \| 3–5 mm \| 82 \| 0.439 \| 0.54 \| 0.64 \| |
| `docs/MODEL_CARD.md:114` | #1, #2, #3, #8 | \| 5–10 mm \| 304 \| 0.773 \| 0.46 \| 0.72 \| |
| `docs/MODEL_CARD.md:115` | #1, #2, #3, #8 | \| 10–20 mm \| 144 \| 0.903 \| 0.12 \| 0.81 \| |
| `docs/MODEL_CARD.md:116` | #1, #2, #3, #8 | \| > 20 mm \| 73 \| 0.986 \| 0.02 \| 0.87 \| |
| `docs/MODEL_CARD.md:136` | #1, #2, #3, #8 | **Sub-3 mm sensitivity is 0.061** — effectively zero. This is at the floor of MRI |
| `docs/MODEL_CARD.md:146` | #2 | **No prospective validation.** All results are retrospective on a single held-out cohort. |
| `docs/MODEL_CARD.md:163` | #2, #3, #8 | anxiety, and at 1.68 per case they are frequent enough to matter for workflow. |
| `docs/TROUBLESHOOTING.md:127` | #1, #2, #3, #8 | 2. **Check the size distribution.** Sensitivity ranges from 0.061 below 3 mm to 0.986 |
| `docs/research/PRODUCTION.md:21` | #2 | ## Headline performance (84-case held-out eval) |
| `docs/research/PRODUCTION.md:27` | #2, #3, #8 | \| Voxel Dice \| **0.7858** \| 0.7778 \| |
| `docs/research/PRODUCTION.md:28` | #2, #3, #8 | \| Lesion-wise Sensitivity \| 0.810 (95% CI 0.758-0.858) \| — \| |
| `docs/research/PRODUCTION.md:29` | #2, #3, #8 | \| Lesion-wise Dice (matched) \| 0.791 \| — \| |
| `docs/research/PRODUCTION.md:30` | #2, #3, #8 | \| Precision \| 0.806 \| 0.805 \| |
| `docs/research/PRODUCTION.md:31` | #2, #3, #8 | \| FPs / case \| 1.68 (95% CI 1.26-2.12) \| — \| |
| `docs/research/PRODUCTION.md:32` | #1, #2, #3, #8 | \| HD95 (mm) \| 18.36 (95% CI 12.5-25.0) \| — \| |
| `docs/research/PRODUCTION.md:33` | #1, #2, #3, #8 | \| MSD (mm) \| 3.53 (95% CI 2.5-4.9) \| — \| |
| `docs/research/PRODUCTION.md:39` | #1, #2, #3, #8 | \| <3mm \| 33 \| **0.061** \| 0.54 \| 0.53 \| |
| `docs/research/PRODUCTION.md:40` | #1, #2, #3, #8 | \| 3-5mm \| 82 \| 0.439 \| 0.54 \| 0.64 \| |
| `docs/research/PRODUCTION.md:41` | #1, #2, #3, #8 | \| 5-10mm \| 304 \| 0.773 \| 0.46 \| 0.72 \| |
| `docs/research/PRODUCTION.md:42` | #1, #2, #3, #8 | \| 10-20mm \| 144 \| **0.903** \| 0.12 \| 0.81 \| |
| `docs/research/PRODUCTION.md:43` | #1, #2, #3, #8 | \| >20mm \| 73 \| **0.986** \| 0.02 \| 0.87 \| |
| `docs/research/PRODUCTION.md:52` | #1, #2, #3, #8 | \| Lesion-wise Sensitivity \| **0.943** (0.90-0.98) \| 0.90 \| ≥ 0.85 ✓ \| |
| `docs/research/PRODUCTION.md:53` | #1, #2, #3, #8 | \| Lesion-wise Dice (matched) \| 0.835 \| 0.86 \| ≥ 0.70 ✓ \| |
| `docs/research/PRODUCTION.md:54` | #1, #2, #3, #8 | \| FPs / case \| 1.61 \| 0.57 \| ≤ 5 ✓ \| |
| `docs/research/PRODUCTION.md:55` | #1, #2, #3, #8 | \| HD95 (mm) \| 17.26 \| 1.78 \| ≤ 2.94 ✗ \| |
| `docs/research/PRODUCTION.md:56` | #1, #2, #3, #8 | \| MSD (mm) \| 3.89 \| 0.36 \| ≤ 0.66 ✗ \| |
| `docs/research/PRODUCTION.md:57` | #1, #2, #3, #8 | \| Cohen's κ (measurable classification) \| **0.867** \| not reported \| ≥ 0.85 ✓ (RANO-BM) \| |
| `docs/research/PRODUCTION.md:58` | #1, #2, #3, #8 | \| SoLD relative error \| **1.84%** \| not reported \| ≤ 10% ✓ (RANO-BM) \| |
| `docs/research/PRODUCTION.md:60` | #1, #2, #3, #8 | **At ≥10mm scope, BMS exceeds Neosoma on lesion-wise sensitivity and matches on |
| `docs/research/stacking_architectures_explored.md:9` | #2, #3, #8 | 7-model V2, Dice 0.7858. All other architectures were either worse overall |
| `docs/research/stacking_architectures_explored.md:18` | #2, #3, #8 | - **Comparison reference:** Old 6-model stacker (v4) at Dice 0.7778 (legacy production); Neosoma Brain Mets 510(k) K252922 as external competitive benchmark |
| `docs/research/stacking_architectures_explored.md:141` | #2, #3, #8 | \| Dice \| 0.7778 \| **0.7858** (+0.008) \| |
| `docs/research/stacking_architectures_explored.md:148` | #2, #3, #8 | \| Lesion-wise Sens \| — \| 0.810 \| |
| `docs/research/stacking_architectures_explored.md:149` | #2, #3, #8 | \| FPs/case \| — \| **1.68** \| |
| `docs/research/stacking_architectures_explored.md:150` | #1, #2, #3, #8 | \| HD95 (mm) \| — \| **18.36** \| |
| `docs/research/stacking_architectures_explored.md:151` | #2, #3, #8 | \| Lesion-wise Dice \| — \| 0.791 \| |
| `docs/research/stacking_architectures_explored.md:154` | #1, #2, #3, #8 | - Lesion-wise Sens **0.943** — exceeds Neosoma's 0.90 |
| `docs/research/stacking_architectures_explored.md:155` | #1, #2, #3, #8 | - Matched Dice **0.835** — near Neosoma's 0.86 |
| `docs/research/stacking_architectures_explored.md:156` | #1, #2, #3, #8 | - Cohen's κ **0.867** — passes RANO-BM ≥0.85 |
| `docs/research/stacking_architectures_explored.md:157` | #1, #2, #3, #8 | - SoLD error **1.84%** — beats RANO-BM ≤10% by 5× |
| `docs/research/stacking_architectures_explored.md:161` | #1, #2, #3, #8 | to achieve Neosoma-competitive numbers on measurable-disease scope. |
| `docs/research/stacking_architectures_explored.md:180` | #2, #3, #8 | \| Dice \| 0.7858 \| 0.7808 (−0.005) \| |
| `docs/research/stacking_architectures_explored.md:183` | #1, #2, #3, #8 | \| HD95 (mm) \| 18.36 \| **21.72** (+3.4, WORSE) \| |
| `docs/research/stacking_architectures_explored.md:184` | #2, #3, #8 | \| FPs/case \| 1.68 \| **2.44** (+0.76, WORSE) \| |
| `docs/research/stacking_architectures_explored.md:223` | #2, #3, #8 | \| Dice \| 0.7858 \| 0.7693 (−0.017) \| |
| `docs/research/stacking_architectures_explored.md:227` | #1, #2, #3, #8 | \| HD95 (mm) \| 18.36 \| **24.02** (+5.7, WORSE) \| |
| `docs/research/stacking_architectures_explored.md:228` | #2, #3, #8 | \| FPs/case \| 1.68 \| **4.42** (+2.74, WORSE) \| |
| `docs/research/stacking_architectures_explored.md:263` | #2, #3, #8 | \| Dice \| 0.7858 \| 0.7802 (−0.006) \| |
| `docs/research/stacking_architectures_explored.md:265` | #1, #2, #3, #8 | \| HD95 (mm) \| 18.36 \| 20.46 (+2.1) \| |
| `docs/research/stacking_architectures_explored.md:266` | #2, #3, #8 | \| FPs/case \| 1.68 \| 1.87 (+0.19) \| |
| `docs/research/stacking_architectures_explored.md:297` | #2, #3, #8 | \| Dice \| 0.7858 \| 0.7761 (−0.010) \| |
| `docs/research/stacking_architectures_explored.md:300` | #1, #2, #3, #8 | \| HD95 (mm) \| 18.36 \| 20.43 (+2.1) \| |
| `docs/research/stacking_architectures_explored.md:301` | #2, #3, #8 | \| FPs/case \| 1.68 \| 2.13 (+0.45) \| |
| `docs/research/stacking_architectures_explored.md:326` | #2, #3, #8 | Tuned 7-model V2 (CHAMPION)       0.7858    —                production |
| `docs/research/stacking_architectures_explored.md:331` | #2, #3, #8 | Old 6-model (v4)                  0.7778    -0.008           legacy |
| `docs/research/stacking_architectures_explored.md:336` | #2, #3, #8 | v1 uniform-256 hybrid (broken)    0.6453    -0.141           abandoned |
| `docs/research/stacking_architectures_explored.md:345` | #1, #2, #3, #8 | clear. The remaining FDA gap (HD95 18.36mm vs ≤2.94mm) is not addressable at |
| `docs/research/stacking_architectures_explored.md:353` | #1, #2, #3, #8 | recommendation**, as numbers are already Neosoma-competitive at that scope. |
| `docs/research/stacking_v4_vs_v5_writeup.md:11` | #2, #3, #8 | - Dice 0.7858 (overall, all lesions), beats old 6-model by +0.008 |
| `docs/research/stacking_v4_vs_v5_writeup.md:12` | #2, #3, #8 | - Lesion-wise Sens 0.810, FPs/case 1.68, Lesion-wise Dice 0.791 |
| `docs/research/stacking_v4_vs_v5_writeup.md:13` | #1, #2, #3, #8 | - **When scoped to ≥10mm ("measurable disease"): Sens 0.943, Dice-matched 0.835, Cohen's κ 0.867 (passes RANO-BM), SoLD error 1.84% — Neosoma-competitive** |
| `docs/research/stacking_v4_vs_v5_writeup.md:14` | #1, #2, #3, #8 | - **Remaining FDA blocker:** HD95 (18.36mm standalone, 17.26mm measurable-only) vs FDA ≤2.94mm. |
| `docs/research/stacking_v4_vs_v5_writeup.md:27` | #2 | \| Eval cases \| 84 held-out \| 84 held-out (same) \| |
| `docs/research/stacking_v4_vs_v5_writeup.md:29` | #2 | All metrics are computed on the same 84 held-out validation cases used for the v4 evaluation (`stacking_v5_results_per_case.json`). Threshold is swept per model from 0.05–0.95 in 0 |
| `docs/research/stacking_v4_vs_v5_writeup.md:39` | #2, #3, #8 | \| Dice \| 0.7778 \| **0.6453** \| **−0.133** \| |
| `docs/research/stacking_v4_vs_v5_writeup.md:49` | #2, #3, #8 | \| Trained stacker \| 0.6453 \| 0.7778 (6-model) \| 0.65 \| |
| `docs/research/stacking_v4_vs_v5_writeup.md:92` | #2, #3, #8 | \| Dice \| 0.7778 \| 0.7673 \| −0.0105 \| |
| `docs/research/stacking_v4_vs_v5_writeup.md:176` | #2, #3, #8 | \| Dice \| 0.7778 \| 0.7673 \| 0.7237 \| 0.7531 \| |
| `docs/research/stacking_v4_vs_v5_writeup.md:288` | #2, #3, #8 | Goal was to break past the 7-model Dice 0.7858 plateau. Spoiler: nothing in this round beat it. Useful negative information. |
| `docs/research/stacking_v4_vs_v5_writeup.md:309` | #2, #3, #8 | Result: **Dice 0.7858**, first config to beat the old 6-model aggregate. |
| `docs/research/stacking_v4_vs_v5_writeup.md:313` | #2, #3, #8 | \| Dice \| 0.7778 \| **0.7858** \| |
| `docs/research/stacking_v4_vs_v5_writeup.md:317` | #2, #3, #8 | \| FPs/case \| — \| 1.68 \| |
| `docs/research/stacking_v4_vs_v5_writeup.md:318` | #1, #2, #3, #8 | \| HD95 (mm) \| — \| 18.36 \| |
| `docs/research/stacking_v4_vs_v5_writeup.md:322` | #1, #2, #3, #8 | Per-base FDA diagnostic showed patch_8 was the worst base (HD95 35mm, 9.93 FPs/case). Dropping it: Dice 0.7808 (−0.005 overall), tiny bucket 0.4222 (+0.02), **but HD95 went UP** (1 |
| `docs/research/stacking_v4_vs_v5_writeup.md:353` | #2, #3, #8 | - 141 total FPs across 84 cases (1.68/case) |
| `docs/research/stacking_v4_vs_v5_writeup.md:371` | #1, #2, #3, #8 | \| 0 (all) \| — \| 0.786 \| 0.810 \| 1.68 \| 18.36 \| 636/636 \| |
| `docs/research/stacking_v4_vs_v5_writeup.md:372` | #2, #3, #8 | \| 5 vox \| 2.1mm \| 0.786 \| 0.841 \| 1.66 \| 18.34 \| 601/636 \| |
| `docs/research/stacking_v4_vs_v5_writeup.md:375` | #1, #2, #3, #8 | \| **100 vox** \| **5.8mm** \| 0.784 \| **0.951** ✓ \| 1.61 \| 14.36 \| 287/636 \| |
| `docs/research/stacking_v4_vs_v5_writeup.md:380` | #1, #2, #3, #8 | - IFU scoped to **≥6mm** → Sens 0.951, Dice steady at 0.78 |
| `docs/research/stacking_v4_vs_v5_writeup.md:390` | #1, #2, #3, #8 | \| Lesion-wise Sensitivity \| **0.943** (0.90-0.98) \| 0.90 \| ≥0.85 \| ✓ **exceeds Neosoma** \| |
| `docs/research/stacking_v4_vs_v5_writeup.md:391` | #1, #2, #3, #8 | \| FPs / case \| 1.61 \| 0.57 \| ≤5 \| ✓ passes \| |
| `docs/research/stacking_v4_vs_v5_writeup.md:392` | #1, #2, #3, #8 | \| Lesion-wise Dice (matched) \| **0.835** \| 0.86 \| ≥0.70 \| ✓ near Neosoma \| |
| `docs/research/stacking_v4_vs_v5_writeup.md:393` | #1, #2, #3, #8 | \| HD95 (mm) \| 17.26 \| 1.78 \| ≤2.94 \| ✗ fails \| |
| `docs/research/stacking_v4_vs_v5_writeup.md:394` | #1, #2, #3, #8 | \| MSD (mm) \| 3.89 \| 0.36 \| ≤0.66 \| ✗ fails \| |
| `docs/research/stacking_v4_vs_v5_writeup.md:398` | #1, #2, #3, #8 | - **Cohen's κ: 0.867** ✓ passes RANO-BM ≥0.85 |
| `docs/research/stacking_v4_vs_v5_writeup.md:399` | #1, #2, #3, #8 | - **SoLD relative error: 1.84%** ✓ beats RANO-BM ≤10% by 5× |
| `docs/research/stacking_v4_vs_v5_writeup.md:402` | #1, #2, #3, #8 | **This is the first credibly FDA-competitive profile we've achieved.** Two Neosoma metrics exceeded, one matched, Cohen's κ passes RANO-BM. Only HD95/MSD remain failing. |
| `docs/research/stacking_v4_vs_v5_writeup.md:410` | #1, #2, #3, #8 | \| <3mm \| 33 \| **0.061** \| 0.54 \| 0.53 \| |
| `docs/research/stacking_v4_vs_v5_writeup.md:411` | #1, #2, #3, #8 | \| 3-5mm \| 82 \| 0.439 \| 0.54 \| 0.64 \| |
| `docs/research/stacking_v4_vs_v5_writeup.md:412` | #1, #2, #3, #8 | \| 5-10mm \| 304 \| 0.773 \| 0.46 \| 0.72 \| |
| `docs/research/stacking_v4_vs_v5_writeup.md:413` | #1, #2, #3, #8 | \| 10-20mm \| 144 \| **0.903** \| 0.12 \| 0.81 \| |
| `docs/research/stacking_v4_vs_v5_writeup.md:414` | #1, #2, #3, #8 | \| >20mm \| 73 \| **0.986** \| 0.02 \| 0.87 \| |
| `docs/research/stacking_v4_vs_v5_writeup.md:422` | #1, #2, #3, #8 | \| 1 \| **Tuned 7-model V2** \| **0.7858** \| 0.4028 \| 18.36 \| 1.68 \| **production** \| |
| `docs/research/stacking_v4_vs_v5_writeup.md:426` | #2, #3, #8 | \| 5 \| Old 6-model (v4) \| 0.7778 \| 0.3182 \| — \| — \| legacy \| |
| `docs/research/stacking_v4_vs_v5_writeup.md:438` | #1, #2, #3, #8 | - At ≥10mm: Sens 0.943, Dice 0.835, RANO-BM κ 0.867, SoLD error 1.84% — all Neosoma-competitive. |
| `docs/research/stacking_v4_vs_v5_writeup.md:476` | #2, #3, #8 | - `model/stacking_v5_seven_model_classifier.pth` — **production champion (Dice 0.7858)** |
| `docs/research/stacking_v4_vs_v5_writeup.md:514` | #2, #3, #8 | 6. **Bigger stacker ≠ better without tuned training.** StackingClassifierV2 (1.36M params) regressed to Dice 0.7629 under stock hyperparams but reached 0.7858 with warmup + augment |
| `scripts/evaluation/seven_model_stacker.py:15` | #2, #3, #8 | Hypothesis: the old 6-model tops the aggregate Dice leaderboard (0.7778); |
| `src/jannus/configs/models.yaml:18` | #2, #3, #8 | # Held-out 84-case eval: Dice 0.7858, lesion-wise sens 0.810, FPs/case 1.68. |
| `src/jannus/configs/models.yaml:19` | #1, #2, #3, #8 | # At RANO-BM measurable-disease scope (>=10mm): lesion-wise sens 0.943, |
| `src/jannus/configs/models.yaml:20` | #1, #2, #3, #8 | # matched Dice 0.835. See docs/PRODUCTION.md. |
| `src/jannus/configs/models.yaml:151` | #2, #3, #8 | # Stacking meta-learner — production StackingClassifierV2 (Dice 0.7858) |
| `src/jannus/configs/models.yaml:166` | #2, #3, #8 | # hardcoded 0.5 while every published metric (Dice 0.7858, measurable-disease |
| `src/jannus/configs/models.yaml:167` | #1, #2, #3, #8 | # sensitivity 0.943) was produced by the evaluation scripts at 0.55. A site |
| `src/jannus/core/config.py:35` | #1, #2, #3, #8 | #: (Dice 0.7858, lesion sensitivity 0.810, RANO-BM >=10mm sensitivity 0.943). |
| `src/jannus/evaluation/harness.py:14` | #1, #2, #3, #8 | which JANNUS is competitive with the cleared predicate. |
| `src/jannus/evaluation/metrics.py:17` | #2 | Assumes 1mm isotropic voxels (true for the 84 held-out eval cases at 256^3). |
| `src/jannus/evaluation/report.py:15` | #2 | #: Published JANNUS figures on the 84-case internal held-out cohort. Shown |
| `src/jannus/evaluation/report.py:20` | #2, #3, #8 | "voxel_dice": 0.7858, |
| `src/jannus/evaluation/report.py:21` | #2, #3, #8 | "lesion_sensitivity_all": 0.810, |
| `src/jannus/evaluation/report.py:22` | #2, #3, #8 | "fp_per_case_all": 1.68, |
| `src/jannus/evaluation/report.py:23` | #1, #2, #3, #8 | "lesion_sensitivity_measurable": 0.943, |
| `src/jannus/evaluation/report.py:24` | #1, #2, #3, #8 | "lesion_dice_measurable": 0.835, |
| `src/jannus/evaluation/report.py:25` | #1, #2, #3, #8 | "hd95_measurable": 17.26, |
| `src/jannus/evaluation/report.py:76` | #2 | "Reference (internal 84-case held-out):", |
| `src/jannus/segmentation/model_adapters.py:22` | #2, #3, #8 | that produced the documented Dice 0.7858. |
| `src/jannus/segmentation/pipeline.py:38` | #2 | Headline metrics (held-out 84-case eval; see docs/PRODUCTION.md): |
| `src/jannus/segmentation/pipeline.py:39` | #2, #3, #8 | voxel Dice 0.7858, lesion-wise sens 0.810, FPs/case 1.68 |
| `src/jannus/segmentation/pipeline.py:40` | #1, #2, #3, #8 | RANO-BM measurable-disease (>=10mm): sens 0.943, matched Dice 0.835 |
| `src/jannus/segmentation/stacking.py:14` | #2, #3, #8 | val Dice     = 0.7858 on 84-case held-out |

## RUNTIME-UNMARKED (3)

| file:line | defects | text (original) |
|---|---|---|
| `demo/app.py:2889` | #2, #3, #8 | v1.4 7-Model Stacking V2 (78.6% Dice) |
| `demo/app.py:2895` | #2, #3, #8 | RANO-BM Sens 94.3% vs Neosoma 90% |
| `demo/app.py:3071` | #2, #3, #8 | <p style="color: #a8a8a3;">7-Model Stacking V2 (78.6% Dice)  ·  RANO-BM Sens 94.3% vs Neosoma K252922 (90.0%)  ·  RAG-Enhanced Reports  ·  BiomedCLIP + BM25</p> |

## UNAUDITED (23)

| file:line | defects | text (original) |
|---|---|---|
| `docs/research/BrainMetScan_improvements.md:4` | 'external' = external sites, folders, dependencies or a different study; not a claim about the 84 | presentable for external collaborators, lab PIs, and potential PhD applications. |
| `docs/research/BrainMetScan_improvements.md:10` | 'external' = external sites, folders, dependencies or a different study; not a claim about the 84 | ## 1. README.md — Rewrite for External Audience |
| `docs/research/BrainMetScan_improvements.md:83` | 'external' = external sites, folders, dependencies or a different study; not a claim about the 84 | - The `1.2/` file tree block (confusing to external readers, implied by repo structure) |
| `docs/research/BrainMetScan_improvements.md:178` | 'external' = external sites, folders, dependencies or a different study; not a claim about the 84 | # Model weights (too large, use releases or external hosting) |
| `docs/research/BrainMetScan_improvements.md:298` | 'external' = external sites, folders, dependencies or a different study; not a claim about the 84 | - Multi-site MRI datasets for external validation |
| `docs/research/atlas_experiment_postmortem.md:6` | 'external' = external sites, folders, dependencies or a different study; not a claim about the 84 | **Objective**: Train a dedicated segmentation model on atlas-registered (SRI24) MRI data to address the domain shift that causes the native-geometry model (Dice 0.747) to drop to 0 |
| `docs/research/atlas_experiment_postmortem.md:43` | cohort word refers to a different split (per-fold, internal-val, test name, other study) | ### Full-Volume Evaluation (36 held-out val cases, threshold 0.40, overlap 0.25) |
| `docs/research/atlas_experiment_postmortem.md:72` | 'external' = external sites, folders, dependencies or a different study; not a claim about the 84 | The native model drops from 0.747 → 0.17–0.20 on atlas-registered data (documented in external_validation_report.md). This is a systematic geometric domain shift from SRI24 atlas w |
| `docs/research/atlas_experiment_postmortem.md:101` | 'external' = external sites, folders, dependencies or a different study; not a claim about the 84 | - Data symlinks: `data/atlas_registered/` (symlinks only; source data in `data/external_validation/` preserved) |
| `docs/research/atlas_experiment_postmortem.md:105` | 'external' = external sites, folders, dependencies or a different study; not a claim about the 84 | The external validation datasets (PRETREAT in `data/external_validation/`, PROTEAS in `data/external_validation_proteas/`) and their evaluation scripts are preserved — they remain  |
| `docs/research/external_validation_report.md:1` | 'external' = external sites, folders, dependencies or a different study; not a claim about the 84 | # External Validation Report |
| `docs/research/external_validation_report.md:5` | 'external' = external sites, folders, dependencies or a different study; not a claim about the 84 | External validation was performed on two independent, publicly available brain metastasis datasets to assess generalization of the BrainMetScan segmentation pipeline beyond the tra |
| `docs/research/external_validation_report.md:13` | 'external' = external sites, folders, dependencies or a different study; not a claim about the 84 | ## External Datasets |
| `docs/research/external_validation_report.md:33` | 'external' = external sites, folders, dependencies or a different study; not a claim about the 84 | The top-3 ensemble (average of 12-patch, 24-patch, and 36-patch models at threshold 0.40) was evaluated on both external datasets. Each external case was resized from 240×240×155 t |
| `docs/research/external_validation_report.md:55` | 'external' = external sites, folders, dependencies or a different study; not a claim about the 84 | Both external datasets show a consistent ~75% reduction in Dice score compared to internal validation. The consistency of this drop across two independent institutions (Yale and Cy |
| `docs/research/external_validation_report.md:57` | 'external' = external sites, folders, dependencies or a different study; not a claim about the 84 | The key difference between training and external data: |
| `docs/research/external_validation_report.md:59` | 'external' = external sites, folders, dependencies or a different study; not a claim about the 84 | \| Property \| Training Data \| External Data \| |
| `docs/research/external_validation_report.md:66` | 'external' = external sites, folders, dependencies or a different study; not a claim about the 84 | Skull-stripping is not the source of the domain shift — both training and external datasets are skull-stripped. The critical difference is that training data preserves native spati |
| `docs/research/external_validation_report.md:70` | 'external' = external sites, folders, dependencies or a different study; not a claim about the 84 | Despite poor segmentation quality, the model retains moderate lesion detection capability on external data. On PROTEAS, lesion recall was 0.649 (detecting 63 of 116 lesions), sugge |
| `docs/research/external_validation_report.md:78` | 'external' = external sites, folders, dependencies or a different study; not a claim about the 84 | External validation on two independent BraTS-format datasets (PRETREAT, n=200; PROTEAS, n=44) showed reduced performance (Dice 0.174, 0.202 respectively) compared to internal cross |
| `docs/research/external_validation_report.md:86` | 'external' = external sites, folders, dependencies or a different study; not a claim about the 84 | - Evaluation: `scripts/evaluation/eval_external_validation.py --data-dir <path> --output-dir <path>` |
| `docs/research/external_validation_report.md:87` | 'external' = external sites, folders, dependencies or a different study; not a claim about the 84 | - PRETREAT results: `results/external_validation/external_validation_results.json` |
| `docs/research/external_validation_report.md:88` | 'external' = external sites, folders, dependencies or a different study; not a claim about the 84 | - PROTEAS results: `results/external_validation_proteas/external_validation_results.json` |

## INCIDENTAL (72)

| file:line | defects | text (original) |
|---|---|---|
| `.github/workflows/ci.yml:65` | 'external' = external sites, folders, dependencies or a different study; not a claim about the 84 | # default for a suite external sites also run and report back to us. |
| `.github/workflows/update-rag-corpus.yml:14` | 'external' = external sites, folders, dependencies or a different study; not a claim about the 84 | #      external validation sites clone and run against patient data, an |
| `CHANGELOG.md:52` | 'external' = external sites, folders, dependencies or a different study; not a claim about the 84 | handed to an external hospital or academic group and run against their own patients. |
| `CHANGELOG.md:67` | 'external' = external sites, folders, dependencies or a different study; not a claim about the 84 | the single most likely cause of a failed external cross-validation. |
| `CHANGELOG.md:146` | 'external' = external sites, folders, dependencies or a different study; not a claim about the 84 | external sites run and report back. |
| `CITATION.cff:2` | 'external' = external sites, folders, dependencies or a different study; not a claim about the 84 | title: "JANNUS: brain metastasis segmentation for multi-site external validation" |
| `CITATION.cff:15` | 'external' = external sites, folders, dependencies or a different study; not a claim about the 84 | packaged for external cross-validation, providing dataset QC, run |
| `CITATION.cff:24` | 'external' = external sites, folders, dependencies or a different study; not a claim about the 84 | - external validation |
| `CONTRIBUTING.md:6` | 'external' = external sites, folders, dependencies or a different study; not a claim about the 84 | Collaboration is especially welcome on: external validation cohorts, consensus |
| `Dockerfile:58` | 'external' = external sites, folders, dependencies or a different study; not a claim about the 84 | # Default to the CLI, which is what an external validation site needs. |
| `README.md:3` | 'external' = external sites, folders, dependencies or a different study; not a claim about the 84 | **Brain metastasis segmentation on multi-sequence MRI, packaged for multi-site external validation.** |
| `README.md:19` | 'external' = external sites, folders, dependencies or a different study; not a claim about the 84 | If you are an external site, this is your entire workflow: |
| `docker-compose.yml:3` | 'external' = external sites, folders, dependencies or a different study; not a claim about the 84 | # For the external-validation workflow you do NOT need this. Use the CLI: |
| `docs/CROSS_VALIDATION.md:1` | 'external' = external sites, folders, dependencies or a different study; not a claim about the 84 | # External cross-validation protocol |
| `docs/CROSS_VALIDATION.md:164` | 'external' = external sites, folders, dependencies or a different study; not a claim about the 84 | precisely the information external validation exists to produce — please report it as |
| `docs/INTENDED_USE.md:155` | 'external' = external sites, folders, dependencies or a different study; not a claim about the 84 | - [ ] Multi-site external validation on independent native-geometry data |
| `docs/INTENDED_USE.md:174` | 'external' = external sites, folders, dependencies or a different study; not a claim about the 84 | - The optional RAG literature-retrieval pipeline operates against a local PubMed-derived corpus and does not transmit case data externally. |
| `docs/MODEL_CARD.md:34` | 'external' = external sites, folders, dependencies or a different study; not a claim about the 84 | external validation studies; volumetric measurement research under RANO-BM. |
| `docs/MODEL_CARD.md:63` | 'external' = external sites, folders, dependencies or a different study; not a claim about the 84 | for a model intended for multi-site use, and is a primary reason external validation |
| `docs/research/ROADMAP_FDA_95_SENSITIVITY.md:117` | 'external' = external sites, folders, dependencies or a different study; not a claim about the 84 | ## Phase 2: External Validation (Weeks 9-16) |
| `docs/research/ROADMAP_FDA_95_SENSITIVITY.md:119` | 'external' = external sites, folders, dependencies or a different study; not a claim about the 84 | ### Step 2.1: Acquire External Datasets |
| `docs/research/ROADMAP_FDA_95_SENSITIVITY.md:130` | 'external' = external sites, folders, dependencies or a different study; not a claim about the 84 | - 200+ patients external to training data |
| `docs/research/ROADMAP_FDA_95_SENSITIVITY.md:136` | 'external' = external sites, folders, dependencies or a different study; not a claim about the 84 | **For Each External Site**: |
| `docs/research/ROADMAP_FDA_95_SENSITIVITY.md:144` | 'external' = external sites, folders, dependencies or a different study; not a claim about the 84 | **Target**: Sensitivity ≥95% maintained across all external sites |
| `docs/research/ROADMAP_FDA_95_SENSITIVITY.md:233` | 'external' = external sites, folders, dependencies or a different study; not a claim about the 84 | \| External datasets \| $0-5K \| |
| `docs/research/ROADMAP_FDA_95_SENSITIVITY.md:247` | 'external' = external sites, folders, dependencies or a different study; not a claim about the 84 | \| External validation \| Weeks 9-16 \| 3+ site validation complete \| |
| `docs/research/ROADMAP_FDA_95_SENSITIVITY.md:288` | 'external' = external sites, folders, dependencies or a different study; not a claim about the 84 | - [ ] External validation maintains performance |
| `docs/research/ROADMAP_FDA_95_SENSITIVITY.md:408` | 'external' = external sites, folders, dependencies or a different study; not a claim about the 84 | \| External validation drops significantly \| Domain adaptation, scanner-specific fine-tuning \| |
| `docs/research/dead_configs/README.md:24` | 'external' = external sites, folders, dependencies or a different study; not a claim about the 84 | actively misleading to an external site — someone would reasonably tune `num_workers` or |
| `docs/research/stacking_architectures_explored.md:22` | cohort word refers to a different split (per-fold, internal-val, test name, other study) | with a held-out 48-case internal val for checkpoint selection. All inference uses |
| `docs/research/stacking_architectures_explored.md:64` | cohort word refers to a different split (per-fold, internal-val, test name, other study) | (skipped to save ~10 hrs of compute); meta trained on a 96-case held-out from |
| `docs/research/stacking_v4_vs_v5_writeup.md:83` | cohort word refers to a different split (per-fold, internal-val, test name, other study) | - Best checkpoint selected by internal-val loss (59-case held-out split of training pool). |
| `docs/research/stacking_v4_vs_v5_writeup.md:141` | cohort word refers to a different split (per-fold, internal-val, test name, other study) | - Splits: 385 SC-train / 42 SC-val / 96 meta-train (held out from SCs) / 84 eval |
| `docs/risk_analysis.md:28` | 'external' = external sites, folders, dependencies or a different study; not a claim about the 84 | > pre-submission, or any external representation of the system's controls, those |
| `pyproject.toml:7` | 'external' = external sites, folders, dependencies or a different study; not a claim about the 84 | description = "Brain metastasis segmentation on multi-sequence MRI, packaged for multi-site external validation" |
| `run_server.py:6` | 'external' = external sites, folders, dependencies or a different study; not a claim about the 84 | The API is OPTIONAL. External validation sites do not need it: use the `jannus` |
| `scripts/evaluation/full_stacking_comparison_v2.py:168` | cohort word refers to a different split (per-fold, internal-val, test name, other study) | print(f"  Train: {len(t_ids)}, internal val: {len(v_ids)}, eval held out: {len(eval_ids)}") |
| `scripts/evaluation/seven_model_stacker.py:203` | cohort word refers to a different split (per-fold, internal-val, test name, other study) | print(f"  Train: {len(t_ids)}, internal val: {len(v_ids)}, eval held out: {len(eval_ids)}") |
| `scripts/inference/nnunet_probs.py:10` | cohort word refers to a different split (per-fold, internal-val, test name, other study) | fold's held-out validation cases. This avoids data leakage — each case is |
| `scripts/inference/nnunet_probs.py:368` | cohort word refers to a different split (per-fold, internal-val, test name, other study) | print(f"\nMode: VALIDATION — predicting per-fold held-out cases") |
| `scripts/inference/run_inference.py:3` | 'external' = external sites, folders, dependencies or a different study; not a claim about the 84 | This script was the external-site entry point through v1.40. It is retained |
| `scripts/inference/run_inference.py:9` | 'external' = external sites, folders, dependencies or a different study; not a claim about the 84 | The v1.40 implementation carried three defects that specifically hurt external |
| `scripts/preprocessing/download_proteas.py:40` | 'external' = external sites, folders, dependencies or a different study; not a claim about the 84 | OUTPUT_DIR = PROJECT_ROOT / "data" / "external_validation_proteas" / "test" |
| `scripts/preprocessing/prepare_pretreat_validation.py:2` | 'external' = external sites, folders, dependencies or a different study; not a claim about the 84 | Prepare PRETREAT-METSTOBRAIN-MASKS dataset as external validation set. |
| `scripts/preprocessing/prepare_pretreat_validation.py:22` | 'external' = external sites, folders, dependencies or a different study; not a claim about the 84 | python scripts/prepare_pretreat_validation.py --input-dir "Raw Data/PRETREAT" --output-dir data/external_validation |
| `scripts/preprocessing/prepare_pretreat_validation.py:25` | 'external' = external sites, folders, dependencies or a different study; not a claim about the 84 | python scripts/prepare_pretreat_validation.py --download --output-dir data/external_validation |
| `scripts/preprocessing/prepare_pretreat_validation.py:28` | 'external' = external sites, folders, dependencies or a different study; not a claim about the 84 | python scripts/prepare_pretreat_validation.py --input-dir "Raw Data/PRETREAT" --output-dir data/external_validation --setup-nnunet |
| `scripts/preprocessing/prepare_pretreat_validation.py:148` | 'external' = external sites, folders, dependencies or a different study; not a claim about the 84 | "  4. Re-run: python scripts/prepare_pretreat_validation.py --input-dir <folder> --output-dir data/external_validation" |
| `scripts/preprocessing/prepare_pretreat_validation.py:346` | 'external' = external sites, folders, dependencies or a different study; not a claim about the 84 | Set up external validation data in nnU-Net format for inference. |
| `scripts/preprocessing/prepare_pretreat_validation.py:412` | 'external' = external sites, folders, dependencies or a different study; not a claim about the 84 | "description": "PRETREAT-METSTOBRAIN-MASKS external validation set (TCIA)", |
| `scripts/preprocessing/prepare_pretreat_validation.py:429` | 'external' = external sites, folders, dependencies or a different study; not a claim about the 84 | description="Prepare PRETREAT-METSTOBRAIN-MASKS as external validation set", |
| `scripts/preprocessing/prepare_pretreat_validation.py:434` | 'external' = external sites, folders, dependencies or a different study; not a claim about the 84 | python scripts/prepare_pretreat_validation.py --download --output-dir data/external_validation |
| `scripts/preprocessing/prepare_pretreat_validation.py:437` | 'external' = external sites, folders, dependencies or a different study; not a claim about the 84 | python scripts/prepare_pretreat_validation.py --input-dir "Raw Data/PRETREAT" --output-dir data/external_validation |
| `scripts/preprocessing/prepare_pretreat_validation.py:440` | 'external' = external sites, folders, dependencies or a different study; not a claim about the 84 | python scripts/prepare_pretreat_validation.py --input-dir "Raw Data/PRETREAT" --output-dir data/external_validation --setup-nnunet |
| `scripts/preprocessing/prepare_pretreat_validation.py:447` | 'external' = external sites, folders, dependencies or a different study; not a claim about the 84 | parser.add_argument('--output-dir', type=str, default='data/external_validation', |
| `scripts/preprocessing/prepare_pretreat_validation.py:485` | 'external' = external sites, folders, dependencies or a different study; not a claim about the 84 | logger.info("PRETREAT-METSTOBRAIN-MASKS → External Validation Set") |
| `scripts/preprocessing/prepare_pretreat_validation.py:605` | 'external' = external sites, folders, dependencies or a different study; not a claim about the 84 | logger.info(f"    --output-dir results/external_validation") |
| `scripts/preprocessing/prepare_pretreat_validation.py:608` | 'external' = external sites, folders, dependencies or a different study; not a claim about the 84 | logger.info(f"    --pred-dir results/external_validation \\") |
| `scripts/preprocessing/prepare_proteas_validation.py:2` | 'external' = external sites, folders, dependencies or a different study; not a claim about the 84 | Download and preprocess PROTEAS brain metastasis dataset for external validation. |
| `scripts/preprocessing/prepare_proteas_validation.py:35` | 'external' = external sites, folders, dependencies or a different study; not a claim about the 84 | OUTPUT_DIR = PROJECT_ROOT / "data" / "external_validation_proteas" / "test" |
| `src/jannus/__init__.py:34` | 'external' = external sites, folders, dependencies or a different study; not a claim about the 84 | kept at the top level so external sites have one obvious import. |
| `src/jannus/api/pdf_report.py:6` | 'external' = external sites, folders, dependencies or a different study; not a claim about the 84 | Uses reportlab for PDF generation (lightweight, no external dependencies). |
| `src/jannus/cli/main.py:3` | 'external' = external sites, folders, dependencies or a different study; not a claim about the 84 | The intended path for an external validation site, in order: |
| `src/jannus/data/validation.py:1` | 'external' = external sites, folders, dependencies or a different study; not a claim about the 84 | """Pre-flight dataset QC for external validation sites. |
| `src/jannus/evaluation/harness.py:3` | 'external' = external sites, folders, dependencies or a different study; not a claim about the 84 | Wraps :mod:`jannus.evaluation.metrics` into the thing an external site actually |
| `src/jannus/evaluation/report.py:98` | 'external' = external sites, folders, dependencies or a different study; not a claim about the 84 | add(f"# JANNUS external validation report — {site_name}") |
| `src/jannus/segmentation/detection.py:20` | 'external' = external sites, folders, dependencies or a different study; not a claim about the 84 | nnDetection is an external heavyweight dependency whose dataset fingerprint |
| `src/jannus/segmentation/model_registry.py:19` | 'external' = external sites, folders, dependencies or a different study; not a claim about the 84 | Handles copying externally-trained models into the project, |
| `src/jannus/segmentation/model_registry.py:48` | 'external' = external sites, folders, dependencies or a different study; not a claim about the 84 | source_path: Path to the externally-trained .pth file |
| `src/jannus/segmentation/nnunet_raw_dataset.py:247` | cohort word refers to a different split (per-fold, internal-val, test name, other study) | comparable to nnU-Net 3D's on the same held-out cases. |
| `tests/unit/test_metrics.py:65` | cohort word refers to a different split (per-fold, internal-val, test name, other study) | def test_disjoint_lesions_do_not_match(self, empty): |
| `tests/unit/test_packaging.py:61` | 'external' = external sites, folders, dependencies or a different study; not a claim about the 84 | that looks live but is inert misleads an external site into tuning values |
