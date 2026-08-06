# Model card — JANNUS 1.50

Following the model-card conventions of Mitchell et al. (2019), adapted for a medical
imaging model under regulatory development.

---

## Model details

**Name.** JANNUS brain metastasis segmentation pipeline
**Version.** 1.50.0 — pipeline revision `7base-stackerv2-r1`
**Type.** Supervised 3D semantic segmentation; seven-model ensemble fused by a learned
stacking meta-classifier.
**Developer.** Clark Kirby, University of Arkansas.
**License.** Code MIT; weights under separate terms.

**Architecture.** Four co-registered MRI sequences → seven base models, each producing a
full-volume probability map → a 9-channel feature cube (7 predictions + voxel-wise
variance + max−min range) → `StackingClassifierV2` (~1.36 M parameters; residual 3D CNN
with squeeze-excitation attention and a multi-scale branch, applied as 32³ sliding windows
at 0.5 overlap) → threshold 0.55 → morphological cleanup → binary mask.

Base models: nnU-Net 3D full-res, nnU-Net 2D, four `LightweightUNet3D` variants at patch
sizes 8/12/24/36, and SwinUNETR at 96³ ROI.

The variance and range channels give the meta-learner explicit access to base-model
disagreement, which is informative about boundary uncertainty.

---

## Intended use

**Intended.** Retrospective research on brain-metastasis segmentation; methods comparison;
external validation studies; volumetric measurement research under RANO-BM.

**Proposed Indication for Use (not cleared).**

> Known brain metastases on T1 post-contrast and FLAIR MRI of adult patients, longest-axis
> lesion diameter ≥ 10 mm. Output: per-voxel binary segmentation masks for radiologist
> review and longitudinal volumetric measurement.

**Out of scope.**

- Any clinical decision-making. JANNUS is **not a medical device** and is not cleared or
  approved by any regulatory body.
- Autonomous interpretation without radiologist review.
- Primary brain tumours (glioma, meningioma), which the model was not trained on.
- Paediatric patients — training data is adult.
- Detection of lesions below ~5 mm, where sensitivity is not clinically useful.
- Post-treatment response assessment as a standalone determination.

---

## Training data

Multi-institutional brain-metastasis MRI with expert segmentations, including the
Stanford BrainMetShare cohort and the UCSF Brain Metastases Stereotactic Radiosurgery
dataset. All four sequences co-registered; per-channel z-score normalisation.

Not characterised in the training set, and therefore a source of unquantified bias:
patient demographics (age, sex, race, ethnicity), primary cancer distribution, scanner
vendor and field-strength distribution, and geographic origin. This is a real limitation
for a model intended for multi-site use, and is a primary reason external validation
matters.

---

## Evaluation data

84-case held-out cohort, disjoint from training at the patient level.

Confidence intervals are 1000-resample bootstrap over per-case values. Lesion matching is
greedy by IoU with a 0.1 threshold — deliberately permissive, because for a 4 mm
metastasis a stricter threshold measures annotator boundary disagreement more than it
measures detection.

---

## Quantitative results

### All annotated lesions

| Metric | Value (95% CI) |
|---|---|
| Voxel Dice | 0.7858 |
| Lesion-wise sensitivity | 0.810 (0.758–0.858) |
| Lesion-wise Dice (matched) | 0.791 |
| Precision | 0.806 |
| False positives / case | 1.68 (1.26–2.12) |
| HD95 | 18.36 mm (12.5–25.0) |
| Mean surface distance | 3.53 mm (2.5–4.9) |

### RANO-BM measurable disease (≥ 10 mm)

| Metric | JANNUS | Neosoma K252922 | FDA threshold | Result |
|---|---|---|---|---|
| Lesion-wise sensitivity | **0.943** (0.90–0.98) | 0.90 | ≥ 0.85 | pass |
| Lesion-wise Dice (matched) | 0.835 | 0.86 | ≥ 0.70 | pass |
| False positives / case | 1.61 | 0.57 | ≤ 5 | pass |
| HD95 | 17.26 mm | 1.78 mm | ≤ 2.94 mm | **fail** |
| Mean surface distance | 3.89 mm | 0.36 mm | ≤ 0.66 mm | **fail** |
| Cohen's κ (measurable classification) | **0.867** | n/r | ≥ 0.85 | pass |
| SoLD relative error | **1.84 %** | n/r | ≤ 10 % | pass |

Predicate values are as published for that device on its own cohort. This is context, not
a paired comparison — no head-to-head study has been run.

### Stratified by lesion size

| Longest axis | GT lesions | Sensitivity | FP/case | Mean Dice |
|---|---|---|---|---|
| < 3 mm | 33 | 0.061 | 0.54 | 0.53 |
| 3–5 mm | 82 | 0.439 | 0.54 | 0.64 |
| 5–10 mm | 304 | 0.773 | 0.46 | 0.72 |
| 10–20 mm | 144 | 0.903 | 0.12 | 0.81 |
| > 20 mm | 73 | 0.986 | 0.02 | 0.87 |

Performance is strongly size-dependent. **Any headline sensitivity figure is really a
statement about the lesion-size distribution of the cohort it was measured on.** Compare
the stratified table before comparing headline numbers across sites.

---

## Limitations

**HD95 and MSD fail the FDA thresholds by a wide margin.** These are surface-distance
metrics computed over the whole volume, and they are dominated by isolated false-positive
components far from any true lesion: a single spurious 3-voxel component 40 mm away drags
the 95th-percentile surface distance enormously, regardless of how well the real lesions
were segmented. The lesion-wise metrics, which are what a radiologist experiences, are
substantially better. Closing this gap requires either new imaging modalities (DWI/ADC as
additional input channels) or consensus radiologist re-annotation — both data-side
investments. Eleven stacker configurations across seven rounds did not move it; see
`docs/research/stacking_architectures_explored.md`.

**Sub-3 mm sensitivity is 0.061** — effectively zero. This is at the floor of MRI
resolution and inter-annotator agreement, and is why the proposed IFU scopes to ≥ 10 mm.

**Single-reader ground truth.** The development cohort was not consensus-annotated, so
reported Dice conflates model error with annotator variability, and no inter-rater ceiling
is established.

**nnU-Net 2D uses fold 0 only.** Folds 1–4 stopped early without a final checkpoint, so
that base model is weaker than a full 5-fold ensemble would be.

**No prospective validation.** All results are retrospective on a single held-out cohort.

**Demographic performance is uncharacterised.** No subgroup analysis by age, sex, race,
ethnicity or primary cancer type has been performed, so differential performance across
patient groups is unknown. This is a genuine equity gap.

**Distribution shift.** Post-treatment imaging (post-radiosurgery change, resection
cavities, treatment-related enhancement) is under-represented in training. Contrast dose
and timing differences move enhancement and therefore sensitivity.

---

## Ethical considerations

A false negative in this task means a missed metastasis. Whatever the aggregate metrics
say, the model must not be used to rule out disease, and its output requires radiologist
review. False positives carry a real cost too — unnecessary follow-up imaging, patient
anxiety, and at 1.68 per case they are frequent enough to matter for workflow.

Because no demographic subgroup analysis exists, equitable performance across patient
populations cannot currently be claimed.

---

## Reproducibility

Every run writes a provenance manifest recording the JANNUS version, pipeline revision,
git commit, package versions, hardware, the effective configuration including the
operating point, achieved determinism, and the SHA-256 of every checkpoint loaded. Two
sites can confirm they ran identical weights by comparing those hashes.

The operating threshold is **0.55**. (v1.40 and earlier silently used 0.5 while publishing
figures generated at 0.55 — see [CHANGELOG.md](../CHANGELOG.md).)

---

## Regulatory status

Not cleared, not approved, not a medical device. A 510(k) pathway is under exploration
with Neosoma Brain Mets (K252922, cleared December 2025) as the candidate predicate. Steps
outstanding: IFU finalisation, IRB-approved sequestered pivotal cohort (≥ 70 cases),
consensus neuroradiologist reads, and an FDA pre-submission meeting.

See [INTENDED_USE.md](intended_use.md).
