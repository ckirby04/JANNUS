# BrainMetScan — Intended Use Statement

**Product Name:** BrainMetScan (segmentation engine: JANNUS v1.4)
**Version:** 1.4.0
**Classification:** Software as a Medical Device (SaMD) — Class II
**Predicate Device:** Neosoma Brain Mets — FDA 510(k) **K252922** (cleared December 2025)
**Date:** April 2026

---

## 1. Intended Use

BrainMetScan is an AI-powered software tool intended to assist qualified neuroradiologists, radiation oncologists, and clinical-trial radiology readers in the **detection, volumetric segmentation, and longitudinal measurement of measurable brain metastases** on multi-sequence contrast-enhanced MRI scans.

The software analyzes four standard MRI sequences (T1 pre-contrast, T1 post-gadolinium, FLAIR, and T2-weighted) and produces:

1. **Automated segmentation masks** for suspected metastatic lesions
2. **Per-lesion volumetric measurements** (volume in mm³, longest diameter in mm)
3. **RANO-BM-aligned measurements** including Sum of Longest Diameters (SoLD) for treatment-response classification (CR / PR / SD / PD)
4. **Longitudinal comparison** between baseline and follow-up scans, with measurable-disease classification (Cohen's κ vs. radiologist read) and per-lesion delta tracking
5. **Structured clinical narrative** contextualized to the case (lesion count, size distribution, spatial distribution, treatment-decision framing)

## 2. Indications for Use

BrainMetScan is indicated for use as a **computer-aided detection and quantification (CADe/CADx)** tool for:

- **Detecting and delineating measurable brain metastases (≥10 mm longest axis per RANO-BM)** in adult patients with known systemic malignancy on contrast-enhanced MRI
- **Quantifying lesion volumes and longest diameters** for treatment planning, surgical/SRS targeting, and clinical-trial enrollment review
- **Computing Sum of Longest Diameters (SoLD)** and per-lesion deltas across sequential MRI scans to support RANO-BM treatment-response classification
- **Reducing inter-reader variability** in lesion measurement and counting in clinical-trial workflows

The system is intended as an **assistive tool**: all outputs are reviewed and finalized by a qualified clinician before they enter the medical record or trial CRF.

## 3. Target User Population

- **Primary users:** Board-certified neuroradiologists, diagnostic radiologists, radiation oncologists, and clinical-trial radiology readers (iCRO trial radiologists)
- **Secondary users:** Neurosurgeons, medical oncologists, and clinical-trial coordinators reviewing AI-assisted findings under radiologist supervision
- All users must have training in brain MRI interpretation, familiarity with brain-metastasis imaging patterns, and working knowledge of RANO-BM measurement criteria

## 4. Target Patient Population

- Adult patients (≥18 years) undergoing brain MRI for known or suspected brain metastases
- Patients with primary cancers known to metastasize to the brain (lung, breast, melanoma, renal cell carcinoma, colorectal, etc.)
- Patients enrolled in oncology clinical trials with a brain-metastasis cohort or stratification
- Patients undergoing longitudinal treatment-response monitoring for established brain metastases

## 5. Clinical Environment

- Hospital radiology departments and outpatient imaging centers (PACS-integrated workflows)
- Academic medical centers and clinical-trial sites
- Imaging Contract Research Organizations (iCROs) running pharma-sponsored CNS-metastasis trials
- Neuro-oncology and radiation-oncology clinics

## 6. Input Requirements

| Sequence | Required | Typical Parameters |
|----------|----------|--------------------|
| T1 pre-contrast | Yes | 3D volumetric, ≤1.5 mm slice thickness |
| T1 post-gadolinium | Yes | 3D volumetric, ≤1.5 mm slice thickness |
| FLAIR | Yes | 3D or 2D, ≤3 mm slice thickness |
| T2-weighted | Yes | 3D or 2D, ≤3 mm slice thickness |

**Supported formats:** DICOM, NIfTI
**Supported field strengths:** 1.5 T and 3.0 T MRI scanners
**Supported vendors:** Scanner-agnostic (validated on Siemens; multi-vendor validation pending — see Limitations)
**Voxel spacing assumption:** Internal pipeline expects ~1 mm isotropic; non-isotropic inputs are resampled before inference.

## 7. Contraindications

BrainMetScan should **NOT** be used:

- As a standalone diagnostic tool without qualified clinician review
- For primary brain tumors (glioma, meningioma, lymphoma) — the model is trained specifically on metastatic lesions and is contraindicated for primary CNS tumor segmentation
- On pediatric patients (<18 years)
- On MRI scans without gadolinium-based contrast enhancement
- On scans with significant motion artifacts, susceptibility artifacts, or incomplete sequence coverage
- For treatment-planning decisions without independent radiologist confirmation
- **As a screening tool for occult / sub-clinical brain metastases** — see size-floor limitation in §8

## 8. Limitations

1. **Research use only at this time** — BrainMetScan has not yet been cleared by the FDA, CE-marked, or approved by any regulatory body. The performance profile below targets a 510(k) submission with the indication scoped to measurable disease (§2).
2. **Size-floor limitation (clinically critical).** Detection sensitivity drops sharply below 5 mm:
   - **<3 mm: 6.1 % lesion sensitivity** (effectively at noise floor — at the intersection of scanner resolution and annotator variability)
   - 3–5 mm: 43.9 % lesion sensitivity
   - 5–10 mm: 77.3 % lesion sensitivity
   - 10–20 mm: 90.3 % lesion sensitivity
   - ≥20 mm: 98.6 % lesion sensitivity

   The Indications for Use are scoped to **measurable disease (≥10 mm)** for this reason. The system is not validated as a small-lesion screening tool.
3. **Boundary-precision gap.** On the measurable-disease scope, the system reaches 95 % HD95 confidence interval of 10.4–25.1 mm, which exceeds the FDA threshold of ≤2.94 mm achieved by the predicate device. This is a base-model limitation (every individual base in the ensemble has HD95 in the 18–35 mm range), and is the primary remaining engineering target. Closing it requires either per-base retraining with explicit boundary losses, anatomical-prior input channels, or DWI/ADC sequences. The current submission strategy treats this as a defensible trade under a measurable-disease IFU; it is **not** a configuration suitable for surgical-margin or stereotactic-radiosurgery target-volume definition without radiologist boundary review.
4. **Single-institution training.** The current production model is trained on 566 cases from Stanford BrainMetShare and UCSF-BMSR. Independent multi-institutional validation is required before clinical deployment outside those institutions.
5. **Atlas-registered domain shift.** The system was trained on native-geometry MRI. Performance on atlas-registered (SRI24) data drops sharply (Dice 17–20 % on PRETREAT/PROTEAS); domain-adaptation fine-tuning on ~200–400 atlas-registered cases is required for use on data distributed in BraTS-style format.
6. **Leptomeningeal disease.** Not designed to detect or measure leptomeningeal metastatic disease; a separate workflow is required.
7. **Post-surgical cavities.** May produce false positives in post-surgical resection cavities; radiologist review is required for post-resection cases.
8. **Radiation necrosis vs. viable tumor.** The system cannot distinguish radiation necrosis from recurrent viable tumor on conventional MRI alone (a known MRI-modality limitation, not specific to this software).
9. **Scanner / protocol generalization.** Performance may vary across MRI scanner vendors, field strengths, and acquisition protocols not represented in the training data.

## 9. Performance Summary

All metrics evaluated on **84 held-out cases** from UCSF Brain Metastases. Production model: **JANNUS v1.4 — tuned 7-model V2 stacking classifier** (4 LightweightUNet3D patch variants + nnU-Net 3D + nnU-Net 2D + SwinUNETR, fused by StackingClassifierV2: ~1.36 M parameters with squeeze-excitation attention and a multi-scale branch).

### Whole-volume metrics (any-size lesions)

| Metric | Value | Notes |
|--------|-------|-------|
| Voxel-wise Dice | 78.6 % | Threshold 0.55, std 0.154 |
| Voxel-wise Sensitivity | 78.2 % | Whole-volume |
| Voxel-wise Precision | 80.6 % | Whole-volume |
| Specificity | 99.99 % | Voxel-level |
| FPs / case | 1.68 | Lesion-level |
| Tiny-lesion Dice (<100 vox) | 40.3 % | +8.5 pt over v1.3 baseline |

### RANO-BM measurable-disease scope (≥10 mm longest axis) — **submission-scope metrics**

| Metric | Value (95 % CI) | Predicate (Neosoma K252922) | FDA Threshold | Status |
|---|---|---|---|---|
| Lesion-wise Sensitivity | **0.943** (0.900 – 0.977) | 0.90 (0.87 – 0.94) | ≥ 0.85 | ✓ exceeds predicate |
| Lesion-wise Dice (matched) | 0.835 (0.814 – 0.853) | 0.86 (0.83 – 0.89) | ≥ 0.70 | ≈ near predicate |
| FPs / case | 1.61 (1.19 – 2.05) | 0.57 (0.35 – 0.80) | ≤ 5 | ✓ passes FDA |
| HD95 (mm) | 17.26 (10.4 – 25.1) | 1.78 (1.02 – 2.54) | ≤ 2.94 | ✗ structural gap (§8.3) |
| MSD (mm) | 3.89 (2.34 – 5.65) | 0.36 (0.16 – 0.56) | ≤ 0.66 | ✗ structural gap (§8.3) |
| Cohen's κ (measurable classification) | **0.867** | — | ≥ 0.85 (RANO-BM) | ✓ passes RANO-BM |
| SoLD relative error | **1.84 %** | — | ≤ 10 % (RANO-BM) | ✓ 5× below threshold |

All confidence intervals computed via 1000-resample bootstrap on the 84-case held-out cohort.

### Training Configuration

| | |
|---|---|
| Training cases | 566 (Stanford BrainMetShare + UCSF-BMSR), native geometry |
| Held-out evaluation | 84 cases (subject-level split) |
| Stacker training | BCE + (1 − Dice), AdamW lr 3e-4, 5-epoch warmup → cosine, dropout 0.2, weight_decay 1e-4, 3D flip/rotate/intensity augmentation |
| Inference | Sliding-window 32³ patches, 50 % overlap, threshold 0.55, min_component_size 0 |

*Detailed validation results, per-case breakdowns, and FDA-aligned size-stratified metrics are available in `docs/stacking_v4_vs_v5_writeup.md` and `model/evaluation_results/`.*

## 10. Regulatory Pathway

BrainMetScan is being developed under the FDA's **predetermined change control plan (PCCP)** framework for AI/ML-based SaMD:

- **Target classification:** 510(k) Class II under a measurable-disease Indication for Use
- **Predicate device:** Neosoma Brain Mets — **K252922** (cleared December 2025)
- **Substantial-equivalence argument:** Same intended use (CADe/CADx for brain metastases on multi-sequence MRI), same imaging modality, same patient population. Differs in scope (BrainMetScan IFU is restricted to measurable disease ≥10 mm vs. Neosoma's broader detection claim) and in performance profile (BrainMetScan exceeds predicate on lesion sensitivity at the measurable-disease scope; trades boundary precision for broader detection coverage). The narrower IFU is the basis for the substantial-equivalence claim under §513(i)(1)(A) of the FD&C Act.
- **Quality Management System:** ISO 13485:2016 (in development)
- **Risk Management:** ISO 14971:2019 (see `docs/risk_analysis.md`)
- **Software Lifecycle:** IEC 62304:2006+AMD1:2015
- **Pre-submission (Q-Sub) plan:** discuss size-stratified evaluation framework with FDA, negotiate the measurable-disease cut-off, present pre-specified Statistical Analysis Plan for HD95 on the submission scope.

### Required prior to submission

- [ ] Consensus neuroradiologist reads on the 84-case dev cohort (3-reader majority; current ground truth is single-radiologist annotation)
- [ ] Sequestered pivotal cohort (target n ≥ 150, multi-site)
- [ ] Multi-site external validation on independent native-geometry data
- [ ] Domain-adaptation fine-tuning for atlas-registered data (PRETREAT / BraTS-METS), if those datasets are part of the pivotal cohort
- [ ] Pre-submission (Q-Sub) meeting with FDA
- [ ] Cybersecurity and software-bill-of-materials documentation per FDA premarket cybersecurity guidance (Sep 2023)

## 11. Post-Market Surveillance Plan

Upon regulatory clearance, BrainMetScan will implement:

- Continuous performance monitoring on production cases (lesion sensitivity, FPs/case, SoLD agreement)
- Automated drift detection comparing production metrics to validation baselines, with quarterly review
- Quarterly performance reports to deployed sites
- Adverse event reporting per 21 CFR Part 803
- User-feedback collection and analysis (radiologist-flagged false negatives / false positives, RANO-BM disagreement cases)
- PCCP-governed model updates: any change that does not alter the cleared IFU or the substantial-equivalence basis is deployed under the PCCP; any change that does triggers a 510(k) supplement.

## 12. Cybersecurity and Privacy

- All inference is performed locally (on-premise or in customer-controlled cloud); no patient imaging data leaves the deployment environment.
- The optional RAG literature-retrieval pipeline operates against a local PubMed-derived corpus and does not transmit case data externally.
- DICOM ingestion follows ATA / DICOM PS3.15 security profiles; PHI handling is documented in the QMS.

---

**DISCLAIMER:** This document describes the intended future use of BrainMetScan. The software is currently designated **Research Use Only (RUO)** and must not be used for clinical decision-making until FDA clearance is obtained. Performance numbers above represent current internal-validation results on a single-institution held-out cohort and are not yet FDA-substantiated.
