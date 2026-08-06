# JANNUS — Risk Analysis

**Document Type:** Risk Management Report (per ISO 14971:2019)
**Product:** JANNUS v1.50.0 (formerly BrainMetScan)
**Classification:** SaMD Class II (anticipated)
**Date:** February 2026; **revised 5 August 2026**
**Status:** Draft — Research Use Only

---

> ## ⚠️ Revision notice — 5 August 2026
>
> A security and privacy audit conducted during the v1.50 release found that **four
> mitigations in this document were marked implemented when they were absent or only
> partially present in the code**:
>
> | ID | Claim | Actual state through v1.40 |
> |---|---|---|
> | M-013 | DICOM handler validates PatientID consistency | Never implemented — `PatientID` is not read |
> | M-021 | No PHI in system logs — only anonymized case IDs | No anonymization existed; raw `case_id` was logged |
> | M-022 | Temporary file cleanup after processing | Mask-download handler leaked temp dirs containing patient masks |
> | M-023 | HTTPS/TLS encryption for all API communication | No TLS configuration, implementation or guidance exists |
>
> M-021 and M-022 are implemented as of v1.50. **M-013 and M-023 remain open** and are
> marked as such below.
>
> If a prior revision of this document accompanied an IRB submission, an FDA
> pre-submission, or any external representation of the system's controls, those
> statements were inaccurate and the discrepancy should be disclosed to the receiving
> body. Under ISO 14971 a risk control that is documented but not implemented is a
> nonconformity, not a documentation error.
>
> The residual-risk ratings below have been raised accordingly. Corresponding code-level
> detail is in [`../CHANGELOG.md`](../CHANGELOG.md) and
> [`../SECURITY.md`](../SECURITY.md).

---

## 1. Scope

This document identifies, evaluates, and mitigates risks associated with the BrainMetScan brain metastasis segmentation system. It follows the framework of ISO 14971:2019 "Medical devices — Application of risk management to medical devices."

## 2. Risk Acceptability Criteria

| Severity | Probability: Frequent | Probability: Probable | Probability: Occasional | Probability: Remote | Probability: Improbable |
|----------|----------------------|----------------------|------------------------|--------------------|-----------------------|
| **Catastrophic** | Unacceptable | Unacceptable | Unacceptable | ALARP | ALARP |
| **Critical** | Unacceptable | Unacceptable | ALARP | ALARP | Acceptable |
| **Serious** | Unacceptable | ALARP | ALARP | Acceptable | Acceptable |
| **Minor** | ALARP | ALARP | Acceptable | Acceptable | Acceptable |
| **Negligible** | Acceptable | Acceptable | Acceptable | Acceptable | Acceptable |

**ALARP** = As Low As Reasonably Practicable (requires mitigation)

---

## 3. Hazard Identification and Risk Assessment

### H-001: False Negative — Missed Brain Metastasis

| Attribute | Value |
|-----------|-------|
| **Hazard** | AI fails to detect an existing brain metastasis |
| **Hazardous Situation** | Clinician relies on AI output and does not independently identify the lesion |
| **Harm** | Delayed treatment; disease progression; potential neurological decline |
| **Severity** | Critical |
| **Pre-mitigation Probability** | Occasional |
| **Pre-mitigation Risk** | ALARP |

**Mitigations:**
1. Clear labeling as **CADe tool** — not a replacement for radiologist review (M-001)
2. Intended use requires physician review of all scans regardless of AI findings (M-002)
3. Multi-scale ensemble architecture improves sensitivity to 92.2% lesion detection (M-003)
4. System reports confidence scores for each detection — low-confidence regions flagged (M-004)
5. Training on diverse lesion sizes including tiny lesions (<5mm) (M-005)

**Post-mitigation Probability:** Remote
**Post-mitigation Risk:** ALARP — acceptable given physician oversight requirement

---

### H-002: False Positive — Incorrectly Identified Metastasis

| Attribute | Value |
|-----------|-------|
| **Hazard** | AI identifies a non-metastatic structure as a brain metastasis |
| **Hazardous Situation** | Clinician acts on false positive without verification |
| **Harm** | Unnecessary biopsy, surgery, or radiation; patient anxiety; healthcare costs |
| **Severity** | Serious |
| **Pre-mitigation Probability** | Probable |
| **Pre-mitigation Risk** | ALARP |

**Mitigations:**
1. Postprocessing pipeline removes small spurious detections (M-006)
2. Per-lesion confidence scores allow filtering at adjustable thresholds (M-007)
3. DICOM-SEG output enables side-by-side review in PACS viewers (M-008)
4. Intended use requires radiologist confirmation before any intervention (M-002)

**Post-mitigation Probability:** Occasional
**Post-mitigation Risk:** ALARP — acceptable given physician confirmation requirement

---

### H-003: Incorrect Volume Measurement

| Attribute | Value |
|-----------|-------|
| **Hazard** | Segmentation boundary is inaccurate, leading to incorrect volume measurement |
| **Hazardous Situation** | Treatment response assessed incorrectly based on volume change |
| **Harm** | Premature treatment discontinuation or unnecessary treatment escalation |
| **Severity** | Serious |
| **Pre-mitigation Probability** | Probable |
| **Pre-mitigation Risk** | ALARP |

**Mitigations:**
1. RECIST 1.1 response classification uses diameter-based criteria with standard thresholds (M-009)
2. Longitudinal comparison shows matched lesion pairs with volume change percentages (M-010)
3. Probability maps available for manual boundary adjustment (M-011)
4. Measurement precision validated against expert annotations (M-012)

**Post-mitigation Probability:** Occasional
**Post-mitigation Risk:** ALARP — acceptable with physician review of measurements

---

### H-004: Wrong Patient / Data Mismatch

| Attribute | Value |
|-----------|-------|
| **Hazard** | Input data from different patients is processed together, or results are attributed to wrong patient |
| **Hazardous Situation** | Clinician receives segmentation results for wrong patient |
| **Harm** | Incorrect treatment decisions for both patients involved |
| **Severity** | Critical |
| **Pre-mitigation Probability** | Remote |
| **Pre-mitigation Risk** | ALARP |

**Mitigations:**
1. ~~DICOM handler validates PatientID consistency within uploaded series (M-013)~~
   — **NOT IMPLEMENTED.** Corrected 2026-08-05. `PatientID` (0010,0020) is never read
   by `jannus/api/dicom_handler.py`; the handler groups by `SeriesInstanceUID` only.
   This mitigation was listed as implemented in v1.40 and earlier. It is not.
   **Status: open. Do not claim this control.**
2. Case ID tracked through the entire pipeline (M-014) — implemented.
3. API audit logging records all input files and timestamps (M-015) — implemented.
4. Database persistence enables traceability from result back to input (M-016)
   — implemented.
5. **NEW (v1.50)** Every inference run writes a provenance manifest recording the
   SHA-256 of each checkpoint loaded, the effective configuration and the code
   revision, enabling result-to-input-and-model traceability (M-013a).

**Post-mitigation Probability:** Remote (raised from Improbable — M-013 is not in place)
**Post-mitigation Risk:** ALARP

---

### H-005: System Unavailability

| Attribute | Value |
|-----------|-------|
| **Hazard** | System crashes or becomes unavailable during clinical workflow |
| **Hazardous Situation** | Clinician cannot access AI results when needed for time-sensitive decision |
| **Harm** | Treatment delay; clinician must proceed without AI assistance |
| **Severity** | Minor |
| **Pre-mitigation Probability** | Occasional |
| **Pre-mitigation Risk** | Acceptable |

**Mitigations:**
1. Docker containerization with automatic restart policy (M-017)
2. Health check endpoint for monitoring (M-018)
3. System designed as supplementary tool — clinical workflow not dependent on AI availability (M-019)

**Post-mitigation Risk:** Acceptable

---

### H-006: Data Privacy Breach

| Attribute | Value |
|-----------|-------|
| **Hazard** | Patient imaging data or PHI is exposed to unauthorized parties |
| **Hazardous Situation** | Data transmitted without encryption or stored insecurely |
| **Harm** | HIPAA violation; patient privacy compromise; legal liability |
| **Severity** | Critical |
| **Pre-mitigation Probability** | Remote |
| **Pre-mitigation Risk** | ALARP |

**Mitigations:**
1. API key authentication with rate limiting (M-020) — implemented, and **strengthened
   in v1.50**: authentication now defaults to required (it defaulted to *disabled*
   through v1.40), anonymous callers are denied by `check_permission`, the mask-download
   endpoint gained its missing authentication dependency, and anonymous requests are
   rate-limited by client IP (they previously bypassed the limiter entirely).
2. No PHI stored in system logs — only anonymized case IDs (M-021)
   — **WAS NOT IMPLEMENTED through v1.40.** No anonymization existed; the raw `case_id`
   was written to logs, the database and PDF headers. **Implemented in v1.50**: case
   identifiers are pseudonymised with a salted SHA-256 digest and a redaction filter
   strips identifier-shaped text. Note this protects the *logs*, not the data — JANNUS
   still performs no DICOM de-identification.
3. Temporary file cleanup after processing (M-022)
   — **PARTIALLY IMPLEMENTED through v1.40.** The mask-download handler used
   `tempfile.mkdtemp()` with no cleanup, leaking a directory containing a patient
   segmentation on every download. **Fixed in v1.50.**
4. ~~HTTPS/TLS encryption for all API communication (M-023)~~
   — **NOT IMPLEMENTED.** Corrected 2026-08-05. JANNUS provides no TLS configuration,
   no reverse proxy, and no deployment instructions for either. This is a *deployment
   requirement placed on the operating site*, not a control the software supplies, and
   must not be counted as an implemented mitigation. **Status: open.**
5. Audit logging for all data access events (M-024) — implemented.

**Post-mitigation Probability:** Remote (raised from Improbable — M-023 is not in place)
**Post-mitigation Risk:** ALARP, contingent on the deploying site terminating TLS

---

### H-007: Model Degradation Over Time

| Attribute | Value |
|-----------|-------|
| **Hazard** | Model performance degrades due to distribution shift (new scanners, protocols, patient populations) |
| **Hazardous Situation** | System produces increasing false negatives or false positives without detection |
| **Harm** | Systematic diagnostic errors across patient population |
| **Severity** | Critical |
| **Pre-mitigation Probability** | Probable (over long deployment) |
| **Pre-mitigation Risk** | Unacceptable |

**Mitigations:**
1. Performance monitoring via database analytics — detection rate tracked over time (M-025)
2. Benchmark suite for periodic revalidation (M-026)
3. Model registry with version tracking — rollback capability (M-027)
4. Post-market surveillance plan with quarterly performance reviews (M-028)
5. Multi-site validation planned before production deployment (M-029)

**Post-mitigation Probability:** Remote
**Post-mitigation Risk:** ALARP — acceptable with monitoring

---

### H-008: Incorrect Sequence Identification (DICOM)

| Attribute | Value |
|-----------|-------|
| **Hazard** | DICOM handler misidentifies MRI sequence type (e.g., FLAIR as T2) |
| **Hazardous Situation** | Model receives incorrectly ordered input channels |
| **Harm** | Degraded segmentation accuracy; false negatives or positives |
| **Severity** | Serious |
| **Pre-mitigation Probability** | Occasional |
| **Pre-mitigation Risk** | ALARP |

**Mitigations:**
1. Sequence identification uses multiple DICOM tags (SeriesDescription, ProtocolName, etc.) (M-030)
2. Heuristic matching with confidence reporting (M-031)
3. NIfTI upload path allows explicit sequence labeling by user (M-032)
4. Warning when sequence identification confidence is low (M-033)

**Post-mitigation Probability:** Remote
**Post-mitigation Risk:** Acceptable

---

## 4. Risk-Benefit Analysis

### Benefits
- Reduced inter-reader variability in lesion counting and measurement
- Faster lesion detection and volumetric analysis (seconds vs. minutes)
- Standardized RECIST 1.1 measurements for clinical trials
- Automated longitudinal tracking reduces manual tracking burden
- Potential to detect small lesions missed by visual inspection

### Residual Risks
All residual risks are at or below ALARP level, contingent on:
- System use within intended use conditions
- Qualified physician review of all AI findings
- Proper deployment with security controls (HTTPS, authentication)
- Ongoing performance monitoring

### Conclusion
The benefits of BrainMetScan outweigh the residual risks when used within the specified intended use by qualified clinicians with appropriate oversight.

---

## 5. Mitigation Traceability Matrix

| Mitigation ID | Description | Implemented | Verified |
|--------------|-------------|-------------|----------|
| M-001 | CADe labeling in UI and reports | Yes | - |
| M-002 | Physician review requirement in intended use | Yes | - |
| M-003 | Multi-scale ensemble architecture | Yes | Yes |
| M-004 | Per-lesion confidence scores | Yes | Yes |
| M-005 | Tiny lesion training data | Yes | Yes |
| M-006 | Small component removal postprocessing | Yes | Yes |
| M-007 | Adjustable confidence threshold | Yes | Yes |
| M-008 | DICOM-SEG output for PACS | Yes | - |
| M-009 | RECIST 1.1 response classification | Yes | Yes |
| M-010 | Longitudinal lesion matching | Yes | Yes |
| M-011 | Probability map download | Yes | Yes |
| M-012 | Validation against expert annotations | Partial | - |
| M-013 | DICOM PatientID consistency check | Yes | - |
| M-014 | Case ID tracking | Yes | Yes |
| M-015 | API audit logging | Yes | Yes |
| M-016 | Database persistence | Yes | Yes |
| M-017 | Docker auto-restart | Yes | - |
| M-018 | Health check endpoint | Yes | Yes |
| M-019 | Supplementary tool design | Yes | - |
| M-020 | API key authentication | Yes | Yes |
| M-021 | No PHI in logs | Yes | - |
| M-022 | Temp file cleanup | Yes | Yes |
| M-023 | HTTPS/TLS (deployment) | Config | - |
| M-024 | Audit logging | Yes | Yes |
| M-025 | Performance analytics | Yes | - |
| M-026 | Benchmark suite | Yes | Yes |
| M-027 | Model registry versioning | Yes | Yes |
| M-028 | Post-market surveillance plan | Documented | - |
| M-029 | Multi-site validation | Planned | - |
| M-030 | Multi-tag sequence identification | Yes | - |
| M-031 | Sequence ID confidence | Partial | - |
| M-032 | NIfTI explicit labeling | Yes | Yes |
| M-033 | Low confidence warning | Partial | - |

---

**Document Control:**
- Author: BrainMetScan Engineering Team
- Review Status: Draft
- Next Review: Upon initiation of regulatory submission process
