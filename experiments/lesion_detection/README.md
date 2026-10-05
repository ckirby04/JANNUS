# Lesion-level detection evaluation of JANNUS v1.50

**Status:** in progress. Phase 1 (lesion extraction and matching) is implemented; no
results yet. This README is completed in Phase 5 with full reproduction steps.

## What this is

Every JANNUS evaluation so far has graded segmentation: Dice, HD95, voxel sensitivity.
The realistic near-term clinical use is different: pointing a radiologist to where the
lesions are. This experiment grades JANNUS the way a computer-aided **detection** (CADe)
system is graded. That means lesion-wise sensitivity against false positives per scan
(FROC), stratified by lesion size, with patient-level bootstrap confidence intervals.

It is isolated from everything else in the repo:

- library code lives in `src/jannus/evaluation/detection/` and modifies no existing module;
- everything else lives in this directory;
- `jannus evaluate`, `report.md`, `provenance.json`, `weights.lock.json` and `docs/` are unchanged.

## Read these first

| File | Contents |
|---|---|
| `PLAN.md` | Phase 0 recon and the approved plan |
| `DECISIONS.md` | Every methodological choice, with rationale and alternatives |
| `EXPERIMENT_LOG.md` | Dated, append-only record of every run and which split it used |
| `SPACING_DEFECT.md` | The `preprocessed_256` geometry defect and what it affects |
| `configs/splits/SOURCES.md` | Where the split ID lists came from |

## Baseline it is compared against

Overall lesion sensitivity **0.810 at 1.68 FP/case** on the 84-case cohort at threshold
0.55. (0.943 is sensitivity for **lesions ≥10 mm only**.) These figures were computed
under the 1 mm isotropic assumption described in SPACING_DEFECT.md.

## Important caveat

This repo has **no untouched test set**. The 84-case cohort was previously used to choose
the 0.55 threshold and the stacker variant, and it overlaps base-model and stacker
training at the scan and patient level (DECISIONS.md, D5).

## Future work (not done here)

- **Lesion-weighted training losses**, such as blob loss, which train for per-lesion recall rather than voxel overlap. They are the natural next step if detection is the target use, but they require retraining, which this experiment rules out.
- A **reader study**. Standalone FROC does not establish reader benefit.

## Running the tests

```bash
python -m pytest tests/unit/test_detection_matching.py
```
