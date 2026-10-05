> **STATUS: SUPERSEDED (2026-10-05).** This experiment was not completed. JANNUS 1.50's published figures are invalid as performance estimates (see CLOSEOUT.md, tag `audit/1.50-defects`). Successor: JANNUS 1.60 (`G:\BrainMetShare\1.60`).

# Lesion-level detection evaluation of JANNUS v1.50

## What this is

Every JANNUS evaluation so far has graded segmentation: Dice, HD95, voxel sensitivity.
The realistic near-term clinical use is different: pointing a radiologist to where the
lesions are. This experiment grades JANNUS the way a computer-aided **detection** (CADe)
system is graded. Its main outputs are:

- lesion-wise sensitivity against false positives per scan (FROC);
- sensitivity at fixed operating points of 0.5, 1, 2 and 3 FP/scan;
- size-stratified results, which are the most important output, because small lesions are where a detection aid matters;
- scan-level hit rates;
- a characterisation of the false positives;
- patient-level bootstrap confidence intervals.

The current production setting (stacker threshold 0.55, every component kept) is the
baseline. The experiment asks whether a different voxel threshold combined with a
per-lesion confidence score gives a better detection operating point. It retrains nothing
and leaves the stacker unchanged.

### Isolation

- Library code is in `src/jannus/evaluation/detection/`. It imports from existing modules and modifies none.
- Everything else is in this directory.
- `jannus evaluate`, `report.md`, `provenance.json`, `weights.lock.json` and `docs/` are unchanged.
- The cached maps and component tables (`cache/`) are gitignored and can be regenerated.

## Read these first

| File | Contents |
|---|---|
| `PLAN.md` | Phase 0 recon and the approved plan |
| `DECISIONS.md` | Every methodological choice, with rationale and alternatives |
| `EXPERIMENT_LOG.md` | Dated, append-only record of every run and which split it used |
| `PRIOR_DEFECTS.md` | Index of defects found in existing code and figures (none fixed here) |
| `SPACING_DEFECT.md` | `preprocessed_256` misstates voxel spacing |
| `LEAKAGE_FINDING.md` | The "held-out" 84 overlap training; the clean/contaminated split |
| `configs/splits/SOURCES.md` | Where every split ID list came from |
| `results/<run_id>/REPORT.md` | The results |

## Baseline it is compared against

Prior figure: overall lesion sensitivity **0.810 at 1.68 FP/case** on the 84-case cohort
at threshold 0.55. (0.943 is sensitivity for **lesions ≥10 mm only**.) Those figures used
the 1 mm isotropic assumption (SPACING_DEFECT.md) and the leaked cohort (LEAKAGE_FINDING.md).
This experiment re-measures the baseline under its own conventions, side by side with the
selected operating point.

## Important caveats

- **This repo has no untouched test set.** The 84 scans were used to choose the 0.55 threshold and the stacker variant.
- **Training overlap.** 24 scans are in nnU-Net fold-0 training, and 33 of the 77 patients have other timepoints in the stacker pool.
- **Primary design.** Select on the 55 contaminated scans, then evaluate once on the 29 clean scans (28 patients).
- **"Clean" means clean with respect to nnU-Net and stacker training only; NOT clean for the patch models.** Recipe recovery in 1.60 found the legacy patch models trained on 77 of the 84 eval cases. The provenance of the synthetic BMS cases is still unknown.
- **Standalone FROC does not establish reader benefit.** A CADe claim needs a reader study.

## Reproduce from scratch

Requirements:

- the JANNUS environment (`pip install -e '.[segmentation]'`) plus matplotlib, which ships in the `demo` extra;
- a CUDA GPU (inference took about 1 minute per case on an RTX 3070 Ti);
- weights installed and verified (`jannus doctor` → READY);
- read-only access to `G:/BrainMetShare/Data` and `G:/BrainMetShare/Raw Data`. These paths are set in `configs/*.yaml`.

Run the steps from the repo root, in order. Every step resumes or refuses to overwrite.

```bash
# 0. Split lists (already committed; regenerating them is deterministic)
python experiments/lesion_detection/tools/build_leakage_splits.py --data-root G:/BrainMetShare/Data

# 1. Probability maps: stacker + 7 base maps per case, about 1.5 GPU-hours for 84 cases
python experiments/lesion_detection/run_inference.py --config experiments/lesion_detection/configs/inference.yaml

# 2. Reproduction gate: must PASS before anything else is read (DECISIONS D8)
python experiments/lesion_detection/repro_gate.py --config experiments/lesion_detection/configs/00_repro_gate.yaml

# 3. Lesion and component tables, all thresholds, rules and scorers (CPU; set workers in the config)
python experiments/lesion_detection/build_components.py --config experiments/lesion_detection/configs/components.yaml

# 4. PRIMARY: select on contaminated, then evaluate ONCE on clean
python experiments/lesion_detection/select_operating_point.py --config experiments/lesion_detection/configs/10_select_primary.yaml
python experiments/lesion_detection/evaluate.py --config experiments/lesion_detection/configs/20_evaluate_primary.yaml

# 5. SECONDARY: patient-grouped 2-fold over all 84
python experiments/lesion_detection/tools/make_2fold.py --seed 20261004
python experiments/lesion_detection/select_operating_point.py --config experiments/lesion_detection/configs/30_select_foldA.yaml
python experiments/lesion_detection/select_operating_point.py --config experiments/lesion_detection/configs/31_select_foldB.yaml
python experiments/lesion_detection/evaluate.py --config experiments/lesion_detection/configs/32_evaluate_foldB.yaml
python experiments/lesion_detection/evaluate.py --config experiments/lesion_detection/configs/33_evaluate_foldA.yaml
python experiments/lesion_detection/pool_2fold.py --folds experiments/lesion_detection/configs/32_evaluate_foldB.yaml experiments/lesion_detection/configs/33_evaluate_foldA.yaml

# 6. Supplementary: original vs BraTS-revised annotation
python experiments/lesion_detection/brats_side_analysis.py --config experiments/lesion_detection/configs/40_brats_side.yaml

# 7. Report
python experiments/lesion_detection/make_report.py --run experiments/lesion_detection/results/<run_id>

# Reproducibility check: re-evaluates into results/<run_id>_reproduce and compares metrics.json
python experiments/lesion_detection/evaluate.py --config experiments/lesion_detection/configs/20_evaluate_primary.yaml --reproduce
```

`<run_id>` is printed by step 4 and is `primary_<hash>`, where the hash covers the evaluation
config plus the frozen operating point.

## What the results mean

- **FROC.** Each point answers: "if the system flags every candidate scoring at or above τ, what fraction of real lesions does it flag, and how many false flags does it raise per scan?"
- **Fixed-FP rows.** These use the τ frozen on the tuning split. They are what a deployment would see, and the realised FP/scan on test shows whether the calibration transfers.
- **Size bins.** Bins use the equivalent spherical diameter of the ground-truth lesion. Every per-bin number carries its lesion count. Small bins are underpowered and say so.

## Future work (not done here)

- **Lesion-weighted training losses**, such as blob loss, which train for per-lesion recall rather than voxel overlap. If detection is the target use they are the natural next step, but they require retraining, which this experiment rules out.
- A **multi-reader, multi-case reader study**, which is required for any reader-benefit (CADe) claim.
- A sequestered external cohort, and fixing the prior defects listed in PRIOR_DEFECTS.md.

## Tests

```bash
python -m pytest tests/unit/test_detection_matching.py tests/unit/test_detection_scoring.py tests/unit/test_detection_metrics.py
```
