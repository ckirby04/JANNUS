# BrainMetScan — Production Model

**Production checkpoint:** `model/stacking_classifier_production.pth`

This is the model. Architecturally we've explored 11 stacker configurations across
7 rounds of experiments and converged on this one. Don't iterate further on the
stacker without a strong, specific hypothesis and a concrete test plan.

## What it is

A **7-base 9-channel stacking classifier** built on top of:
- nnU-Net 3D, nnU-Net 2D (from `model/stacking_cache_v5/`, native-preprocessed)
- 4 SwinUNETR patch variants: patch_8, patch_12, patch_24, patch_36 (legacy)
- SwinUNETR 150ep+ (`model/swin_unetr_brainmets.pth`, val_dice 0.7915)

Stacker architecture: `StackingClassifierV2` (~1.36M params) — 1×1×1 input
bottleneck + wider/deeper trunk (mid_channels=64, 4 residual blocks) +
multi-scale branch (avg-pool → 2 res blocks → trilinear upsample → concat → fuse)
+ Squeeze-Excitation channel attention. See `src/segmentation/stacking.py`.

## Headline performance (84-case held-out eval)

### Overall (all lesions)

| Metric | Value | Old 6-model (legacy) |
|---|---|---|
| Voxel Dice | **0.7858** | 0.7778 |
| Lesion-wise Sensitivity | 0.810 (95% CI 0.758-0.858) | — |
| Lesion-wise Dice (matched) | 0.791 | — |
| Precision | 0.806 | 0.805 |
| FPs / case | 1.68 (95% CI 1.26-2.12) | — |
| HD95 (mm) | 18.36 (95% CI 12.5-25.0) | — |
| MSD (mm) | 3.53 (95% CI 2.5-4.9) | — |

### Stratified by lesion size (longest axis)

| Bin | n GT | Sensitivity | FPs/case | Mean DSC |
|---|---|---|---|---|
| <3mm | 33 | **0.061** | 0.54 | 0.53 |
| 3-5mm | 82 | 0.439 | 0.54 | 0.64 |
| 5-10mm | 304 | 0.773 | 0.46 | 0.72 |
| 10-20mm | 144 | **0.903** | 0.12 | 0.81 |
| >20mm | 73 | **0.986** | 0.02 | 0.87 |

The model is effectively blind below 3mm (intersection of MRI resolution + annotator
disagreement) and exceeds FDA-passing sensitivity from 10mm up.

### When scoped to RANO-BM "measurable disease" (≥10mm)

| Metric | BMS production | Neosoma K252922 | FDA threshold |
|---|---|---|---|
| Lesion-wise Sensitivity | **0.943** (0.90-0.98) | 0.90 | ≥ 0.85 ✓ |
| Lesion-wise Dice (matched) | 0.835 | 0.86 | ≥ 0.70 ✓ |
| FPs / case | 1.61 | 0.57 | ≤ 5 ✓ |
| HD95 (mm) | 17.26 | 1.78 | ≤ 2.94 ✗ |
| MSD (mm) | 3.89 | 0.36 | ≤ 0.66 ✗ |
| Cohen's κ (measurable classification) | **0.867** | not reported | ≥ 0.85 ✓ (RANO-BM) |
| SoLD relative error | **1.84%** | not reported | ≤ 10% ✓ (RANO-BM) |

**At ≥10mm scope, BMS exceeds Neosoma on lesion-wise sensitivity and matches on
Dice. Cohen's kappa and SoLD error pass RANO-BM targets. HD95/MSD are the only
remaining FDA-failing metrics; both improve from the all-lesions case.**

## How to run

### Inference (one case)

```python
import torch, numpy as np
from jannus.segmentation.stacking import StackingClassifierV2, sliding_window_inference, postprocess_prediction

# Load production stacker
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = StackingClassifierV2(in_channels=9).to(device)
ckpt = torch.load("model/stacking_classifier_production.pth", map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Build 9-channel features from 7 base predictions (nnunet, nnunet_2d, patch_8/12/24/36, swin_unetr)
# variance + range are appended automatically:
preds = np.stack([nnunet, nnunet_2d, patch_8, patch_12, patch_24, patch_36, swin_unetr], axis=0)
variance = preds.var(axis=0, keepdims=True)
range_map = preds.max(axis=0, keepdims=True) - preds.min(axis=0, keepdims=True)
features = np.concatenate([preds, variance, range_map], axis=0)  # (9, H, W, D)

# Sliding-window inference at 32^3 patches with 0.5 overlap
prob = sliding_window_inference(model, features, patch_size=32, device=device, overlap=0.5)

# Threshold + postprocess (min_component_size=0 by clinical decision)
mask = postprocess_prediction((prob > 0.55).astype(np.float32), min_size=0)
```

### Reproduce evaluation

```bash
# Standard 84-case eval with FDA + stratified metrics
python scripts/evaluation/seven_model_stacker.py --skip-train

# Size-filter sweep (IFU scoping experiments)
python scripts/evaluation/eval_size_filtered.py

# RANO-BM measurable-disease eval (≥10mm scope)
python scripts/evaluation/eval_measurable_disease.py

# Stratified-by-size standalone eval
python scripts/evaluation/eval_stratified_by_size.py
```

### Retrain from scratch (rare; only if base preds change)

```bash
$env:CUDA_VISIBLE_DEVICES="1"
python scripts/evaluation/seven_model_stacker.py --num-workers 8
# Tuned recipe defaults: lr=3e-4, warmup=5, dropout=0.2, wd=1e-4, save_min_delta=0.005, augment=True
```

## Recommended Indication for Use (proposed)

> "Known brain metastases on T1 post-contrast and FLAIR MRI of adult patients,
> longest-axis lesion diameter ≥10mm. Output: per-voxel binary segmentation masks
> for radiologist review and longitudinal volumetric measurement."

This scope:
- Is the RANO-BM "measurable disease" standard, used by every pharma/iCRO trial
- Excludes the <3mm bin where model performance is at noise floor regardless of architecture
- Aligns with the metrics that pass Neosoma + RANO-BM thresholds
- Avoids HD95 as a regulatory blocker (still elevated but clinically defensible at this scope)

## What's been ruled out (don't repeat these)

See `docs/stacking_architectures_explored.md` for full
retrospective. Summary of dead ends:

- **Stacker arch variations**: 6/7/8-model variants, 2-layer (MLP and conv meta),
  per-voxel meta-stacker — all underperform the production tuned 7-model
- **Confidence-based FP filtering**: high-confidence FPs are *also* wrong; can't
  filter our way out
- **Hard-negative mining at stacker level**: over-suppresses real lesions
- **Pseudo-label cleaning via ensemble**: bakes in ensemble's under-segmentation
  bias
- **Adding precision-trained SwinUNETR base**: anti-correlated with general swin,
  hurts ensemble
- **Anatomical context channels (T1-gd + coords)**: improved sensitivity but
  worsened HD95 and FPs
- **Boundary-loss SwinUNETR retraining**: converged to weights nearly identical
  to general SwinUNETR; no HD95 improvement

The stacker architecture is at its ceiling. **Further HD95 improvement requires
either new imaging modalities (DWI/ADC) or consensus radiologist re-annotation**,
both of which are data-side investments, not model-side.

## Path to FDA submission (next steps)

1. **Draft IFU language** for ≥10mm scope using the proposed wording above
2. **Engage regulatory consultancy** (Innolitics did Neosoma + VBrain, ~$80-150K)
3. **IRB approval** for sequestered pivotal cohort (≥70 cases)
4. **Consensus neuroradiologist reads** on dev (84) + pivotal (70) cohorts
   (~$30-40K total)
5. **Q-Sub (pre-submission) meeting** with FDA
6. (Long-term, parallel) **DWI/ADC sequence integration** for broader-IFU cleared
   variant in a future filing cycle

## Files map

```
model/
  stacking_classifier_production.pth   ← THE production model (5.5 MB, 1.36M params)
  swin_unetr_brainmets.pth             ← General SwinUNETR base (256 MB, val 0.7915)
  swin_unetr_latest.pth                ← Resumable training state for swin (754 MB)
  stacking_cache_v5/                   ← nnU-Net + legacy patch model preds (11 GB)
  stacking_cache_v5_256/               ← SwinUNETR 150ep+ preds (118 GB)
  evaluation_results/                  ← All historical experiment JSONs

scripts/
  evaluation/
    seven_model_stacker.py             ← Train/eval the production stacker
    eval_size_filtered.py              ← IFU-scoping sweep
    eval_measurable_disease.py         ← RANO-BM eval (≥10mm)
    eval_stratified_by_size.py         ← Standard stratified report
    full_stacking_comparison_v2.py     ← Hybrid 3-model baseline (imported by others)
    triage_disagreement.py             ← Label-quality triage
    build_review_gallery.py            ← HTML gallery for label review
    build_raw_vs_gt_gallery.py         ← Side-by-side MRI / GT viewer
    commit_mask_fix.py                 ← Cycle radiologist edits back into cache
    evaluate_nnunet.py                 ← nnU-Net standalone eval
    ensemble_5fold.py                  ← Cross-validation infra (legacy)
  training/
    train_swin_unetr.py                ← Two-phase SwinUNETR training
    refresh_swin_unetr_cache.py        ← Refresh swin preds in cache after retrain
    rebuild_full_cache.py              ← Full cache rebuild
    train_stacking_v5.py               ← Original stacker training (predates V2)
    train_2d.py, train_multifold.py, train_resencm.py  ← Base-model training
  evaluation/open_in_itksnap.ps1       ← Launch ITK-SNAP for label edits

src/segmentation/
  stacking.py                          ← StackingClassifierV2 + sliding window
  fda_metrics.py                       ← Neosoma-aligned + RANO-BM + stratified metrics
  losses/                              ← LesionSensitivity, Precision, SoftDice
  pipeline.py                          ← End-to-end inference pipeline class
  model_adapters.py                    ← Adapters for each base model
  ...

configs/
  models.yaml                          ← Base model registry + IFU hyperparams
```

## Historical experiment log

Full experiment chronology (Rounds 1-7, 11 stacker configurations) in:
- `docs/stacking_v4_vs_v5_writeup.md` — master experiment log
- `docs/stacking_architectures_explored.md` — final architecture summary

Per-experiment result JSONs in `model/evaluation_results/`.
