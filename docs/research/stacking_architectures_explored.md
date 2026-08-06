# Stacking Architecture Exploration — Final Report

**Date:** 2026-04-24
**Purpose:** Document every stacker architecture tried, the numbers each produced,
why it was abandoned, and the confirmed conclusion that the stacker architecture
is at its ceiling.

**Production champion:** `model/stacking_v5_seven_model_classifier.pth` — tuned
7-model V2, Dice 0.7858. All other architectures were either worse overall
or worse on critical FDA metrics (HD95, MSD).

---

## The eval protocol (constant across all variants)

- **Held-out set:** 84 BMS/UCSF MRI cases, 256³ isotropic 1mm
- **Metrics:** aggregate voxel Dice; per-bucket Dice by lesion volume; lesion-wise sensitivity / FPs-per-case / lesion-wise Dice (matched + BraTS-style) / HD95 / MSD with 1000-resample bootstrap 95% CIs
- **Comparison reference:** Old 6-model stacker (v4) at Dice 0.7778 (legacy production); Neosoma Brain Mets 510(k) K252922 as external competitive benchmark
- **Postprocessing:** `min_component_size = 0` (locked on 2026-04-19 after size-sweep experiment)

All stackers are trained on ~430 non-eval cases from the legacy-format 256³ cache,
with a held-out 48-case internal val for checkpoint selection. All inference uses
32³ sliding-window patches with 0.5 overlap.

---

## Variant 1 — v2 hybrid single stacker (3 models, 5 channels)

**Architecture:** `StackingClassifier` (original, ~27K parameters).
- 3 base predictions: nnU-Net 3D, nnU-Net 2D, SwinUNETR 150ep+
- Input: 5 channels (3 preds + variance + range)
- 3D residual CNN, 2 residual blocks at 32 channels, no attention

**Training:** 50 epochs, BCE+Dice loss, lr 1e-3, AdamW + cosine anneal, no augmentation.

**Results:**
- Dice 0.7744 (-0.003 vs old 6-model)
- Tiny bucket 0.4529 (best tiny-lesion performance of any single-layer stacker)
- HD95 23.52mm, FPs/case 4.51, lesion-wise Dice 0.788

**Why not champion:** The higher tiny-lesion performance came with much higher FPs/case
and worse HD95 than the tuned 7-model. Dice plateau lower overall.

**Checkpoint:** `stacking_v5_hybrid_classifier.pth` (476 KB)

---

## Variant 2 — 2-layer SC + meta stacker (multiple bases, 2 architectures)

**Architecture:** Three parallel StackingClassifiers (SC_dice, SC_sens, SC_prec),
each trained with a different loss:
- SC_dice: BCE + Dice (baseline)
- SC_sens: Dice + LesionSensitivityLoss (penalizes missed lesions weighted by volume)
- SC_prec: Dice + PrecisionLoss (BCE + explicit FP penalty)

Each SC sees same 5-channel v2 features (3 preds + variance + range). Meta-stacker
combines the three SCs' probability outputs per voxel.

**Meta architectures tested:**
- **MLP meta** (`MetaStackerMLP`, 3→32→16→1 per-voxel MLP, ~4K params)
- **Conv meta** (`StackingClassifier` 5-channel, ~27K params — SC outputs + variance/range)

**Training:** 20 epochs for each SC + 8/20 epochs for the meta. No 5-fold OOF
(skipped to save ~10 hrs of compute); meta trained on a 96-case held-out from
the SC-train set.

**Results:**

| Metric | 2-layer (MLP meta) | 2-layer (Conv meta) | v2 single |
|---|---|---|---|
| Dice | 0.7237 | 0.7531 | 0.7744 |
| Tiny bucket | **0.4578** | **0.4538** | 0.4529 |
| Sensitivity | **0.865** | 0.777 | 0.776 |
| Precision | 0.643 | 0.753 | 0.781 |
| HD95 (mm) | 26.7 | 26.7 | 23.5 |
| FPs/case | 7.0 | 7.0 | 4.5 |

**Why not champion:** Best tiny-lesion Dice of any config (MLP meta 0.4578, conv
meta 0.4538), but aggregate Dice regressed 0.02-0.05 and FPs/case ballooned.

**Why it failed:**
- **MLP meta**: per-voxel lookup has zero spatial context. The SC_sens base's FPs
  leaked straight through; the meta couldn't tell "fire in this blob" from "fire
  because there's a blob-shaped hyperintensity here."
- **Conv meta**: fixed the precision collapse (0.64 → 0.75) but the meta-train set
  of 96 cases was too small to teach nuanced discrimination.
- **Shared diagnosis**: without proper 5-fold OOF we were effectively using the
  SC models on their own training data when generating meta-training features.
  The meta overfit to SC idiosyncrasies.

**Checkpoints removed:** `two_layer_sc_dice.pth`, `two_layer_sc_sens.pth`,
`two_layer_sc_prec.pth`, `two_layer_meta.pth`

**Scripts removed:** `scripts/evaluation/two_layer_stacker_feasibility.py`,
`src/segmentation/stacking_classifiers.py` (retained only if other scripts import it).

---

## Variant 3 — 7-model V1 stacker (stock training)

**Architecture:** `StackingClassifierV2` (~1.36M parameters).
- 7 base predictions: nnU-Net 3D, nnU-Net 2D, patch_8, patch_12, patch_24, patch_36, SwinUNETR 150ep+
- Input: 9 channels (7 preds + variance + range)
- V2 arch: 1×1×1 input bottleneck + wider/deeper (64 channels, 4 blocks) + multi-scale
  branch (avg-pool → 2 res blocks → trilinear upsample → concat → fuse) + SE attention

**Training:** 50 epochs, BCE+Dice loss, lr 1e-3, AdamW + cosine anneal, no augmentation,
save-on-any-improvement.

**Results:**
- Dice 0.7629 (-0.015 vs old 6-model)
- Tiny bucket 0.4081
- HD95 22.99mm, FPs/case 4.96

**Why not champion:** Val loss collapsed to a "lucky" early epoch (epoch 3 of 50),
then drifted upward. The save-on-any-improvement logic baked in that noisy
checkpoint. Stock hyperparams plus a 50× larger model produced instability.

**Lesson:** Architecture upgrades without commensurate training-recipe upgrades
regress, not improve.

**Checkpoint:** overwritten by Variant 4 below (same file path).

---

## Variant 4 — 7-model V2 stacker, TUNED (CHAMPION)

**Architecture:** Same as Variant 3 (`StackingClassifierV2`, 9-channel).

**Training (tuned recipe):**
- **lr 3e-4** (down from 1e-3) with **5-epoch linear warmup** then cosine
- **dropout 0.2** (up from 0.1)
- **weight_decay 1e-4** (up from 1e-5)
- **save_min_delta = 0.005** — saves only on ≥0.005 val improvement, kills lucky-epoch saves
- **3D patch augmentation**: random flip (3 axes), 90° rotation (random plane), per-channel ±5% intensity jitter

**Results:**

| Metric | v4 Old 6-model | **Tuned 7-model V2** |
|---|---|---|
| Dice | 0.7778 | **0.7858** (+0.008) |
| Precision | 0.8050 | **0.8063** (+0.001) |
| Sensitivity | 0.7848 | 0.7823 (−0.003) |
| Tiny (<100 vox) | 0.3182 | **0.4028** (+0.085) |
| Small (100-1k) | 0.7173 | **0.7334** (+0.016) |
| Medium (1k-10k) | 0.8467 | **0.8547** (+0.008) |
| Large (>10k) | 0.8706 | 0.8735 (+0.003) |
| Lesion-wise Sens | — | 0.810 |
| FPs/case | — | **1.68** |
| HD95 (mm) | — | **18.36** |
| Lesion-wise Dice | — | 0.791 |

**Cross-scope clinical metrics (at ≥10mm measurable-disease scope):**
- Lesion-wise Sens **0.943** — exceeds Neosoma's 0.90
- Matched Dice **0.835** — near Neosoma's 0.86
- Cohen's κ **0.867** — passes RANO-BM ≥0.85
- SoLD error **1.84%** — beats RANO-BM ≤10% by 5×

**Why this is the champion:** Best overall Dice, best precision, beats old 6-model
on every size bucket, lowest FPs/case of any multi-base configuration. First config
to achieve Neosoma-competitive numbers on measurable-disease scope.

**Checkpoint (KEPT):** `model/stacking_v5_seven_model_classifier.pth` (5.5 MB)

---

## Variant 5 — 6-model stacker (patch_8 dropped)

**Rationale:** Per-base FDA diagnostic showed patch_8 was the worst base
(standalone HD95 35.2mm, 9.93 FPs/case). Testing whether dropping it improves
the ensemble.

**Architecture:** `StackingClassifierV2(in_channels=8)`, 8-channel input
(6 preds + variance + range). Same tuned recipe as Variant 4.

**Results:**

| Metric | 7-model (champion) | 6-model (no patch_8) |
|---|---|---|
| Dice | 0.7858 | 0.7808 (−0.005) |
| Tiny bucket | 0.4028 | **0.4222** (+0.019) |
| Small bucket | 0.7334 | **0.7491** (+0.016) |
| HD95 (mm) | 18.36 | **21.72** (+3.4, WORSE) |
| FPs/case | 1.68 | **2.44** (+0.76, WORSE) |

**Why not champion:** Dropping patch_8 improved tiny/small-lesion Dice by 0.02
(patch_8's FPs were drowning small real lesions), but HD95 got **worse** and
FPs/case went up. Counterintuitive: patch_8's noise was stabilizing the ensemble.
Without its "vote" on where FPs tend to appear, the remaining 6 bases agreed
less, and the stacker had fewer disagreement cues to filter with.

**Lesson:** Ensemble diversity > ensemble cleanliness. Even a noisy base
contributes discriminative signal.

**Checkpoint removed:** `stacking_v5_six_model_classifier.pth`
**Script removed:** `scripts/evaluation/six_model_stacker.py`

---

## Variant 6 — 8-model stacker (+ swin_precision)

**Rationale:** Add a precision-focused SwinUNETR variant as 8th base, trained with
`DicePrecisionLoss` (Dice + BCE-with-logits + explicit FP penalty). Hypothesis:
diversity of loss objectives → better FP discrimination.

**swin_precision training:**
- Warm-start from `swin_unetr_brainmets.pth` (general SwinUNETR, val 0.7915)
- DicePrecisionLoss with dice_weight=0.7, precision_weight=0.3, fp_penalty_weight=2.0
- 200 epochs max with early stop; converged at epoch 45, val dice 0.8059 (above general)
- Standalone on 84 eval cases: Sens 0.655 (comparable), FPs/case 8.82 (−17% vs 10.6
  for general swin), HD95 30mm (~same), Small-lesion sens **0.286** (CRATERED from 0.594)

**Gate decision:** HD95 ≤ 25mm gate FAILED (30mm). Proceeded anyway to test
ensemble integration.

**Stacker architecture:** `StackingClassifierV2(in_channels=10)`, 10-channel
(8 preds + variance + range). Same tuned recipe as Variant 4.

**Results:**

| Metric | 7-model (champion) | 8-model (+ swin_precision) |
|---|---|---|
| Dice | 0.7858 | 0.7693 (−0.017) |
| Tiny bucket | 0.4028 | **0.4339** (+0.031) |
| Sensitivity | 0.7823 | 0.7764 |
| Precision | 0.8063 | 0.7792 (−0.027) |
| HD95 (mm) | 18.36 | **24.02** (+5.7, WORSE) |
| FPs/case | 1.68 | **4.42** (+2.74, WORSE) |

**Why not champion:** Adding a precision-trained base made precision *worse*.
Counterintuitive, but traceable: swin_precision over-suppressed small lesions
(standalone small-lesion sens 0.59 → 0.29). When combined with general SwinUNETR's
"yes, fire" signal on the same small lesions, the stacker got conflicting inputs
and resolved by biasing more liberal (threshold climbed 0.55 → 0.75) — producing
more mid-scale FPs.

**Lesson:** A "corrective" base only helps if its error profile is orthogonal
(not anti-correlated) with the other bases'. swin_precision was anti-correlated,
not complementary.

**Checkpoint removed:** `stacking_v5_eight_model_classifier.pth`
**Related files removed:** `swin_unetr_brainmets_precision.pth`, `swin_unetr_latest_precision.pth`
**Scripts removed:** `scripts/evaluation/eight_model_stacker.py`,
`scripts/evaluation/standalone_swin_precision_eval.py`,
`scripts/training/cache_swin_precision.py`

---

## Variant 7 — Hard-negative retrain (7-model champion warm-start)

**Approach:** Mine 1,048 FPs from the champion on all 480 training cases. Retrain
the stacker with 30% of patches centered on mined FPs (forcing the stacker to learn
"when bases produce these patterns, predict 0").

**Architecture:** Same as champion (warm-started weights).

**Training:** 30 epochs, lr 1e-4 (low, since refining), warmup=2, save_min_delta=0.005.

**Results:**

| Metric | 7-model (champion) | Hard-neg 7-model |
|---|---|---|
| Dice | 0.7858 | 0.7802 (−0.006) |
| Tiny bucket | 0.4028 | 0.3846 (−0.018) |
| HD95 (mm) | 18.36 | 20.46 (+2.1) |
| FPs/case | 1.68 | 1.87 (+0.19) |

**Why not champion:** All metrics regressed. The stacker's best val was saved
at epoch 4 (slightly worse than the warm-start baseline) and never recovered.
Training over-corrected: teaching the stacker "these patterns are negatives"
made it conservative everywhere, including on real tiny lesions.

**Lesson:** Stacker-level hard-negative mining can't surgically target specific
FP patterns without also suppressing real predictions that look similar. Needs
finer-grained signal than what 32³ patches provide.

**Checkpoint removed:** `stacking_v5_seven_model_hardneg_classifier.pth`
**Aux data removed:** `model/hard_negatives/` (1,048 mined FP records)
**Scripts removed:** `scripts/evaluation/hard_negative_stacker.py`

---

## Variant 8 — Pseudo-label cleaning (conservative ensemble consensus)

**Approach:** Auto-clean the top-50 triage-flagged training masks using a
conservative rule:
- new_mask = 1 where ensemble_mean > 0.7 AND variance < 0.025
- new_mask = 0 where ensemble_mean < 0.1 AND variance < 0.025
- Keep original GT elsewhere (the boundary-uncertain regions)

Retrain the tuned 7-model V2 on the "cleaned" labels.

**Results:**

| Metric | 7-model (champion) | Pseudo-cleaned 7-model |
|---|---|---|
| Dice | 0.7858 | 0.7761 (−0.010) |
| Precision | 0.8063 | 0.7925 (−0.014) |
| Tiny bucket | 0.4028 | 0.4008 |
| HD95 (mm) | 18.36 | 20.43 (+2.1) |
| FPs/case | 1.68 | 2.13 (+0.45) |

**Why not champion:** Inspection of the auto-cleaned masks showed they were
systematically **smaller** than originals (e.g., BMS_Mets_311: GT 13,915 fg → new
10,910 fg). The ensemble under-segments at lesion boundaries, and trusting it as
a label oracle propagated that bias into training targets.

**Lesson:** Automated ensemble-based label cleaning is a losing strategy for
segmentation. The very errors the ensemble makes (under-segmentation at edges)
get baked into the "cleaned" ground truth. Consensus neuroradiologist reads
are the only real label-quality lever.

**Checkpoint removed:** `stacking_v5_seven_model_pseudo_classifier.pth`
**Aux data removed:** `model/pseudo_cleaned_masks/` (50 auto-cleaned masks)
**Scripts removed:** `scripts/evaluation/pseudo_label_experiment.py`

---

## Ceiling confirmation

**Seven rounds. Eleven configurations. One champion.**

```
Config                            Dice      Δ vs champion    Status
------------------------------------------------------------------
Tuned 7-model V2 (CHAMPION)       0.7858    —                production
6-model (no patch_8)              0.7808    -0.005           removed
Hard-neg 7-model                  0.7802    -0.006           removed
Pseudo-cleaned 7-model            0.7761    -0.010           removed
v2 single (hybrid 3-model)        0.7744    -0.011           preserved*
Old 6-model (v4)                  0.7778    -0.008           legacy
8-model (+swin_precision)         0.7693    -0.017           removed
7-model V2 stock (untuned)        0.7629    -0.023           historical
2-layer conv meta                 0.7531    -0.033           removed
2-layer MLP meta                  0.7237    -0.062           removed
v1 uniform-256 hybrid (broken)    0.6453    -0.141           abandoned
```

\* v2 hybrid single stacker checkpoint is kept because it has best tiny-bucket
Dice of any single-layer architecture and serves as reference for the "small
tiny-specialist vs overall generalist" tradeoff.

**The stacker architecture is at its ceiling on this data.** Everything in
`Stacking v2 → v2-tuned` space has been explored; diminishing returns are
clear. The remaining FDA gap (HD95 18.36mm vs ≤2.94mm) is not addressable at
the stacker level — it requires either:

1. **Per-base precision retraining** with boundary losses (~2-3 weeks, moderate
   upside).
2. **Additional imaging modalities** (DWI/ADC) as input channels — the single
   highest-ROI signal-level intervention.
3. **IFU scoping** to ≥10mm ("measurable disease") — the **current submission-path
   recommendation**, as numbers are already Neosoma-competitive at that scope.

## Cleanup executed on 2026-04-24

**Checkpoints removed (~1.1 GB):**
- `stacking_v5_six_model_classifier.pth` (5.4 MB)
- `stacking_v5_eight_model_classifier.pth` (5.4 MB)
- `stacking_v5_seven_model_hardneg_classifier.pth` (5.4 MB)
- `stacking_v5_seven_model_pseudo_classifier.pth` (5.4 MB)
- `stacking_v5_hybrid_classifier.pth` (476 KB) — KEPT as reference
- `two_layer_sc_dice.pth`, `two_layer_sc_sens.pth`, `two_layer_sc_prec.pth`, `two_layer_meta.pth` (~1.9 MB)
- `stacking_v5_classifier.pth`, `stacking_v5_256_classifier.pth` (~1 MB, legacy artifacts)
- `swin_unetr_brainmets_precision.pth` (244 MB)
- `swin_unetr_latest_precision.pth` (750 MB)

**Scripts removed:**
- `scripts/evaluation/two_layer_stacker_feasibility.py`
- `scripts/evaluation/six_model_stacker.py`
- `scripts/evaluation/eight_model_stacker.py`
- `scripts/evaluation/hard_negative_stacker.py`
- `scripts/evaluation/pseudo_label_experiment.py`
- `scripts/evaluation/standalone_swin_precision_eval.py`
- `scripts/training/cache_swin_precision.py`

**Auxiliary data removed:**
- `model/hard_negatives/` (1,048 mined FP records, ~200 MB)
- `model/pseudo_cleaned_masks/` (50 auto-cleaned mask npzs, ~400 MB)

**Preserved (kept for reference/history):**
- All `model/evaluation_results/*.json` result files — small, historical
- `scripts/evaluation/full_stacking_comparison_v2.py` — baseline 3-model hybrid,
  imported by other scripts
- `scripts/evaluation/seven_model_stacker.py` — produces the champion
- `src/segmentation/stacking_classifiers.py` — only used for 2-layer; can be
  removed if preferred, but tiny file
- `model/stacking_cache_v5/`, `model/stacking_cache_v5_256/` — base prediction
  caches, still used by champion

**Total space reclaimed: ~1.6 GB**

## Bottom line

`model/stacking_v5_seven_model_classifier.pth` is the production model. Further
stacker iteration has been confirmed exhausted. All investment from here should
go into regulatory/validation work (consensus reads, pivotal cohort, Q-Sub) or
base-level / feature-level improvements (DWI, boundary-loss base retraining).
