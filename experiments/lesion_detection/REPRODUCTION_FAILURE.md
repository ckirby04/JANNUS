# Reproduction-gate failure: diagnosis

Status: **under diagnosis. No fix is applied on this branch.** Date: 2026-10-04.
Scope set by the owner:

- 5 TUNING (contaminated) cases;
- a one-time read-only exception for `1.40/model/stacking_cache_v5/` and `stacking_cache_v5_256/`;
- **no ground truth**: the cache files' `mask` key was never read;
- no change to any pipeline code, config, weight or weights.lock.

Nothing was run on clean cases.

## The failure

The production pipeline was run unchanged through the `jannus predict` path
(`BrainMetPipeline.from_config(verify=True)` + `load_case`) on the same `preprocessed_256`
inputs the published figures were computed on. It does not reproduce those figures:

| | Logged | Regenerated |
|---|---|---|
| Mean voxel Dice | 0.7858 | 0.6093 |
| Mean lesion sensitivity | 0.810 | 0.5630 |
| Total FPs | 141 | 223 |
| Matched lesion Dice | 0.791 | 0.7143 |

Details: `results/repro_gate/`, EXPERIMENT_LOG.md.

## What was compared

**Cases.** `BMS_Mets_039`, `BMS_Mets_051`, `UCSF_100102A`, `UCSF_100110A`, `UCSF_100319A`. All five are in `contaminated_ids.txt` and none is in `clean_ids.txt`; the script enforces both.

**Cache files read.** Five `.npz` files in each of the two cache directories. Their SHA-256 hashes are in `results/reproduction_diagnosis/provenance.json`.

| Cache | Keys present | Keys read |
|---|---|---|
| `stacking_cache_v5/` | `mask, nnunet, nnunet_2d, patch_8, patch_12, patch_24, patch_36` | all except `mask` |
| `stacking_cache_v5_256/` | `mask, nnunet_2d, nnunet_3d, swin_precision, swin_unetr, t1_gd_norm` | `swin_unetr`, `nnunet_3d`, `nnunet_2d` |

The production stacker was trained and evaluated on `nnunet`, `nnunet_2d` and `patch_*` from
`stacking_cache_v5/`, plus `swin_unetr` from `stacking_cache_v5_256/`
(`scripts/evaluation/seven_model_stacker.py:load_seven_preds`).

**Fresh maps.** Fresh maps were regenerated in float32 by today's adapters. Runtime-only forward
pre-hooks recorded the input tensor each network receives.

**Scripts and raw output.**
- `tools/diagnose_reproduction.py` → `results/reproduction_diagnosis/diagnosis.json`
- `tools/diagnose_nnunet_cache_variants.py` → `results/reproduction_diagnosis/nnunet_cache_variants.json`

## Findings

### 1. Fresh vs cached base maps, model by model

Each cell gives the range over the 5 cases. "Dice ≥0.5" compares the fresh and cached masks
thresholded at 0.5; it is agreement between two predictions, not accuracy.

| Model | Pearson r | Max abs diff | Mean abs diff | Dice ≥0.5 (fresh vs cached) | Voxels ≥0.5: fresh / cached (median) |
|---|---|---|---|---|---|
| nnunet_3d (vs `v5:nnunet`) | 0.72–0.96 | 0.997–1.00 | 0.0000–0.0005 | 0.68–0.95 | 9,518 / 10,274 |
| nnunet_2d (vs `v5:nnunet_2d`) | 0.36–0.94 | 1.00 | 0.0001–0.0011 | **0.23**–0.93 | 3,880 / 10,987 |
| patch_8 | 0.22–0.44 | 0.985–0.998 | 0.020–0.023 | **0.03–0.24** | **117,097** / 11,184 |
| patch_12 | 0.19–0.60 | 0.973–0.998 | 0.007–0.011 | **0.05–0.52** | **52,395** / 10,932 |
| patch_24 | 0.30–0.91 | 0.95–1.00 | 0.0005–0.0014 | 0.21–0.88 | 13,705 / 10,858 |
| patch_36 | 0.52–0.94 | 0.995–1.00 | 0.0002–0.0007 | 0.61–0.91 | 13,530 / 11,313 |
| swin_unetr | **1.000** | **0.004–0.005** | 0.0000 | **1.000** | 11,541 / 11,541 |

**nnU-Net against the *other* cached variant.**

| Comparison | Pearson r | Dice ≥0.5 | Max abs diff |
|---|---|---|---|
| fresh nnunet_3d vs `v5_256:nnunet_3d` | **1.000** (all 5) | **1.000** | 0.002–0.003 |
| fresh nnunet_2d vs `v5_256:nnunet_2d` | **1.000** (all 5) | **1.000** | 0.008–0.039 |
| `v5:nnunet` vs `v5_256:nnunet_3d` (both cached) | 0.72–0.96 | 0.68–0.95 | ~1.0 |

`docs/research/stacking_v4_vs_v5_writeup.md:55` describes the `v5_256` nnU-Net variant as nnU-Net "fed
data it wasn't trained on". In that writeup, "forcing it onto uniformly-resampled 256³ volumes broke its
calibration" and caused a 0.23 Dice regression. The `v5` variant is described as
"native-preprocessed" (`PRODUCTION.md:12`, writeup:61), and **the production stacker was trained and
evaluated on `v5`.** Today's pipeline reproduces the `v5_256` variant exactly.

### 2. Did the cached patch_8 / patch_12 always look wild?

**No.** Here are label-free cross-model statistics on the **cached** maps: voxels ≥0.5, and Dice
against the majority vote of the other six cached models.

| Case | nnunet_3d | patch_8 | patch_12 | patch_24 | patch_36 | swin |
|---|---|---|---|---|---|---|
| BMS_Mets_039 | 1,520 / 0.82 | 1,739 / 0.77 | 1,311 / 0.80 | 1,776 / 0.77 | 1,626 / 0.74 | 1,360 / 0.71 |
| UCSF_100102A | 10,274 / 0.97 | 11,184 / 0.98 | 10,932 / 0.98 | 10,858 / 0.99 | 11,313 / 0.98 | 11,541 / 0.94 |

The other three cases show the same pattern: cached patch voxel counts within ~20 % of the
other models, and majority Dice 0.74–0.99. The cached patch_8/patch_12 maps behaved like every
other model. Fresh patch_8 flags 87k–157k voxels per scan, 7–56× its cached count (median ~11×), and its mask agrees with the cached one at Dice 0.03–0.24. Across the 55 tuning scans, fresh patch_8 has a median majority Dice of 0.022 (EXPERIMENT_LOG).
**The patch-model path changed.** The stacker did not learn around wild inputs; it never saw them.

### 3. Which change moves the fused output: the stacker swap test

The production stacker was run on several input sets. The reference is the stacker output on
**all-cached** inputs. Each cell is the Dice (≥0.55) of a variant against that reference; this is
label-free agreement, not accuracy.

| Variant | 039 | 051 | 102A | 110A | 319A |
|---|---|---|---|---|---|
| all fresh (= today's pipeline) | 0.548 | 0.919 | 0.962 | 0.789 | 0.910 |
| fresh, but cached **nnunet_3d** | 0.709 | **0.965** | 0.972 | **0.962** | **0.970** |
| fresh, but cached nnunet_2d | 0.561 | 0.929 | 0.964 | 0.814 | 0.923 |
| fresh, but cached patch_8 | 0.688 | 0.920 | 0.963 | 0.803 | 0.922 |
| fresh, but cached patch_12 | 0.704 | 0.917 | 0.962 | 0.783 | 0.911 |
| fresh, but cached patch_24 | 0.819 | 0.921 | 0.965 | 0.808 | 0.922 |
| fresh, but cached all 4 patch | **0.860** | 0.923 | 0.971 | 0.827 | 0.936 |
| cached, but fresh all 4 patch (= cached nnU-Net + swin) | 0.716 | **0.976** | **0.973** | **0.967** | **0.971** |
| fresh, but cached swin_unetr | 0.548 | 0.919 | 0.962 | 0.789 | 0.910 |

Swapping back the cached nnU-Net 3D is the single largest restoration in 4 of 5 cases. With
cached nnU-Net and Swin plus *fresh* patches, agreement reaches 0.97–0.98 in those 4 cases. In
`BMS_Mets_039` the patch models dominate instead: swapping all four patch maps gives 0.86, against
0.71 for nnU-Net 3D. Both changes matter. nnU-Net is the larger one on these five cases.

### 4. Input tensors at the point each network receives them (`UCSF_100102A`)

| Model | Tensor | Notes |
|---|---|---|
| Pipeline input (`load_case`) | (4, 256, 256, 256) float32, per-channel z-score: mean 0, std 1, min −0.33, max 12.1 | z-score over the whole volume, background included |
| nnunet_3d | (1, 4, 112, 160, 128) float32 patches, mean 0.02, std 1.02 | nnU-Net re-crops and re-normalises internally. It is told the 256³ grid has **1.0 mm isotropic spacing** (`model_adapters.py:313-318`), so it resamples to its plan spacing from a false geometry. |
| nnunet_2d | (1, 4, 192, 160) float32 slices | same false 1 mm spacing |
| patch_8/12/24/36 | (B, 4, p, p, p) float32 → **fp16 autocast** inside `stacking.sliding_window_inference` | built from a 128³ trilinear resample of the 256³ volume, then a second z-score (`model_adapters.py:441-465`) |
| swin_unetr | (1, 4, 96, 96, 96) float32 | MONAI sliding window, roi 96, overlap 0.5, gaussian |

The hook records the first forward call, so the patch and 2D statistics come from a background
patch or slice at the volume corner. Shapes and dtypes are representative; the means are not.
Full per-case statistics are in `diagnosis.json`.

**What the cache-era code fed the models cannot be determined from 1.50.** The cache-build script
(`1.30/scripts/training/native_overnight.py`, cited in `model_adapters.py:17`) and the nnU-Net
"native" inference path are in frozen trees outside this exception. The cache keeps no inputs.
`t1_gd_norm` in `v5_256` is the one stored input channel, and it was not read. Swin reproduces
exactly, which shows the shared `load_case` normalisation matches what Swin's cache was built
from. It says nothing about the nnU-Net-native or legacy-patch inputs.

### 5. Checkpoint hashes, today vs the cache era

- **Today.** `jannus.core.checksums.verify_manifest` passes for all 12 weights.lock entries, so today's files match weights.lock.json.
  - nnunet_3d `d62bce38…`, nnunet_2d `c54d77ec…`
  - patch_8 `0c33c2e0…`, patch_12 `54fc0aa8…`, patch_24 `f64dd6ba…`, patch_36 `853cd691…`
  - swin `c88036a2…`, stacker `8416674c…`
- **Cache era.** No hash is recorded anywhere this diagnosis could read. The npz files carry no metadata keys, and 1.50 has no record from April 2026.
- **Git history of weights.lock.json.** Three commits:
  - `e4c1cee` (2026-08-05, v1.50 productisation) created it.
  - `c440748` (2026-09-08) added the four nnU-Net sidecar entries (`plans.json`, `dataset.json`).
  - `d43dc97` (2026-09-08, fetch-weights `--source`) changed only the `_comment` text.
  - **No checkpoint hash changed across these commits.** weights.lock pins what was in `model/` when v1.50 was cut. It does not, and cannot, show that this equals what produced the cache.
- **What the outputs imply.**
  - nnU-Net 3D/2D and Swin reproduce their `v5_256` cache values to ≤0.04, so those checkpoints are almost certainly the ones behind `v5_256`.
  - The patch checkpoints carry `training: 'v3 (SmallLesionLoss + MONAI aug + weighted sampling)'` (epochs 17/34/87/472). `configs/models.yaml` says the v3 files were "copied from 1.30/model/native_v3/… on **2026-04-25**".
  - The stacker experiments that produced the production checkpoint ran 2026-04-18 → 04-24 (writeup header; Round 6 dated 04-22 → 04-24).
  - `PRODUCTION.md:13` and `:170` call the cached patch predictions "legacy".

### 6. Output space: pre- or post-sigmoid?

Every fresh and every cached map lies in [0, 1] (`output_range` in `diagnosis.json`):

| Model | How it reaches [0, 1] |
|---|---|
| patch_* | `torch.sigmoid` inside `stacking.sliding_window_inference` (`stacking.py:308-310`) |
| nnU-Net | softmax foreground channel from `predict_single_npy_array(..., save_probabilities=True)` |
| Swin | `softmax(logits)[1]` |

The stacker was trained on the cached [0, 1] maps, and it builds variance and range features that
assume probabilities. **No pre/post-sigmoid mismatch was found.**

### 7. Patch, sliding-window and overlap settings, today vs training/cache era

| Model | Today | Cache era / training | Match? |
|---|---|---|---|
| nnU-Net 3D/2D | `tile_step_size=0.5`, gaussian, **no mirroring TTA**, spacing forced to 1 mm on the 256³ grid | `v5_256`: identical (exact reproduction). `v5` ("native"): unknown, script not readable | **No** vs `v5` |
| patch_8/12/24/36 | 128³ resample → z-score → sliding window at p = 8/12/24/36, overlap 0.5, fp16 autocast → trilinear upsample | Adapter docstring claims this mirrors `native_overnight.py`; cannot be verified. Training patch sizes 8/12/24/36 match the checkpoints' `patch_size`; `model_config` (base 20, attention, residual, no deep supervision) matches the config | Weights differ (v3 vs legacy); protocol unverifiable |
| swin_unetr | roi 96³, overlap 0.5, gaussian, softmax | identical (exact reproduction) | Yes |
| stacker | 32³ windows, overlap 0.5, 9 channels (7 + variance + range) | `seven_model_stacker.py`: `STACKING_PATCH = 32`, `STACKING_OVERLAP = 0.5`, `build_features_9ch` with the same formula | Yes |

## Root-cause hypotheses, ranked by evidence

1. **The nnU-Net 3D/2D input path is not the one the stacker was trained on. Confirmed on 5/5 cases.**
   - Today's adapter runs nnU-Net on the resampled 256³ grid and declares it 1 mm isotropic (`model_adapters.py:313-318`; the comment assumes real spacing is "handled upstream", and nothing upstream handles it).
   - It reproduces, exactly, the `v5_256` nnU-Net variant that the 2026-04 writeup documented as calibration-broken.
   - The production stacker was trained and evaluated on the `v5` "native-preprocessed" variant.
   - In the swap test, restoring cached nnU-Net 3D alone is the largest single correction in 4/5 cases.
   - The checkpoints are not implicated: outputs match `v5_256` to ≤0.003.

2. **The patch models are a different generation from the ones whose outputs the stacker learned. Strong but circumstantial.**
   - Cached patch_8/12 behave like the other models. Fresh patch_8 flags 7–56× more voxels than cached (patch_12: 3–35×) and barely agrees with anything, and patch_24/36 also diverge (fresh-vs-cached Dice 0.21–0.91).
   - The loaded checkpoints are tagged v3 and were copied in 2026-04-25, after the stacker work (04-18 → 04-24), and the docs call the cached patch predictions "legacy".
   - In the swap test the patch models are the largest factor in 1 of 5 cases.
   - **Not excluded:** the patch protocol (128³ resample, second z-score) may also differ from the cache build. That cannot be separated from a weight change without the legacy patch weights or `native_overnight.py`.
   - A side observation, not evidence: the v3 checkpoints' stored validation is patch-level (`dice_scores` only for `tiny`/`small`, `medium`/`large` = None). That fits models tuned for lesion-centred patches that fire on background when slid over a whole volume.

3. **Shared input normalisation (`load_case`). Ruled out as a global cause.** SwinUNETR, fed the same tensor, reproduces its cache exactly.

4. **Sigmoid/logit mismatch. Ruled out.** Every map is in [0, 1], and the stacker consumes probabilities in both eras.

5. **Stacker configuration. No evidence of a change.** Window 32, overlap 0.5 and the feature formula match the training script, and the stacker hash is unchanged since v1.50 was cut.
   - Not tested: whether `stacking_classifier_production.pth` reproduces the published figures when fed the *cached* maps. That needs labels, so it is outside this diagnosis.
   - That check is the single test that would confirm hypotheses 1 and 2 together explain the whole gap.

6. **Storage precision (float16 stacker map, `>` vs `>=`) and header spacing. Negligible or not involved.**
   - The magnitude of the gap is orders above float16 resolution.
   - The pipeline ignores header spacing for inference: nnU-Net's spacing is hard-coded, and `voxel_spacing` feeds only lesion statistics.

## Implications (not acted on)

- On these inputs, **`jannus predict` as shipped does not reproduce the published figures.** The base maps it produces are not those the stacker was trained on, for nnU-Net 3D/2D and all four patch models.
- On native-resolution site data, nnU-Net would again be told 1 mm isotropic spacing, and the patch models would still be v3. A gap is therefore likely there too. This is a hypothesis; option 2 would test it.
- External sites following CROSS_VALIDATION.md compare their numbers to figures produced by a different set of base-model inputs than the software they run.
- The lesion-detection experiment cannot use the regenerated maps as "the production pipeline that produced the published figures". It could proceed as a study of the pipeline *as shipped* (option 4), or on the cached maps for the tuning split. Both are decisions for the owner.

## Correction to an earlier document

While diagnosing, it emerged that SPACING_DEFECT.md overstated one detail. Only the 105 BMS
volumes have identity affines with 1 mm pixdim. The 461 UCSF volumes keep their **native**
affine and zooms unchanged after the resize; for example, `UCSF_100102A` claims 1.17 × 1.17 × 1.6 mm
where the truth is 1.17 × 1.17 × 0.675 mm. Both kinds of header are wrong. The true-spacing
derivation used by this experiment is unaffected. SPACING_DEFECT.md has been corrected, and the
change is logged.
