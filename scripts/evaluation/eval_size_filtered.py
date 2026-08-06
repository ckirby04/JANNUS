"""
Size-filtered evaluation: what do the metrics look like if we exclude
micromets below a minimum diameter?

Sweeps several exclusion cutoffs against the tuned 7-model V2 stacker on
the 84 eval cases. For each cutoff:
  - Drop GT connected components below the volume threshold (treat them as
    "don't-care" -- clinical indication excludes them).
  - For predicted components that overlap ONLY removed GTs, treat them as
    don't-care too (not FPs, they found micromets we chose not to count).
  - For predicted components overlapping kept GTs, keep (TP candidates).
  - For predicted components overlapping no GT, keep (FPs).
  - Compute all FDA metrics on the filtered pair.

Volume cutoffs (1mm isotropic voxels):
    5  voxels  ~= 2.1mm diameter
    15 voxels  ~= 3.0mm diameter
    30 voxels  ~= 3.9mm diameter
    100 voxels ~= 5.7mm diameter  (clinical "measurable disease" floor)
    500 voxels ~= 9.9mm diameter  (RANO-BM "measurable" by longest dia)

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/eval_size_filtered.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from scipy.ndimage import label as ndimage_label

PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT))

from jannus.segmentation.stacking import (
    StackingClassifierV2, postprocess_prediction, sliding_window_inference,
)
from jannus.evaluation.metrics import (
    aggregate_with_ci, compute_case_metrics, format_fda_report,
)
from scripts.evaluation.full_stacking_comparison_v2 import (
    OLD_PER_CASE_PATH, voxel_dice, voxel_sens, voxel_prec, per_lesion_dice, bucket,
)
from scripts.evaluation.seven_model_stacker import (
    load_seven_preds, build_features_9ch, IN_CHANNELS,
    STACKING_PATCH, STACKING_OVERLAP,
)

CHECKPOINT_PATH = PROJECT / "model" / "stacking_classifier_production.pth"
OUTPUT_PATH = PROJECT / "model" / "evaluation_results" / "size_filtered_eval.json"

THRESHOLD = 0.55           # from tuned 7-model eval
MIN_COMPONENT_SIZE = 0     # default postproc (no size filter on pred alone)
SIZE_CUTOFFS = [0, 5, 15, 30, 100, 500]   # in voxels; 0 = unfiltered baseline


def filter_by_gt_size(pred_bin: np.ndarray, gt_bin: np.ndarray,
                      min_voxels: int) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Exclude GT components < min_voxels and their don't-care pred overlaps."""
    if min_voxels <= 0:
        return pred_bin, gt_bin, {"removed_gt": 0, "removed_pred": 0,
                                     "kept_gt": int(ndimage_label(gt_bin)[1])}
    gt_cc, n_gt = ndimage_label(gt_bin)
    pred_cc, n_pred = ndimage_label(pred_bin)

    removed_gt_ids = set()
    gt_filtered = np.zeros_like(gt_bin, dtype=np.uint8)
    for i in range(1, n_gt + 1):
        comp = (gt_cc == i)
        if comp.sum() < min_voxels:
            removed_gt_ids.add(i)
        else:
            gt_filtered[comp] = 1

    kept_gt = n_gt - len(removed_gt_ids)
    removed_pred = 0
    pred_filtered = np.zeros_like(pred_bin, dtype=np.uint8)
    for i in range(1, n_pred + 1):
        comp = (pred_cc == i)
        overlapping_gt = np.unique(gt_cc[comp])
        overlapping_gt = overlapping_gt[overlapping_gt > 0]
        # If this pred overlaps at least one kept GT, keep it -- it's still a
        # valid prediction attempt on a real ≥2mm lesion.
        if len(overlapping_gt) > 0 and all(g in removed_gt_ids for g in overlapping_gt):
            # pred matches ONLY micromets we now consider don't-care -> drop
            removed_pred += 1
            continue
        pred_filtered[comp] = 1

    return pred_filtered, gt_filtered, {
        "removed_gt": len(removed_gt_ids),
        "removed_pred": removed_pred,
        "kept_gt": kept_gt,
    }


def main():
    with open(OLD_PER_CASE_PATH) as f:
        eval_ids = sorted(json.load(f).keys())
    print(f"Eval cases: {len(eval_ids)}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StackingClassifierV2(in_channels=IN_CHANNELS).to(device)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded {CHECKPOINT_PATH} (epoch {ckpt.get('epoch')}, "
          f"val {ckpt.get('val_loss'):.4f})", flush=True)
    print(f"Threshold: {THRESHOLD}\n", flush=True)

    # Run inference once, cache prob maps + GT masks
    print(f"Running inference on {len(eval_ids)} cases...", flush=True)
    cases = {}
    t0 = time.time()
    for i, cid in enumerate(eval_ids):
        preds, gt = load_seven_preds(cid)
        features = build_features_9ch(preds)
        prob = sliding_window_inference(model, features, STACKING_PATCH, device,
                                        overlap=STACKING_OVERLAP)
        pred_bin = postprocess_prediction((prob > THRESHOLD).astype(np.float32),
                                          min_size=MIN_COMPONENT_SIZE).astype(np.uint8)
        cases[cid] = (pred_bin, (gt > 0.5).astype(np.uint8))
        if (i + 1) % 20 == 0 or i + 1 == len(eval_ids):
            print(f"  [{i+1}/{len(eval_ids)}] ({time.time()-t0:.0f}s)", flush=True)

    # Sweep size cutoffs
    results = {}
    for cutoff in SIZE_CUTOFFS:
        print(f"\n{'=' * 82}")
        print(f"  CUTOFF = {cutoff} voxels (approx >= {cutoff:.0f} vox, "
              f"~{(6 * cutoff / np.pi) ** (1/3):.1f} mm diameter)")
        print(f"{'=' * 82}")
        dices = []
        senss, precs = [], []
        fda_per_case = []
        total_removed_gt = total_removed_pred = total_kept_gt = 0

        for cid, (pred_bin, gt_bin) in cases.items():
            pf, gf, stats = filter_by_gt_size(pred_bin, gt_bin, cutoff)
            total_removed_gt += stats["removed_gt"]
            total_removed_pred += stats["removed_pred"]
            total_kept_gt += stats["kept_gt"]

            dices.append(voxel_dice(pf, gf))
            senss.append(voxel_sens(pf, gf))
            precs.append(voxel_prec(pf, gf))
            fda_per_case.append(compute_case_metrics(pf, gf, spacing=1.0))
        agg = aggregate_with_ci(fda_per_case, n_bootstrap=1000)

        overall_dice = float(np.mean(dices))
        overall_sens = float(np.mean(senss))
        overall_prec = float(np.mean(precs))
        print(f"  GT components removed: {total_removed_gt} "
              f"(kept {total_kept_gt})")
        print(f"  Pred components dropped as don't-care: {total_removed_pred}")
        print(f"  Overall voxel Dice: {overall_dice:.4f}   "
              f"Sens: {overall_sens:.4f}   Prec: {overall_prec:.4f}")
        print(format_fda_report(agg))

        results[str(cutoff)] = {
            "cutoff_voxels": cutoff,
            "approx_diameter_mm": float((6 * cutoff / np.pi) ** (1/3)) if cutoff > 0 else 0.0,
            "removed_gt_total": total_removed_gt,
            "kept_gt_total": total_kept_gt,
            "removed_pred_total": total_removed_pred,
            "voxel_dice": overall_dice,
            "voxel_sens": overall_sens,
            "voxel_prec": overall_prec,
            "fda_metrics": agg,
        }

    # Summary table
    print(f"\n{'=' * 100}")
    print(f"  SIZE-FILTERED SUMMARY")
    print(f"{'=' * 100}\n")
    print(f"  {'cutoff':<10} {'~diam':<10} {'Dice':<10} {'lesion-sens':<14} "
          f"{'FPs/case':<12} {'HD95 (mm)':<14} {'MSD (mm)':<12} {'kept/total GT'}")
    print(f"  {'-' * 100}")
    # Get total GT count from cutoff=0 row
    total_gt = results["0"]["kept_gt_total"]
    for k in SIZE_CUTOFFS:
        r = results[str(k)]
        diam = r['approx_diameter_mm']
        a = r["fda_metrics"]
        print(f"  {k:<10} {diam:<10.1f} {r['voxel_dice']:<10.4f} "
              f"{a['lesion_wise_sensitivity']['mean']:<14.4f} "
              f"{a['fp_per_case']['mean']:<12.3f} "
              f"{a['hd95_mm']['mean']:<14.2f} "
              f"{a['msd_mm']['mean']:<12.2f} "
              f"{r['kept_gt_total']}/{total_gt}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump({
            "description": "Size-filtered eval of tuned 7-model V2 stacker. "
                           "GT components below each cutoff treated as don't-care, "
                           "predicted components overlapping ONLY removed GTs dropped too.",
            "checkpoint": str(CHECKPOINT_PATH),
            "threshold": THRESHOLD,
            "cutoffs": results,
            "eval_cases": eval_ids,
        }, f, indent=2, default=str)
    print(f"\n  Results -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
