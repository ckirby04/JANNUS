"""
Disagreement-based label triage: rank 480 training cases by
(ensemble agreement) x (1 - GT Dice).

Identifies cases where the 7 base models CONFIDENTLY agree on a prediction
that DOESN'T MATCH the GT mask. These are likely label errors: either the
annotator missed a small lesion (FP region where models agree, GT empty)
or drew boundaries too loosely (FN region where models agree, GT had it).

Ranks top-N and writes JSON for gallery + pseudo-label downstream steps.

Usage:
    python scripts/evaluation/triage_disagreement.py --top-n 50
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT))

from jannus.evaluation.metrics import match_lesions
from jannus.segmentation.stacking import postprocess_prediction
from scripts.evaluation.full_stacking_comparison_v2 import (
    OLD_CACHE, NEW_CACHE, OLD_PER_CASE_PATH, BAD_CACHE_IDS, _is_legacy_256,
)
from scripts.evaluation.seven_model_stacker import (
    _is_legacy_seven, load_seven_preds,
)

OUTPUT_PATH = PROJECT / "model" / "evaluation_results" / "triage_ranking.json"


def voxel_dice(p: np.ndarray, g: np.ndarray, smooth: float = 1e-8) -> float:
    tp = ((p > 0) & (g > 0)).sum()
    fp = ((p > 0) & (g == 0)).sum()
    fn = ((p == 0) & (g > 0)).sum()
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0
    return float(2 * tp) / float(2 * tp + fp + fn + smooth)


def case_triage_stats(case_id: str, threshold: float = 0.5) -> dict:
    """Compute per-case disagreement statistics."""
    preds, mask = load_seven_preds(case_id)  # (7, 256, 256, 256), (256, 256, 256)

    ensemble_mean = preds.mean(axis=0)          # (256^3)
    ensemble_var  = preds.var(axis=0)            # (256^3)

    # Binarize ensemble at threshold
    ensemble_bin = (ensemble_mean > threshold).astype(np.float32)
    gt_bin = (mask > 0.5).astype(np.float32)

    # Overall case-level agreement vs GT
    case_dice = voxel_dice(ensemble_bin, gt_bin)

    # High-confidence FP mass: places where ensemble is very confident (>0.7)
    # AND variance is low (<0.03) AND GT is empty. These are candidate
    # missed annotations.
    conf_fp_mask = (ensemble_mean > 0.7) & (ensemble_var < 0.03) & (gt_bin == 0)
    confident_fp_vox = int(conf_fp_mask.sum())

    # High-confidence FN mass: places where ensemble is very UN-confident
    # (<0.1) AND variance is low AND GT says lesion. Candidate over-annotations.
    conf_fn_mask = (ensemble_mean < 0.1) & (ensemble_var < 0.03) & (gt_bin > 0)
    confident_fn_vox = int(conf_fn_mask.sum())

    # Mean agreement (1 - variance) — high = bases agree
    mean_agreement = float(1.0 - ensemble_var.mean())

    # Mean confidence in predicted foreground — how sure bases are when they fire
    pred_fg = ensemble_mean > threshold
    mean_conf_in_pred = float(ensemble_mean[pred_fg].mean()) if pred_fg.any() else 0.0

    # Triage score: prioritize cases with confident disagreement in either direction
    # Normalize voxel counts by expected lesion burden (sum of gt + pred) to
    # avoid large volumes dominating
    burden = max(int(gt_bin.sum()) + int(ensemble_bin.sum()), 1)
    conf_fp_frac = confident_fp_vox / burden
    conf_fn_frac = confident_fn_vox / burden
    disagreement_mass = conf_fp_frac + conf_fn_frac

    # Composite score: agreement-between-bases * disagreement-with-GT
    triage_score = mean_agreement * (1.0 - case_dice)

    return {
        "case_id": case_id,
        "case_dice_vs_ensemble": float(case_dice),
        "mean_agreement": mean_agreement,
        "mean_confidence_in_predicted_fg": mean_conf_in_pred,
        "confident_fp_voxels": confident_fp_vox,
        "confident_fn_voxels": confident_fn_vox,
        "confident_fp_frac_of_burden": float(conf_fp_frac),
        "confident_fn_frac_of_burden": float(conf_fn_frac),
        "disagreement_mass": float(disagreement_mass),
        "triage_score": float(triage_score),
        "gt_voxels": int(gt_bin.sum()),
        "pred_voxels": int(ensemble_bin.sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-n", type=int, default=50,
                    help="How many top-ranked cases to output")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    with open(OLD_PER_CASE_PATH) as f:
        eval_ids = set(json.load(f).keys())

    # Candidate cases: legacy 256^3 with all 7 preds, excluding eval set
    all_ids = sorted({p.stem for p in OLD_CACHE.glob("*.npz")}
                     & {p.stem for p in NEW_CACHE.glob("*.npz")})
    candidates = [c for c in all_ids
                  if c not in eval_ids and c not in BAD_CACHE_IDS
                  and _is_legacy_seven(c)]
    print(f"Triage candidates (legacy 256^3, non-eval): {len(candidates)}",
          flush=True)

    stats: List[dict] = []
    t0 = time.time()
    for i, cid in enumerate(candidates):
        try:
            s = case_triage_stats(cid, threshold=args.threshold)
            stats.append(s)
        except Exception as e:
            print(f"  skip {cid}: {e}", flush=True)
            continue
        if (i + 1) % 25 == 0 or i + 1 == len(candidates):
            print(f"  [{i + 1}/{len(candidates)}] ({time.time() - t0:.0f}s)",
                  flush=True)

    # Rank by triage_score, break ties by disagreement_mass
    stats.sort(key=lambda s: (-s["triage_score"], -s["disagreement_mass"]))

    top = stats[:args.top_n]
    print(f"\nTop {len(top)} cases:", flush=True)
    print(f"  {'rank':<4} {'case_id':<22} {'dice':<7} {'agree':<7} "
          f"{'conf_fp%':<10} {'conf_fn%':<10} {'score':<8}", flush=True)
    print(f"  {'-' * 76}", flush=True)
    for i, s in enumerate(top, 1):
        print(f"  {i:<4} {s['case_id']:<22} {s['case_dice_vs_ensemble']:<7.3f} "
              f"{s['mean_agreement']:<7.3f} "
              f"{100 * s['confident_fp_frac_of_burden']:<10.2f} "
              f"{100 * s['confident_fn_frac_of_burden']:<10.2f} "
              f"{s['triage_score']:<8.4f}", flush=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump({
            "description": "Disagreement-based ranking of training cases. "
                           "Top cases have high ensemble agreement between "
                           "bases AND low Dice vs GT -- candidate label errors.",
            "threshold": args.threshold,
            "n_candidates": len(candidates),
            "top_n": args.top_n,
            "top_cases": top,
            "all_stats": stats,
        }, f, indent=2, default=str)
    print(f"\n  Results -> {OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
