#!/usr/bin/env python
"""
Precompute HF Space demo files for the production architecture.

Runs the production BrainMetPipeline (7-base + StackingClassifierV2) on each
demo case in `Data/preprocessed_256/train/<case>/`, then writes
`hf_space/data/<case>.npz` with the keys hf_space/app.py expects:

    t1_gd            (256, 256, 256) float16  — visualization channel
    ground_truth     (256, 256, 256) float16  — segmentation mask
    fused            (256, 256, 256) float16  — stacker probability map
    agreement        (256, 256, 256) float32  — int 0..7 (count of bases > 0.5)
    individual_<n>   (256, 256, 256) float16  — each base's probability map

Also rebuilds `hf_space/data/manifest.json` with the new architecture
metadata (`num_base_models: 7`, includes swin_unetr, threshold 0.55).

Usage:
    python scripts/inference/precompute_hf_demo.py
    python scripts/inference/precompute_hf_demo.py --case CASE_ID       # one case only (smoke test)
    python scripts/inference/precompute_hf_demo.py --device cuda:1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT))

from jannus.segmentation.pipeline import BrainMetPipeline
from jannus.segmentation.stacking import STACKING_MODEL_NAMES

DATA_ROOT = Path(os.environ.get("JANNUS_DATA_ROOT", PROJECT.parent / "Data")) / "preprocessed_256" / "train"
HF_DATA_DIR = PROJECT / "hf_space" / "data"
DEFAULT_CONFIG = PROJECT / "configs" / "models.yaml"

# Cases to precompute for the public demo.
#
# v1.50: this was a hardcoded list of eight third-party cohort identifiers
# (UCSF-BMSR / BrainMetShare). Those are de-identified public research datasets,
# but embedding another cohort's case IDs in a public repository is not
# something to do incidentally, and the list was meaningless at any other site.
# Supply your own via JANNUS_DEMO_CASES (comma-separated) or --case.
DEMO_CASES = [
    case.strip()
    for case in os.environ.get("JANNUS_DEMO_CASES", "").split(",")
    if case.strip()
]
THRESHOLD = 0.55  # production binarisation threshold, matches inference.threshold
AGREEMENT_THRESHOLD = 0.5  # per-base "fired" threshold for the agreement map


def load_case(case_dir: Path):
    """Load a 4-channel z-scored volume and the ground-truth mask."""
    sequences = ["t1_pre", "t1_gd", "flair", "t2"]
    channels = []
    affine = None
    for seq in sequences:
        nii = nib.load(str(case_dir / f"{seq}.nii.gz"))
        if affine is None:
            affine = nii.affine
        img = np.asarray(nii.dataobj, dtype=np.float32)
        mean, std = img.mean(), img.std()
        if std > 0:
            img = (img - mean) / std
        channels.append(img)
    volume = np.stack(channels, axis=0)  # (4, H, W, D)

    seg_nii = nib.load(str(case_dir / "seg.nii.gz"))
    ground_truth = np.asarray(seg_nii.dataobj, dtype=np.float32)
    ground_truth = (ground_truth > 0.5).astype(np.float32)

    # t1_gd before z-score for visualization (rescale to 0..1 for display)
    t1_gd_raw = np.asarray(nib.load(str(case_dir / "t1_gd.nii.gz")).dataobj,
                           dtype=np.float32)
    t1_gd_min, t1_gd_max = t1_gd_raw.min(), t1_gd_raw.max()
    if t1_gd_max > t1_gd_min:
        t1_gd_disp = (t1_gd_raw - t1_gd_min) / (t1_gd_max - t1_gd_min)
    else:
        t1_gd_disp = t1_gd_raw

    return volume, ground_truth, t1_gd_disp, affine


def compute_dice(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Binary Dice; both inputs assumed already binarized (0/1)."""
    p = (pred_mask > 0.5).astype(np.float32)
    g = (gt_mask > 0.5).astype(np.float32)
    intersection = float((p * g).sum())
    denom = float(p.sum() + g.sum())
    if denom == 0:
        return 1.0  # both empty → perfect by convention
    return (2 * intersection) / denom


def precompute_case(pipeline: BrainMetPipeline, case_name: str,
                    out_dir: Path, force: bool = False) -> dict:
    """Run pipeline on one case; write .npz; return summary dict."""
    case_dir = DATA_ROOT / case_name
    out_path = out_dir / f"{case_name}.npz"

    if out_path.exists() and not force:
        print(f"  [skip] {case_name} (file exists; use --force to overwrite)")
        existing = np.load(str(out_path))
        return {
            "case_name": case_name,
            "dice_score": compute_dice(
                (existing["fused"] > THRESHOLD).astype(np.float32),
                existing["ground_truth"],
            ),
            "shape": list(existing["t1_gd"].shape),
            "skipped": True,
        }

    print(f"  loading {case_name} from {case_dir}")
    t0 = time.time()
    volume, ground_truth, t1_gd_disp, _ = load_case(case_dir)
    print(f"    volume shape: {volume.shape}, GT positives: {int(ground_truth.sum())}")

    print(f"  running pipeline...")
    t1 = time.time()
    result = pipeline.predict_volume(volume)
    elapsed = time.time() - t1
    print(f"    inference: {elapsed:.1f}s, lesions: {result.lesion_count}")

    fused = result.probability_map[0].astype(np.float32)  # (H, W, D)
    pred_mask = (fused > THRESHOLD).astype(np.float32)
    dice = compute_dice(pred_mask, ground_truth)

    # agreement map: count of bases with prob > 0.5 at each voxel
    base_arrays = []
    for name in STACKING_MODEL_NAMES:
        if name not in result.base_probs:
            raise RuntimeError(f"missing base prob '{name}' in result.base_probs")
        base_arrays.append(result.base_probs[name])
    agreement = np.sum(
        np.stack([(b > AGREEMENT_THRESHOLD).astype(np.uint8) for b in base_arrays], axis=0),
        axis=0,
    ).astype(np.float32)  # int values 0..7 stored as float for app.py compatibility

    save_dict = {
        "t1_gd": t1_gd_disp.astype(np.float16),
        "ground_truth": ground_truth.astype(np.float16),
        "fused": fused.astype(np.float16),
        "agreement": agreement,
    }
    for name, arr in result.base_probs.items():
        save_dict[f"individual_{name}"] = arr.astype(np.float16)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out_path), **save_dict)
    size_mb = out_path.stat().st_size / 1e6
    total = time.time() - t0
    print(f"  [done] {case_name}: Dice={dice:.4f}, {size_mb:.1f} MB, total {total:.1f}s")

    return {
        "case_name": case_name,
        "dice_score": dice,
        "shape": list(t1_gd_disp.shape),
        "lesion_count": result.lesion_count,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--case", action="append",
                        help="case name to process (default: all 8 demo cases). "
                             "Pass multiple times for a subset.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG),
                        help=f"path to models.yaml (default: {DEFAULT_CONFIG})")
    parser.add_argument("--device", default=None,
                        help="torch device (default: cuda if available)")
    parser.add_argument("--force", action="store_true",
                        help="overwrite existing .npz files")
    parser.add_argument("--no-manifest", action="store_true",
                        help="skip rewriting manifest.json (single-case smoke test)")
    args = parser.parse_args()

    cases = args.case if args.case else list(DEMO_CASES)
    device = torch.device(args.device) if args.device else (
        torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    print("=" * 60)
    print("  HF Space demo precompute (7-base + StackingClassifierV2)")
    print("=" * 60)
    print(f"  Device: {device}")
    print(f"  Config: {args.config}")
    print(f"  Cases:  {len(cases)} ({', '.join(cases) if len(cases) <= 4 else f'{cases[0]}..{cases[-1]}'})")
    print(f"  Out:    {HF_DATA_DIR}")
    print(f"  Threshold: {THRESHOLD}")
    print()

    print("Building pipeline...")
    pipeline = BrainMetPipeline.from_config(args.config, device=device)
    print(f"  Bases: {[a.name for a in pipeline.model_adapters]}")
    print()

    summaries = []
    t_total = time.time()
    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {case}")
        try:
            summary = precompute_case(pipeline, case, HF_DATA_DIR, force=args.force)
            summaries.append(summary)
        except Exception as e:
            print(f"  [error] {case}: {e}")
            import traceback; traceback.print_exc()
    elapsed_total = time.time() - t_total

    if not args.no_manifest and len(cases) == len(DEMO_CASES):
        # Sort by dice descending so the demo's "highlight reel" stays sorted
        manifest_cases = sorted(
            [{"case_name": s["case_name"], "dice_score": s["dice_score"],
              "shape": s["shape"]} for s in summaries],
            key=lambda c: c["dice_score"], reverse=True,
        )
        manifest = {
            "cases": manifest_cases,
            "ensemble_models": list(STACKING_MODEL_NAMES),
            "threshold": THRESHOLD,
            "model_type": "stacking",
            "stacker_architecture": "StackingClassifierV2",
            "num_base_models": len(STACKING_MODEL_NAMES),
        }
        manifest_path = HF_DATA_DIR / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"\n  Wrote manifest: {manifest_path}")

    print()
    print("=" * 60)
    print(f"  Done in {elapsed_total:.1f}s ({len(summaries)}/{len(cases)} cases)")
    for s in summaries:
        print(f"    {s['case_name']}: Dice={s['dice_score']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
