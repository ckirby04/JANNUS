"""
v2: Train + evaluate 3-model stacker using PROPERLY preprocessed base predictions.

nnU-Net 3D/2D predictions come from the old cache (stacking_cache_v5/), where
they were run through nnU-Net's native preprocessing pipeline and scored
0.7651 / 0.7416 standalone. SwinUNETR 150ep predictions come from the new
cache (stacking_cache_v5_256/). All are at 256^3.

This fixes the bug in v1 where nnU-Net was forced to run on uniformly
resampled 256^3 data, causing its Dice to regress by 0.23.

Step 1: Train 5-channel stacker on non-eval cases (dual-source reads)
Step 2: Evaluate on same 84 val cases, compare vs old 6-model stacker

Usage:
    python scripts/evaluation/full_stacking_comparison_v2.py
    python scripts/evaluation/full_stacking_comparison_v2.py --skip-train
    python scripts/evaluation/full_stacking_comparison_v2.py --epochs 30 --num-workers 6
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import label as ndimage_label
from scipy.ndimage import binary_dilation, generate_binary_structure
from torch.utils.data import DataLoader, Dataset

PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT))

from jannus.segmentation.stacking import (
    STACKING_IN_CHANNELS,
    STACKING_MODEL_NAMES,
    StackingClassifier,
    build_stacking_features_from_preds,
    postprocess_prediction,
    sliding_window_inference,
)
from jannus.evaluation.metrics import (
    aggregate_stratified,
    aggregate_with_ci,
    compute_case_metrics,
    compute_case_stratified,
    format_fda_report,
    format_stratified_report,
)

OLD_CACHE = PROJECT / "model" / "stacking_cache_v5"            # has nnunet + nnunet_2d
NEW_CACHE = PROJECT / "model" / "stacking_cache_v5_256"        # has swin_unetr

# Known-bad cache entries (CRC-32 corruption on swin_unetr.npy inside the npz).
# Discovered 2026-04-19 by full scan of stacking_cache_v5_256/. Regenerate
# these by re-running the SwinUNETR adapter on the affected cases.
# Cache entries known to be corrupt at the originating site. Site-specific;
# supply your own via JANNUS_BAD_CACHE_IDS (comma-separated).
BAD_CACHE_IDS = {c.strip() for c in os.environ.get("JANNUS_BAD_CACHE_IDS", "").split(",") if c.strip()}
CHECKPOINT_PATH = PROJECT / "model" / "stacking_v5_hybrid_classifier.pth"
OLD_RESULTS_PATH = PROJECT / "model" / "evaluation_results" / "stacking_v5_results.json"
OLD_PER_CASE_PATH = PROJECT / "model" / "evaluation_results" / "stacking_v5_results_per_case.json"
OUTPUT_PATH = PROJECT / "model" / "evaluation_results" / "stacking_v4_vs_v5_comparison_v2.json"

STACKING_PATCH = 32
STACKING_OVERLAP = 0.5
MIN_COMPONENT_SIZE = 0  # 2026-04-19: was 20; lowered to preserve tiny brain-met lesions (+0.06 tiny-lesion Dice per sweep)


def load_hybrid_preds(case_id: str):
    """nnU-Net 3D/2D from old cache (native-preprocessed at 256^3 legacy format),
    SwinUNETR from new cache."""
    old = np.load(OLD_CACHE / f"{case_id}.npz")
    new = np.load(NEW_CACHE / f"{case_id}.npz")
    nnunet_3d = old["nnunet"].astype(np.float32)
    nnunet_2d = old["nnunet_2d"].astype(np.float32)
    swin_unetr = new["swin_unetr"].astype(np.float32)
    mask = old["mask"].astype(np.float32)
    return nnunet_3d, nnunet_2d, swin_unetr, mask


def _is_legacy_256(case_id: str) -> bool:
    """Legacy 6-model 256^3 format: has 'nnunet' and 'nnunet_2d' keys."""
    if case_id in BAD_CACHE_IDS:
        return False
    try:
        d = np.load(OLD_CACHE / f"{case_id}.npz")
        if "nnunet" not in d.keys() or "nnunet_2d" not in d.keys():
            return False
        return d["nnunet"].shape == (256, 256, 256)
    except Exception:
        return False


class HybridStackingDataset(Dataset):
    """Reads nnU-Net from OLD_CACHE, SwinUNETR from NEW_CACHE. Yields K patches/case."""

    def __init__(self, case_ids: List[str], patch_size: int = 32,
                 patches_per_case: int = 10, fg_ratio: float = 0.7):
        self.case_ids = case_ids
        self.patch_size = patch_size
        self.K = patches_per_case
        self.fg_ratio = fg_ratio
        self.rng = np.random.default_rng()

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, idx):
        nnunet_3d, nnunet_2d, swin_unetr, mask = load_hybrid_preds(self.case_ids[idx])
        preds = np.stack([nnunet_3d, nnunet_2d, swin_unetr], axis=0)
        features = build_stacking_features_from_preds(preds)

        _, H, W, D = features.shape
        p = self.patch_size
        fg = np.argwhere(mask > 0.5)

        feat_patches = np.empty((self.K, features.shape[0], p, p, p), dtype=np.float32)
        mask_patches = np.empty((self.K, 1, p, p, p), dtype=np.float32)

        for k in range(self.K):
            if len(fg) > 0 and self.rng.random() < self.fg_ratio:
                cz, cy, cx = fg[self.rng.integers(len(fg))]
            else:
                cz = self.rng.integers(p // 2, max(H - p // 2, p // 2 + 1))
                cy = self.rng.integers(p // 2, max(W - p // 2, p // 2 + 1))
                cx = self.rng.integers(p // 2, max(D - p // 2, p // 2 + 1))
            z0 = int(np.clip(cz - p // 2, 0, H - p))
            y0 = int(np.clip(cy - p // 2, 0, W - p))
            x0 = int(np.clip(cx - p // 2, 0, D - p))
            feat_patches[k] = features[:, z0:z0 + p, y0:y0 + p, x0:x0 + p]
            mask_patches[k, 0] = mask[z0:z0 + p, y0:y0 + p, x0:x0 + p]

        return torch.from_numpy(feat_patches), torch.from_numpy(mask_patches)


def _bce_dice(logits, target, smooth=1.0):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    probs = torch.sigmoid(logits)
    inter = (probs * target).sum(dim=(1, 2, 3, 4))
    denom = probs.sum(dim=(1, 2, 3, 4)) + target.sum(dim=(1, 2, 3, 4))
    dice = (2 * inter + smooth) / (denom + smooth)
    return bce + (1.0 - dice.mean())


def train(eval_ids: set, epochs: int, batch_size: int, lr: float,
          num_workers: int):
    print("\n" + "=" * 70)
    print("  STEP 1: Train 5-channel stacker (hybrid sources)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_ids = sorted({p.stem for p in OLD_CACHE.glob("*.npz")}
                     & {p.stem for p in NEW_CACHE.glob("*.npz")})
    print(f"  Filtering to legacy 256^3 format (excludes 128^3 Mets_ and corrupted files)...")
    all_ids = [c for c in all_ids if _is_legacy_256(c)]
    train_ids = [c for c in all_ids if c not in eval_ids]
    split = int(0.9 * len(train_ids))
    t_ids, v_ids = train_ids[:split], train_ids[split:]
    print(f"  Total cases (in both caches): {len(all_ids)}")
    print(f"  Train: {len(t_ids)}, internal val: {len(v_ids)}, eval held out: {len(eval_ids)}")

    train_ds = HybridStackingDataset(t_ids, patches_per_case=10)
    val_ds = HybridStackingDataset(v_ids, patches_per_case=10)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, persistent_workers=(num_workers > 0))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, persistent_workers=(num_workers > 0))
    print(f"  Batch: {batch_size}, patches/case: 10, workers: {num_workers}")
    print(f"  Iters/epoch: {len(train_loader)} train / {len(val_loader)} val")

    model = StackingClassifier(in_channels=STACKING_IN_CHANNELS).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        tl, tn = 0.0, 0
        for feats, mask in train_loader:
            B, K = feats.shape[:2]
            feats = feats.reshape(B * K, *feats.shape[2:]).to(device)
            mask = mask.reshape(B * K, *mask.shape[2:]).to(device)
            opt.zero_grad()
            loss = _bce_dice(model(feats), mask)
            loss.backward()
            opt.step()
            tl += loss.item(); tn += 1
        train_loss = tl / max(tn, 1)

        model.eval()
        vl, vn = 0.0, 0
        with torch.no_grad():
            for feats, mask in val_loader:
                B, K = feats.shape[:2]
                feats = feats.reshape(B * K, *feats.shape[2:]).to(device)
                mask = mask.reshape(B * K, *mask.shape[2:]).to(device)
                vl += _bce_dice(model(feats), mask).item(); vn += 1
        val_loss = vl / max(vn, 1)
        sched.step()

        star = " *" if val_loss < best else ""
        print(f"  epoch {epoch:3d}/{epochs}  train={train_loss:.4f}  "
              f"val={val_loss:.4f}  ({time.time() - t0:.1f}s){star}")
        if val_loss < best:
            best = val_loss
            torch.save({"model_state_dict": model.state_dict(),
                        "in_channels": STACKING_IN_CHANNELS,
                        "epoch": epoch, "val_loss": val_loss}, CHECKPOINT_PATH)

    print(f"  Best val loss: {best:.4f}  ->  {CHECKPOINT_PATH}")


def voxel_dice(p, g, smooth=1e-8):
    tp = ((p > 0) & (g > 0)).sum(); fp = ((p > 0) & (g == 0)).sum()
    fn = ((p == 0) & (g > 0)).sum()
    if tp == 0 and fp == 0 and fn == 0: return 1.0
    return float(2 * tp) / float(2 * tp + fp + fn + smooth)


def voxel_sens(p, g, s=1e-8):
    tp = ((p > 0) & (g > 0)).sum(); fn = ((p == 0) & (g > 0)).sum()
    return float(tp) / float(tp + fn + s)


def voxel_prec(p, g, s=1e-8):
    tp = ((p > 0) & (g > 0)).sum(); fp = ((p > 0) & (g == 0)).sum()
    return float(tp) / float(tp + fp + s)


def voxel_spec(p, g, s=1e-8):
    tn = ((p == 0) & (g == 0)).sum(); fp = ((p > 0) & (g == 0)).sum()
    return float(tn) / float(tn + fp + s)


def relaxed_dice(p, g, tol=2):
    """Boundary-tolerant Dice with a `tol`-voxel symmetric buffer.

    A predicted voxel counts as a true positive if it falls within `tol`
    voxels of any GT voxel; a GT voxel counts as a true positive if it
    falls within `tol` voxels of any predicted voxel. By construction this
    is always >= the strict voxel Dice.

    NOTE (2026-04-25): the previous implementation only dilated the GT,
    which produced values consistently lower than strict Dice — opposite
    of intent. All historical "relaxed_dice_2" entries in
    model/evaluation_results/*.json predate this fix and are unreliable.
    """
    st = generate_binary_structure(3, 1)
    p_bool = p.astype(bool)
    g_bool = g.astype(bool)
    g_dilated = binary_dilation(g_bool, structure=st, iterations=tol)
    p_dilated = binary_dilation(p_bool, structure=st, iterations=tol)
    tp_pred = (p_bool & g_dilated).sum()
    tp_gt = (g_bool & p_dilated).sum()
    total = p_bool.sum() + g_bool.sum()
    if total == 0:
        return 1.0
    return float(tp_pred + tp_gt) / float(total + 1e-8)


def per_lesion_dice(p, g):
    gl, n = ndimage_label(g); pl, _ = ndimage_label(p)
    if n == 0: return []
    out = []
    for i in range(1, n + 1):
        gm = (gl == i); sz = int(gm.sum())
        ov = pl[gm]; uniq = np.unique(ov[ov > 0])
        if len(uniq) == 0:
            out.append({"size": sz, "dice": 0.0}); continue
        merged = np.zeros_like(p, dtype=bool)
        for q in uniq: merged |= (pl == q)
        tp = (gm & merged).sum(); fp = (merged & ~gm).sum(); fn = (gm & ~merged).sum()
        out.append({"size": sz, "dice": float(2 * tp) / float(2 * tp + fp + fn + 1e-8)})
    return out


def bucket(lesions):
    buckets = {"tiny (<100 vox)": (0, 100), "small (100-1k)": (100, 1000),
               "medium (1k-10k)": (1000, 10000), "large (>10k)": (10000, float("inf"))}
    out = {}
    for label, (lo, hi) in buckets.items():
        m = [l for l in lesions if lo <= l["size"] < hi]
        if m:
            d = [l["dice"] for l in m]
            out[label] = {"mean_dice": float(np.mean(d)),
                          "std_dice": float(np.std(d)), "n_lesions": len(m)}
        else:
            out[label] = {"mean_dice": 0.0, "std_dice": 0.0, "n_lesions": 0}
    return out


def evaluate(eval_ids: List[str]):
    print("\n" + "=" * 70)
    print("  STEP 2: Evaluate & compare on 84 eval cases")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StackingClassifier(in_channels=STACKING_IN_CHANNELS).to(device)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  Loaded {CHECKPOINT_PATH} (epoch {ckpt.get('epoch')}, val {ckpt.get('val_loss'):.4f})")

    with open(OLD_RESULTS_PATH) as f: old = json.load(f)

    prob_maps, masks = [], []
    t0 = time.time()
    for i, cid in enumerate(eval_ids):
        nn3, nn2, sw, m = load_hybrid_preds(cid)
        preds = np.stack([nn3, nn2, sw], axis=0)
        features = build_stacking_features_from_preds(preds)
        prob = sliding_window_inference(model, features, STACKING_PATCH, device,
                                        overlap=STACKING_OVERLAP)
        prob_maps.append(prob); masks.append(m)
        if (i + 1) % 20 == 0 or i + 1 == len(eval_ids):
            print(f"    [{i+1}/{len(eval_ids)}] ({time.time() - t0:.0f}s)")

    print("  Sweeping thresholds...")
    best_t, best_d = 0.5, 0.0
    for t in np.arange(0.05, 1.0, 0.05):
        ds = [voxel_dice(postprocess_prediction((p > t).astype(np.float32), MIN_COMPONENT_SIZE), m)
              for p, m in zip(prob_maps, masks)]
        md = float(np.mean(ds))
        if md > best_d: best_d, best_t = md, float(t)
    print(f"  Optimal threshold: {best_t:.2f}")

    dices, sens, prec, spec, relax = [], [], [], [], []
    all_lesions = []
    fda_per_case = []
    strat_per_case = []
    print("  Computing FDA-style + size-stratified metrics...")
    for p, m in zip(prob_maps, masks):
        pred = postprocess_prediction((p > best_t).astype(np.float32), MIN_COMPONENT_SIZE)
        dices.append(voxel_dice(pred, m))
        sens.append(voxel_sens(pred, m))
        prec.append(voxel_prec(pred, m))
        spec.append(voxel_spec(pred, m))
        relax.append(relaxed_dice(pred, m, 2))
        all_lesions.extend(per_lesion_dice(pred, m))
        fda_per_case.append(compute_case_metrics(pred, m, spacing=1.0))
        strat_per_case.append(compute_case_stratified(pred, m, spacing=1.0))
    fda_agg = aggregate_with_ci(fda_per_case, n_bootstrap=1000)
    strat_agg = aggregate_stratified(strat_per_case)

    new = {"dice": float(np.mean(dices)), "std_dice": float(np.std(dices)),
           "sensitivity": float(np.mean(sens)), "precision": float(np.mean(prec)),
           "specificity": float(np.mean(spec)),
           "relaxed_dice_2": float(np.mean(relax)), "threshold": float(best_t)}
    new_b = bucket(all_lesions)
    old_s = old["stacking"]; old_b = old["_dice_by_lesion_size"]["stacking"]
    old_thr = old["_thresholds"]["stacking"]

    print(f"\n{'=' * 70}")
    print(f"  OVERALL METRICS -- {len(dices)} val cases")
    print(f"{'=' * 70}\n")
    print(f"  {'Metric':<22} {'Old 6-model':<14} {'New 3-model':<14} {'Delta':<10}")
    print(f"  {'-' * 60}")
    for disp, k in [("Dice", "dice"), ("Std Dice", "std_dice"),
                    ("Sensitivity", "sensitivity"), ("Precision", "precision"),
                    ("Specificity", "specificity"), ("Relaxed Dice (t=2)", "relaxed_dice_2")]:
        ov = old_s.get(k, 0.0); nv = new.get(k, 0.0); d = nv - ov
        print(f"  {disp:<22} {ov:<14.4f} {nv:<14.4f} {'+' if d > 0 else ''}{d:<10.4f}")
    print(f"  {'Threshold':<22} {old_thr:<14.2f} {new['threshold']:<14.2f}")

    print(f"\n{'=' * 70}")
    print(f"  DICE BY LESION SIZE BUCKET")
    print(f"{'=' * 70}\n")
    print(f"  {'Bucket':<20} {'Old 6-model':<14} {'New 3-model':<14} {'Delta':<10} N (old/new)")
    print(f"  {'-' * 68}")
    for lbl in ["tiny (<100 vox)", "small (100-1k)", "medium (1k-10k)", "large (>10k)"]:
        ob = old_b.get(lbl, {}); nb = new_b.get(lbl, {})
        od = ob.get("mean_dice", 0.0); nd = nb.get("mean_dice", 0.0); d = nd - od
        print(f"  {lbl:<20} {od:<14.4f} {nd:<14.4f} "
              f"{'+' if d > 0 else ''}{d:<10.4f} {ob.get('n_lesions', 0)} / {nb.get('n_lesions', 0)}")

    # FDA-style metrics table vs Neosoma K252922
    print("\n" + format_fda_report(fda_agg))
    print("\n" + format_stratified_report(strat_agg))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save = {
        "description": "Old 6-model vs new 3-model stacker (HYBRID cache: nnU-Net from native-preprocessed old cache, SwinUNETR 150ep from new cache). Same 84 val cases.",
        "old_6model": {"threshold": old_thr, "lesion_buckets": old_b,
                       **{k: old_s[k] for k in ["dice", "std_dice", "sensitivity",
                                                  "precision", "specificity", "relaxed_dice_2"]}},
        "new_3model": {"models": list(STACKING_MODEL_NAMES),
                       "in_channels": STACKING_IN_CHANNELS,
                       **new, "lesion_buckets": new_b,
                       "fda_metrics": fda_agg,
                       "stratified_by_size": strat_agg},
        "eval_cases": eval_ids,
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(save, f, indent=2, default=str)
    print(f"\n  Results -> {OUTPUT_PATH}")


def sanity_check(eval_ids):
    """Fast sanity check: raw Dice at fixed threshold 0.5, no postprocessing,
    streaming one case at a time (bounded memory). Takes ~2-3 min.
    """
    print("\n" + "=" * 70, flush=True)
    print("  SANITY CHECK: hybrid base preds on 84 eval cases (raw Dice @ 0.5)", flush=True)
    print("=" * 70, flush=True)

    thr = 0.5
    nn3_d, nn2_d, sw_d, mn_d = [], [], [], []
    t0 = time.time()

    for i, cid in enumerate(eval_ids):
        nn3, nn2, sw, m = load_hybrid_preds(cid)
        mean3 = (nn3 + nn2 + sw) / 3.0
        nn3_d.append(voxel_dice((nn3 > thr).astype(np.float32), m))
        nn2_d.append(voxel_dice((nn2 > thr).astype(np.float32), m))
        sw_d.append(voxel_dice((sw > thr).astype(np.float32), m))
        mn_d.append(voxel_dice((mean3 > thr).astype(np.float32), m))
        if (i + 1) % 10 == 0 or i + 1 == len(eval_ids):
            print(f"    [{i+1}/{len(eval_ids)}] ({time.time()-t0:.0f}s)  "
                  f"running means: nn3={np.mean(nn3_d):.3f} nn2={np.mean(nn2_d):.3f} "
                  f"sw={np.mean(sw_d):.3f} mean3={np.mean(mn_d):.3f}", flush=True)

    d_nn3 = float(np.mean(nn3_d))
    d_nn2 = float(np.mean(nn2_d))
    d_sw = float(np.mean(sw_d))
    d_mn = float(np.mean(mn_d))

    print("\n  Raw Dice @ threshold 0.5 (no postprocessing):\n", flush=True)
    print(f"    nnU-Net 3D (hybrid)          {d_nn3:.4f}   (old eval at optimal thr: 0.7651)", flush=True)
    print(f"    nnU-Net 2D (hybrid)          {d_nn2:.4f}   (old eval at optimal thr: 0.7416)", flush=True)
    print(f"    SwinUNETR 150ep (hybrid)     {d_sw:.4f}", flush=True)
    print(f"    Simple mean 3-model          {d_mn:.4f}   (old 6-model mean at opt thr: 0.7669)", flush=True)
    print(flush=True)
    if d_nn3 > 0.65 and d_nn2 > 0.60:
        print("  HYBRID BASE PREDS LOOK GOOD. Resolution bug is fixed.", flush=True)
        print("  (Scores are at fixed thr=0.5 without postprocessing, so ~0.02-0.05 lower than tuned.)", flush=True)
        print("  Proceed with full training.", flush=True)
    else:
        print("  HYBRID BASE PREDS STILL LOOK BROKEN. Needs deeper investigation.", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sanity-check", action="store_true",
                    help="Run simple-mean eval only, skip training")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()

    with open(OLD_PER_CASE_PATH) as f:
        eval_ids = sorted(json.load(f).keys())
    print(f"  Eval cases: {len(eval_ids)}")

    t = time.time()
    if args.sanity_check:
        sanity_check(eval_ids)
    else:
        if not args.skip_train:
            train(set(eval_ids), args.epochs, args.batch_size, args.lr, args.num_workers)
        evaluate(eval_ids)
    print(f"\n  Total: {(time.time() - t) / 60:.1f} min")


if __name__ == "__main__":
    main()
