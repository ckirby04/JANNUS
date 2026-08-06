"""
Seven-model stacker: old 6-model ensemble + new SwinUNETR 150ep+.

Bases (all at 256^3, native-preprocessed):
  From model/stacking_cache_v5/ (legacy 6-model, properly preprocessed):
    nnunet (3D), nnunet_2d, patch_8, patch_12, patch_24, patch_36
  From model/stacking_cache_v5_256/ (refreshed with 0.7915-val SwinUNETR):
    swin_unetr

Stacker input: 9 channels (7 preds + variance + range)
Architecture:  StackingClassifierV2 with in_channels=9
               (1x1x1 bottleneck + wider/deeper trunk + multi-scale branch + SE
               attention; ~700k params vs ~27k for the original StackingClassifier)

Hypothesis: the old 6-model tops the aggregate Dice leaderboard (0.7778);
today's v2 single stacker tops tiny-lesion Dice (0.4529). A 7-model stacker
with both capabilities should keep both advantages.

Usage:
    python scripts/evaluation/seven_model_stacker.py --num-workers 8
    python scripts/evaluation/seven_model_stacker.py --skip-train --num-workers 8
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_dilation, generate_binary_structure
from scipy.ndimage import label as ndimage_label
from torch.utils.data import DataLoader, Dataset

PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT))

from jannus.segmentation.stacking import (
    StackingClassifierV2,
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
from scripts.evaluation.full_stacking_comparison_v2 import (
    OLD_CACHE, NEW_CACHE, OLD_PER_CASE_PATH, OLD_RESULTS_PATH, BAD_CACHE_IDS,
    voxel_dice, voxel_sens, voxel_prec, voxel_spec, relaxed_dice,
    per_lesion_dice, bucket,
)

CHECKPOINT_PATH = PROJECT / "model" / "stacking_classifier_production.pth"
OUTPUT_PATH = PROJECT / "model" / "evaluation_results" / "seven_model_stacker_results.json"

# 7 bases: 6 from old cache + swin_unetr from new cache
IN_CHANNELS = 9              # 7 preds + variance + range
STACKING_PATCH = 32
STACKING_OVERLAP = 0.5
MIN_COMPONENT_SIZE = 0


def load_seven_preds(case_id: str):
    """Returns (preds, mask) where preds.shape = (7, 256, 256, 256)."""
    old = np.load(OLD_CACHE / f"{case_id}.npz")
    new = np.load(NEW_CACHE / f"{case_id}.npz")
    preds = np.stack([
        old["nnunet"].astype(np.float32),
        old["nnunet_2d"].astype(np.float32),
        old["patch_8"].astype(np.float32),
        old["patch_12"].astype(np.float32),
        old["patch_24"].astype(np.float32),
        old["patch_36"].astype(np.float32),
        new["swin_unetr"].astype(np.float32),
    ], axis=0)
    mask = old["mask"].astype(np.float32)
    return preds, mask


def _is_legacy_seven(case_id: str) -> bool:
    """Legacy cache must have all 6 old keys + not corrupted."""
    if case_id in BAD_CACHE_IDS:
        return False
    try:
        d = np.load(OLD_CACHE / f"{case_id}.npz")
        keys = set(d.keys())
        required = {"nnunet", "nnunet_2d", "patch_8", "patch_12", "patch_24", "patch_36"}
        if not required.issubset(keys):
            return False
        return d["nnunet"].shape == (256, 256, 256)
    except Exception:
        return False


def build_features_9ch(preds: np.ndarray) -> np.ndarray:
    """(7, H, W, D) -> (9, H, W, D) by appending variance and range."""
    variance = preds.var(axis=0, keepdims=True)
    range_map = preds.max(axis=0, keepdims=True) - preds.min(axis=0, keepdims=True)
    return np.concatenate([preds, variance, range_map], axis=0)


def _augment_patch(feat: np.ndarray, mask: np.ndarray,
                   rng: np.random.Generator) -> tuple:
    """Symmetric 3D augmentation: random flips on 3 spatial axes, random 90
    rotation in a random plane, small per-channel intensity jitter. Shapes
    preserved. feat: (C, p, p, p), mask: (1, p, p, p)."""
    # Random flips (each axis independently, 50% probability)
    for ax in range(3):
        if rng.random() < 0.5:
            feat = np.flip(feat, axis=ax + 1)
            mask = np.flip(mask, axis=ax + 1)
    # Random 90 rotation in a random spatial plane (50% probability)
    if rng.random() < 0.5:
        plane = rng.choice([(0, 1), (0, 2), (1, 2)])
        k_rot = int(rng.integers(1, 4))  # 90, 180, or 270 degrees
        feat = np.rot90(feat, k=k_rot, axes=(plane[0] + 1, plane[1] + 1))
        mask = np.rot90(mask, k=k_rot, axes=(plane[0] + 1, plane[1] + 1))
    # Small intensity jitter on base preds (uniform multiplicative)
    if rng.random() < 0.5:
        jitter = rng.uniform(0.95, 1.05, size=(feat.shape[0], 1, 1, 1)).astype(np.float32)
        feat = np.clip(feat * jitter, 0.0, 1.0)
    return np.ascontiguousarray(feat), np.ascontiguousarray(mask)


class SevenModelDataset(Dataset):
    def __init__(self, case_ids: List[str], patch_size: int = 32,
                 patches_per_case: int = 10, fg_ratio: float = 0.7,
                 augment: bool = False):
        self.case_ids = case_ids
        self.patch_size = patch_size
        self.K = patches_per_case
        self.fg_ratio = fg_ratio
        self.augment = augment
        self.rng = np.random.default_rng()

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, idx):
        preds, mask = load_seven_preds(self.case_ids[idx])
        features = build_features_9ch(preds)
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
            f = features[:, z0:z0+p, y0:y0+p, x0:x0+p]
            m = mask[z0:z0+p, y0:y0+p, x0:x0+p][None]
            if self.augment:
                f, m = _augment_patch(f, m, self.rng)
            feat_patches[k] = f
            mask_patches[k] = m
        return torch.from_numpy(feat_patches), torch.from_numpy(mask_patches)


def _bce_dice(logits, target, smooth=1.0):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    probs = torch.sigmoid(logits)
    inter = (probs * target).sum(dim=(1, 2, 3, 4))
    denom = probs.sum(dim=(1, 2, 3, 4)) + target.sum(dim=(1, 2, 3, 4))
    dice = (2 * inter + smooth) / (denom + smooth)
    return bce + (1.0 - dice.mean())


def train(eval_ids: set, epochs: int, batch_size: int, lr: float,
          num_workers: int, dropout: float = 0.1, weight_decay: float = 1e-5,
          warmup_epochs: int = 0, save_min_delta: float = 0.0,
          augment: bool = False):
    print("\n" + "=" * 70)
    print(f"  STEP 1: Train 9-channel 7-model stacker")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_ids = sorted({p.stem for p in OLD_CACHE.glob("*.npz")}
                     & {p.stem for p in NEW_CACHE.glob("*.npz")})
    print(f"  Filtering to cases with all 7 preds at 256^3...")
    all_ids = [c for c in all_ids if _is_legacy_seven(c)]
    train_ids = [c for c in all_ids if c not in eval_ids]
    split = int(0.9 * len(train_ids))
    t_ids, v_ids = train_ids[:split], train_ids[split:]
    print(f"  Total cases: {len(all_ids)}")
    print(f"  Train: {len(t_ids)}, internal val: {len(v_ids)}, eval held out: {len(eval_ids)}")

    train_ds = SevenModelDataset(t_ids, augment=augment)
    val_ds = SevenModelDataset(v_ids, augment=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, persistent_workers=(num_workers > 0))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, persistent_workers=(num_workers > 0))
    print(f"  Batch: {batch_size}, patches/case: 10, workers: {num_workers}, "
          f"augment: {augment}")
    print(f"  lr: {lr}, warmup_epochs: {warmup_epochs}, weight_decay: {weight_decay}, "
          f"dropout: {dropout}")
    print(f"  save-best min-delta: {save_min_delta}")
    print(f"  Iters/epoch: {len(train_loader)} train / {len(val_loader)} val")

    model = StackingClassifierV2(in_channels=IN_CHANNELS, dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Warmup then cosine decay. If warmup_epochs=0, pure cosine.
    if warmup_epochs > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=1e-3, end_factor=1.0, total_iters=warmup_epochs)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(epochs - warmup_epochs, 1))
        sched = torch.optim.lr_scheduler.SequentialLR(
            opt, schedulers=[warmup, cosine], milestones=[warmup_epochs])
    else:
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
            loss.backward(); opt.step()
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

        improvement = best - val_loss
        is_new_best = improvement > save_min_delta
        star = " *" if is_new_best else ""
        lr_now = opt.param_groups[0]['lr']
        print(f"  epoch {epoch:3d}/{epochs}  train={train_loss:.4f}  "
              f"val={val_loss:.4f}  lr={lr_now:.2e}  ({time.time() - t0:.1f}s){star}")
        if is_new_best:
            best = val_loss
            torch.save({"model_state_dict": model.state_dict(),
                        "in_channels": IN_CHANNELS,
                        "epoch": epoch, "val_loss": val_loss}, CHECKPOINT_PATH)

    print(f"  Best val loss: {best:.4f}  ->  {CHECKPOINT_PATH}")


def evaluate(eval_ids: List[str]):
    print("\n" + "=" * 70)
    print(f"  STEP 2: Evaluate 7-model stacker on {len(eval_ids)} cases")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StackingClassifierV2(in_channels=IN_CHANNELS).to(device)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  Loaded {CHECKPOINT_PATH} (epoch {ckpt.get('epoch')}, val {ckpt.get('val_loss'):.4f})")

    with open(OLD_RESULTS_PATH) as f: old = json.load(f)

    prob_maps, masks = [], []
    t0 = time.time()
    for i, cid in enumerate(eval_ids):
        preds, m = load_seven_preds(cid)
        features = build_features_9ch(preds)
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
    print("  Computing FDA-style metrics + size-stratified metrics...")
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
    new_bucket = bucket(all_lesions)
    old_s = old["stacking"]; old_b = old["_dice_by_lesion_size"]["stacking"]
    old_thr = old["_thresholds"]["stacking"]

    print(f"\n{'=' * 70}")
    print(f"  OVERALL METRICS -- {len(dices)} val cases")
    print(f"{'=' * 70}\n")
    print(f"  {'Metric':<22} {'Old 6-model':<14} {'7-model':<14} {'Delta':<10}")
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
    print(f"  {'Bucket':<20} {'Old 6-model':<14} {'7-model':<14} {'Delta':<10} N (old/new)")
    print(f"  {'-' * 68}")
    for lbl in ["tiny (<100 vox)", "small (100-1k)", "medium (1k-10k)", "large (>10k)"]:
        ob = old_b.get(lbl, {}); nb = new_bucket.get(lbl, {})
        od = ob.get("mean_dice", 0.0); nd = nb.get("mean_dice", 0.0); d = nd - od
        print(f"  {lbl:<20} {od:<14.4f} {nd:<14.4f} "
              f"{'+' if d > 0 else ''}{d:<10.4f} {ob.get('n_lesions', 0)} / {nb.get('n_lesions', 0)}")

    print("\n" + format_fda_report(fda_agg))
    print("\n" + format_stratified_report(strat_agg))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump({
            "description": "7-model stacker: 6 legacy bases (nnunet, nnunet_2d, patch_8/12/24/36) + SwinUNETR 150ep+ refreshed. 9-ch input, 256^3 features, same 84 eval cases.",
            "old_6model": {"threshold": old_thr, "lesion_buckets": old_b,
                           **{k: old_s[k] for k in ["dice", "std_dice", "sensitivity",
                                                      "precision", "specificity", "relaxed_dice_2"]}},
            "new_7model": {"in_channels": IN_CHANNELS,
                           "bases": ["nnunet_3d", "nnunet_2d",
                                      "patch_8", "patch_12", "patch_24", "patch_36",
                                      "swin_unetr_150ep_plus"],
                           **new, "lesion_buckets": new_bucket,
                           "fda_metrics": fda_agg,
                           "stratified_by_size": strat_agg},
            "eval_cases": eval_ids,
        }, f, indent=2, default=str)
    print(f"\n  Results -> {OUTPUT_PATH}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.1,
                    help="Dropout in StackingClassifierV2 residual blocks")
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--warmup-epochs", type=int, default=0,
                    help="Linear LR warmup for N epochs before cosine decay")
    ap.add_argument("--save-min-delta", type=float, default=0.0,
                    help="New best must beat old best by >= this amount. "
                         "Prevents lucky-val-epoch spurious saves.")
    ap.add_argument("--augment", action="store_true",
                    help="Apply 3D flip/rotation/intensity-jitter patch augmentation")
    args = ap.parse_args()

    with open(OLD_PER_CASE_PATH) as f:
        eval_ids = sorted(json.load(f).keys())
    print(f"  Eval cases: {len(eval_ids)}")

    t = time.time()
    if not args.skip_train:
        train(set(eval_ids), args.epochs, args.batch_size, args.lr,
              args.num_workers, dropout=args.dropout,
              weight_decay=args.weight_decay,
              warmup_epochs=args.warmup_epochs,
              save_min_delta=args.save_min_delta,
              augment=args.augment)
    evaluate(eval_ids)
    print(f"\n  Total: {(time.time() - t) / 60:.1f} min")


if __name__ == "__main__":
    main()
