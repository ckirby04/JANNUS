"""
Two-phase full-dataset fine-tune of SwinUNETR on BrainMetShare / UCSF.

Goal
----
Bring SwinUNETR's Dice close to nnU-Net 3D's 0.762 baseline so it's a
genuinely useful meta-learner input rather than a weak ensemble member.

Training regime (matches nnU-Net 3D's setup apples-to-apples)
-------------------------------------------------------------
* **Data source**: `nnUNet/nnUNet_raw/Dataset001_BrainMets/` — all 3,120
  cases (real + copy-paste synthetic + HBLI + pseudo-labeled Yale),
  read via `NnUNetRawDataset`.
* **Split**: nnU-Net's canonical `splits_final.json` fold 0 (2,333 train
  / 584 val), so the val Dice is directly comparable to nnU-Net 3D's
  fold-0 validation metrics.
* **Sampling**: 96^3 random patches with 2/3 foreground bias, ~2 samples
  per case per epoch — the same sampling nnU-Net uses internally.
* **Optimizer**:
    Phase 1 — encoder frozen, AdamW on decoder params only, lr 1e-4.
              Short (20 epochs default) — just to let the 4-channel stem
              + 2-class head catch up to the BTCV-pretrained trunk.
    Phase 2 — full fine-tune, two param groups:
              decoder lr 1e-4, encoder lr 1e-5 (decoder/10).
              Long (500 epochs default) — this is where the ~0.76 Dice
              gains come from. nnU-Net 3D used 1000 epochs but with
              ~250 iters/epoch; we use ~2×cases iters/epoch so 500 is
              the rough equivalent.
* **Loss**: Dice + CE, weighted 0.7 / 0.3.
* **AMP**: enabled on CUDA; CPU runs fall back to fp32 silently.
* **Checkpointing**: best val Dice -> `model/swin_unetr_brainmets.pth`.
  Restartable with `--resume`.

Usage
-----
    # Full run (phase 1 -> phase 2 end-to-end). Use on GPU.
    python scripts/training/train_swin_unetr.py

    # Resume an interrupted run from the last best checkpoint
    python scripts/training/train_swin_unetr.py --resume

    # Phase 2 only after a phase-1 checkpoint is already saved
    python scripts/training/train_swin_unetr.py --phase 2 --resume

    # Custom epoch counts
    python scripts/training/train_swin_unetr.py \
        --epochs-phase1 30 --epochs-phase2 600
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT))

from jannus.segmentation.nnunet_raw_dataset import (
    NnUNetRawDataset,
    load_nnunet_split,
)
from jannus.segmentation.swin_unetr_model import (
    build_swin_unetr_from_dict,
    load_pretrained_backbone,
    make_two_phase_optimizer,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

NNUNET_RAW = PROJECT / "nnUNet" / "nnUNet_raw" / "Dataset001_BrainMets"
NNUNET_SPLITS = (PROJECT / "nnUNet" / "nnUNet_preprocessed"
                 / "Dataset001_BrainMets" / "splits_final.json")

CONFIG_PATH = PROJECT / "configs" / "models.yaml"
MODEL_OUT_DIR = PROJECT / "model"
# Two checkpoint files on purpose:
#   - BEST   = model weights only, for inference. Overwritten on every
#              val-Dice improvement. Small file, clean.
#   - LATEST = full training state (model + optimizer + scaler + phase +
#              epoch + best_val_dice). Overwritten EVERY epoch so that
#              `--resume` can pick up mid-phase with correct momentum.
MODEL_OUT_PATH = MODEL_OUT_DIR / "swin_unetr_brainmets.pth"
LATEST_PATH = MODEL_OUT_DIR / "swin_unetr_latest.pth"
STATE_PATH = MODEL_OUT_DIR / "swin_unetr_training_state.json"
# Overridable per-run via --output-suffix so we can train multiple variants
# (e.g. precision-focused) without clobbering the main checkpoints.

# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


class DiceCELoss(nn.Module):
    """Soft-Dice + cross-entropy for 2-class segmentation."""

    def __init__(self, dice_weight: float = 0.7, ce_weight: float = 0.3,
                 smooth: float = 1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.smooth = smooth
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        target = mask.squeeze(1).long()  # (N, H, W, D)
        ce_loss = self.ce(logits, target)

        probs = F.softmax(logits, dim=1)[:, 1]
        target_f = target.float()
        inter = (probs * target_f).sum(dim=(1, 2, 3))
        denom = probs.sum(dim=(1, 2, 3)) + target_f.sum(dim=(1, 2, 3))
        dice = (2.0 * inter + self.smooth) / (denom + self.smooth)
        dice_loss = 1.0 - dice.mean()

        return self.dice_weight * dice_loss + self.ce_weight * ce_loss


class DicePrecisionLoss(nn.Module):
    """Soft-Dice + precision-penalty loss for 2-class segmentation.

    Trades some sensitivity for reduced false-positive voxels by explicitly
    penalizing probability mass outside GT. Used to train a precision-focused
    SwinUNETR base for the stacker ensemble.
    """

    def __init__(self, dice_weight: float = 0.7, precision_weight: float = 0.3,
                 fp_penalty_weight: float = 2.0, smooth: float = 1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.precision_weight = precision_weight
        self.fp_penalty_weight = fp_penalty_weight
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        target = mask.squeeze(1).long()
        target_f = target.float()
        probs = F.softmax(logits, dim=1)[:, 1]

        inter = (probs * target_f).sum(dim=(1, 2, 3))
        denom = probs.sum(dim=(1, 2, 3)) + target_f.sum(dim=(1, 2, 3))
        dice = (2.0 * inter + self.smooth) / (denom + self.smooth)
        dice_loss = 1.0 - dice.mean()

        # Precision: BCE-with-logits (autocast-safe) + explicit FP penalty.
        # For 2-class softmax, the binary logit is logit_1 - logit_0.
        binary_logit = logits[:, 1] - logits[:, 0]
        bce = F.binary_cross_entropy_with_logits(binary_logit, target_f, reduction="mean")
        fp_term = (probs * (1.0 - target_f)).mean()
        precision_loss = bce + self.fp_penalty_weight * fp_term

        return self.dice_weight * dice_loss + self.precision_weight * precision_loss


class DiceBoundaryLoss(nn.Module):
    """Soft-Dice + Kervadec-style boundary loss for direct HD95 optimization.

    Boundary loss (Kervadec et al. 2019): integrates predicted probability
    against the signed distance transform of GT. Correct (prob=1 inside GT,
    prob=0 outside) gives loss -> min; stray predictions far from GT surface
    incur proportionally larger penalty. Directly aligned with HD95/MSD.

    Implementation: we compute the SDT of `(1 - gt)` minus SDT of `gt` per
    sample. Positive inside GT, negative outside. Multiply by `probs` and
    average.
    """

    def __init__(self, dice_weight: float = 0.5, boundary_weight: float = 0.5,
                 smooth: float = 1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        from scipy.ndimage import distance_transform_edt

        target = mask.squeeze(1).long()
        target_f = target.float()
        probs = F.softmax(logits, dim=1)[:, 1]

        inter = (probs * target_f).sum(dim=(1, 2, 3))
        denom = probs.sum(dim=(1, 2, 3)) + target_f.sum(dim=(1, 2, 3))
        dice = (2.0 * inter + self.smooth) / (denom + self.smooth)
        dice_loss = 1.0 - dice.mean()

        # Boundary loss: SDT per sample. Signed distance: positive outside GT.
        with torch.no_grad():
            gt_np = target_f.detach().cpu().numpy().astype(np.uint8)
            sdt = np.zeros_like(gt_np, dtype=np.float32)
            for i in range(gt_np.shape[0]):
                g = gt_np[i]
                if g.any() and (1 - g).any():
                    # distance OUTSIDE GT (positive); penalize prob mass here
                    sdt[i] = distance_transform_edt(1 - g)
                else:
                    sdt[i] = 0.0
            sdt_t = torch.from_numpy(sdt).to(probs.device)

        boundary_term = (probs * sdt_t).mean()
        return self.dice_weight * dice_loss + self.boundary_weight * boundary_term


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def patch_dice(logits: torch.Tensor, mask: torch.Tensor) -> float:
    """Binary Dice over a batch of patches, from argmax predictions."""
    pred = logits.argmax(dim=1)
    target = mask.squeeze(1).long()
    inter = ((pred == 1) & (target == 1)).sum().item()
    denom = (pred == 1).sum().item() + (target == 1).sum().item()
    if denom == 0:
        return 1.0
    return 2.0 * inter / denom


@torch.no_grad()
def volume_dice_sliding_window(
    model: nn.Module,
    case_id: str,
    dataset: NnUNetRawDataset,
    device: torch.device,
    roi_size: Tuple[int, int, int] = (96, 96, 96),
) -> float:
    """Compute whole-volume Dice on one case via MONAI sliding-window.

    Used for validation — patch-level Dice during training is noisy and
    doesn't mirror the number nnU-Net reports. A proper comparison to
    nnU-Net 3D's 0.762 needs whole-volume Dice with the same kind of
    sliding-window inference nnU-Net does.
    """
    from monai.inferers import sliding_window_inference

    image, mask = dataset._load_case(case_id)  # (4, H, W, D), (H, W, D)
    x = torch.from_numpy(image).float().unsqueeze(0).to(device)

    logits = sliding_window_inference(
        inputs=x, roi_size=roi_size, sw_batch_size=1,
        predictor=model, overlap=0.5, mode="gaussian",
    )
    pred = logits.argmax(dim=1)[0].cpu().numpy()   # (H, W, D)
    target = (mask > 0).astype(pred.dtype)

    inter = ((pred == 1) & (target == 1)).sum()
    denom = (pred == 1).sum() + (target == 1).sum()
    if denom == 0:
        return 1.0
    return float(2.0 * inter / denom)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f) or {}
    entries = {m["name"]: m for m in cfg.get("models", [])}
    if "swin_unetr" not in entries:
        raise RuntimeError("models.yaml has no 'swin_unetr' entry")
    return entries["swin_unetr"]


def build_model(
    cfg: dict,
    device: torch.device,
    resume_state: Optional[dict],
    warm_start_from: Optional[Path] = None,
) -> nn.Module:
    """Build the SwinUNETR backbone, optionally loading weights from a
    resume state dict (from LATEST_PATH) or from the BTCV warm-start.

    If `warm_start_from` is provided, loads model weights from that checkpoint
    WITHOUT optimizer/scaler state. Useful for starting a variant (e.g.
    precision-focused) from an already-trained model with a fresh optimizer.
    """
    model = build_swin_unetr_from_dict(cfg).to(device)

    if resume_state is not None:
        print("  Resuming: loading model weights from latest checkpoint")
        model.load_state_dict(resume_state["model_state_dict"], strict=False)
        return model

    if warm_start_from is not None and warm_start_from.exists():
        ck = torch.load(warm_start_from, map_location=device, weights_only=False)
        state = ck.get("model_state_dict", ck)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"  Warm-started from {warm_start_from} "
              f"({len(missing)} missing, {len(unexpected)} unexpected keys)")
        return model

    pretrained = cfg.get("pretrained")
    if pretrained:
        pretrained_path = PROJECT / pretrained
        if pretrained_path.exists():
            missing, _ = load_pretrained_backbone(model, pretrained_path)
            print(f"  Warm-started from {pretrained_path} "
                  f"({len(missing)} mismatched keys re-init'd from scratch)")
        else:
            print(f"  NOTE: pretrained weights not found at {pretrained_path} — "
                  f"training from scratch")
    return model


def save_latest_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    phase: int,
    epoch: int,
    best_val_dice: float,
) -> None:
    """Persist full training state for restartability."""
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "phase": int(phase),
        "epoch": int(epoch),
        "best_val_dice": float(best_val_dice),
    }, path)


def save_best_checkpoint(
    path: Path, model: nn.Module, phase: int, epoch: int, val_dice: float,
) -> None:
    """Small, inference-only checkpoint of the best model."""
    torch.save({
        "model_state_dict": model.state_dict(),
        "phase": int(phase),
        "epoch": int(epoch),
        "val_dice": float(val_dice),
    }, path)


def run_phase(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    val_dataset: NnUNetRawDataset,
    val_case_ids,
    epochs: int,
    phase: int,
    device: torch.device,
    scaler: GradScaler,
    best_val_dice: float,
    val_every: int,
    val_subset: int,
    start_epoch: int = 1,
    early_stop_patience: int = 0,
    early_stop_min_delta: float = 0.0,
    loss_kind: str = "dicece",
    model_out_path: Path = MODEL_OUT_PATH,
    latest_path: Path = LATEST_PATH,
) -> Tuple[float, int]:
    """Run one phase. Returns (best val Dice, last epoch completed).

    If `early_stop_patience > 0`, training stops after that many consecutive
    validation passes with no improvement ≥ `early_stop_min_delta`.
    """
    if loss_kind == "precision":
        loss_fn = DicePrecisionLoss().to(device)
    elif loss_kind == "boundary":
        loss_fn = DiceBoundaryLoss().to(device)
    else:
        loss_fn = DiceCELoss().to(device)
    log_suffix = f"_{loss_kind}" if loss_kind != "dicece" else ""
    log_path = MODEL_OUT_DIR / f"swin_unetr_phase{phase}{log_suffix}_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    no_improve_count = 0

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        running = 0.0
        running_patch_dice = 0.0
        n_batches = 0
        t0 = time.time()
        for batch in train_loader:
            images, mask = batch[0], batch[1]
            images = images.to(device, non_blocking=True).float()
            mask = mask.to(device, non_blocking=True).float()

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=device.type == "cuda"):
                logits = model(images)
                loss = loss_fn(logits, mask)

            if device.type == "cuda":
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            running += float(loss.item())
            running_patch_dice += patch_dice(logits.detach(), mask)
            n_batches += 1

        train_loss = running / max(n_batches, 1)
        train_patch_dice = running_patch_dice / max(n_batches, 1)

        # Whole-volume validation — expensive, so we only do it every
        # `val_every` epochs and on a random subset of `val_subset` cases.
        do_val = (epoch % val_every == 0) or (epoch == epochs)
        val_dice = float("nan")
        if do_val:
            model.eval()
            subset_ids = val_case_ids[:val_subset]
            dices = []
            for cid in subset_ids:
                try:
                    d = volume_dice_sliding_window(
                        model, cid, val_dataset, device)
                    dices.append(d)
                except Exception as e:
                    print(f"    [val warn] {cid}: {e}")
            val_dice = sum(dices) / max(len(dices), 1)

        dt = time.time() - t0
        print(f"  [phase {phase}] epoch {epoch:4d}/{epochs:4d}  "
              f"train_loss={train_loss:.4f}  train_patch_dice={train_patch_dice:.4f}  "
              + (f"val_dice={val_dice:.4f}  " if do_val else "")
              + f"({dt:.1f}s)")

        # Append jsonl row for post-hoc analysis
        with open(log_path, "a") as lf:
            lf.write(json.dumps({
                "phase": phase, "epoch": epoch,
                "train_loss": train_loss,
                "train_patch_dice": train_patch_dice,
                "val_dice": None if val_dice != val_dice else val_dice,
                "elapsed_s": dt,
            }) + "\n")

        if do_val and val_dice == val_dice:
            if val_dice > best_val_dice + early_stop_min_delta:
                best_val_dice = val_dice
                no_improve_count = 0
                MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)
                save_best_checkpoint(
                    model_out_path, model, phase, epoch, val_dice)
                print(f"    new best val_dice={val_dice:.4f}, saved to "
                      f"{model_out_path}")
            else:
                no_improve_count += 1
                if early_stop_patience > 0:
                    print(f"    no improvement ({no_improve_count}/"
                          f"{early_stop_patience} val passes without "
                          f"+{early_stop_min_delta:.3f} over best={best_val_dice:.4f})")

        # Save the full training state every epoch so --resume can pick
        # up with correct optimizer momentum and the right epoch counter.
        # Writing ~250MB every epoch is cheap compared to ~25 min of
        # compute, and protects us from mid-run interruptions.
        save_latest_checkpoint(
            latest_path, model, optimizer, scaler, phase, epoch, best_val_dice)

        if (early_stop_patience > 0
                and no_improve_count >= early_stop_patience):
            print(f"    EARLY STOP at epoch {epoch} -- "
                  f"{no_improve_count} val passes without improvement "
                  f">={early_stop_min_delta:.3f}. Best val_dice={best_val_dice:.4f}")
            return best_val_dice, epoch

    return best_val_dice, epoch


def main():
    parser = argparse.ArgumentParser(
        description="Two-phase full-dataset fine-tune of SwinUNETR")
    parser.add_argument("--phase", type=int, choices=[0, 1, 2], default=0,
                        help="0 = both phases; 1 or 2 = single phase only")
    parser.add_argument("--epochs-phase1", type=int, default=20)
    parser.add_argument("--epochs-phase2", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--samples-per-case", type=int, default=2,
                        help="Random patches sampled per case per epoch")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--patch-size", type=int, nargs=3, default=[96, 96, 96])
    parser.add_argument("--val-every", type=int, default=5,
                        help="Run whole-volume validation every N epochs")
    parser.add_argument("--val-subset", type=int, default=40,
                        help="Number of val cases to evaluate per validation pass (out of 584)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from model/swin_unetr_latest.pth — restores "
                             "model weights, optimizer state, scaler state, phase, "
                             "and epoch counter.")
    parser.add_argument("--early-stop-patience", type=int, default=0,
                        help="Stop if val_dice doesn't improve for N consecutive "
                             "validation passes. 0 disables early stopping.")
    parser.add_argument("--early-stop-min-delta", type=float, default=0.001,
                        help="Minimum val_dice improvement (vs current best) to "
                             "count as progress. Below this is considered noise.")
    parser.add_argument("--loss", choices=["dicece", "precision", "boundary"],
                        default="dicece",
                        help="Loss function: 'dicece' = Dice+CE (default), "
                             "'precision' = Dice+PrecisionLoss (FP-suppressing), "
                             "'boundary' = Dice+BoundaryLoss (HD95-aware).")
    parser.add_argument("--output-suffix", type=str, default="",
                        help="Suffix for checkpoint filenames, e.g. 'precision' "
                             "-> swin_unetr_brainmets_precision.pth. Lets us train "
                             "multiple variants without clobbering the main model.")
    parser.add_argument("--warm-start-from", type=str, default=None,
                        help="Path to an existing checkpoint (e.g. "
                             "model/swin_unetr_brainmets.pth) to warm-start weights "
                             "from WITHOUT loading its optimizer/scaler state.")
    args = parser.parse_args()

    # SwinUNETR needs every spatial dim divisible by 2**5 = 32.
    for axis, s in zip("HWD", args.patch_size):
        if s % 32 != 0:
            raise SystemExit(
                f"--patch-size axis {axis}={s} must be divisible by 32 "
                f"for SwinUNETR. Try 96 or 128.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    cfg = load_config()

    # Variant-aware checkpoint paths. Empty suffix = legacy defaults.
    suffix = f"_{args.output_suffix}" if args.output_suffix else ""
    model_out_path = MODEL_OUT_DIR / f"swin_unetr_brainmets{suffix}.pth"
    latest_path = MODEL_OUT_DIR / f"swin_unetr_latest{suffix}.pth"
    state_path = MODEL_OUT_DIR / f"swin_unetr_training_state{suffix}.json"
    print(f"  Loss: {args.loss}   Best-ckpt: {model_out_path}")
    print(f"  Latest-ckpt: {latest_path}")

    # ----- Split -----
    train_ids, val_ids = load_nnunet_split(NNUNET_SPLITS, fold=args.fold)
    print(f"  nnU-Net fold {args.fold}: {len(train_ids)} train / {len(val_ids)} val")

    # ----- Datasets -----
    train_ds = NnUNetRawDataset(
        dataset_root=NNUNET_RAW,
        case_ids=train_ids,
        patch_size=tuple(args.patch_size),
        samples_per_case=args.samples_per_case,
        foreground_ratio=0.67,
        normalize=True,
    )
    # Separate dataset for validation so `_load_case` reads unnormalised
    # file paths consistently. (We still z-score inside `_load_case`.)
    val_ds = NnUNetRawDataset(
        dataset_root=NNUNET_RAW,
        case_ids=val_ids,
        patch_size=tuple(args.patch_size),
        samples_per_case=1,
        foreground_ratio=0.67,
        normalize=True,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        persistent_workers=args.num_workers > 0,
    )

    # ---------- Resume bookkeeping ----------
    #
    # If --resume is set and swin_unetr_latest.pth exists, we read:
    #   resume_state["phase"] = 1 or 2
    #   resume_state["epoch"] = last completed epoch inside that phase
    #   resume_state["best_val_dice"] = best so far
    #   resume_state["optimizer_state_dict"] = AdamW state (phase-specific)
    #   resume_state["scaler_state_dict"]     = AMP scaler state
    #
    # We then:
    #   - If resumed-phase == 1: continue phase 1 from epoch+1, skip phase 1's
    #     optimizer reinit, then proceed to phase 2 fresh.
    #   - If resumed-phase == 2: skip phase 1 entirely, continue phase 2 from
    #     epoch+1 with the saved optimizer state.
    # The saved optimizer state from phase 1 is NOT compatible with the
    # phase-2 optimizer (different param groups), so we deliberately
    # discard it on the phase transition.
    resume_state: Optional[dict] = None
    if args.resume and latest_path.exists():
        print(f"  Loading resume state from {latest_path}")
        resume_state = torch.load(
            latest_path, map_location=device, weights_only=False)
        print(f"    resumed phase: {resume_state['phase']}  "
              f"epoch: {resume_state['epoch']}  "
              f"best_val_dice: {resume_state.get('best_val_dice', 0.0):.4f}")

    warm_start_path = Path(args.warm_start_from) if args.warm_start_from else None
    model = build_model(cfg, device, resume_state, warm_start_from=warm_start_path)
    scaler = GradScaler(device="cuda", enabled=device.type == "cuda")
    if resume_state is not None and "scaler_state_dict" in resume_state:
        try:
            scaler.load_state_dict(resume_state["scaler_state_dict"])
        except Exception as e:
            print(f"    (could not load scaler state: {e}; starting fresh scaler)")

    best = float(resume_state.get("best_val_dice", 0.0)) if resume_state else 0.0
    resumed_phase = int(resume_state["phase"]) if resume_state else None
    resumed_epoch = int(resume_state["epoch"]) if resume_state else 0

    finetune_cfg = cfg.get("finetune", {}) or {}
    phase1_cfg = finetune_cfg.get("phase1", {}) or {}
    phase2_cfg = finetune_cfg.get("phase2", {}) or {}
    ratio = float(phase2_cfg.get("lr_encoder", 1e-5)) / max(
        float(phase2_cfg.get("lr_decoder", 1e-4)), 1e-12)

    # --------- Phase 1 ----------
    #
    # Skip entirely if:
    #   (a) --phase 2 was requested explicitly, OR
    #   (b) we resumed from a mid-phase-2 checkpoint.
    skip_phase1 = (args.phase == 2) or (resumed_phase == 2)

    if args.phase in (0, 1) and not skip_phase1:
        p1_start_epoch = 1
        if resumed_phase == 1:
            p1_start_epoch = resumed_epoch + 1
        print(f"\n=== Phase 1 — encoder frozen, {args.epochs_phase1} epochs "
              f"(starting at epoch {p1_start_epoch}) ===")
        opt1 = make_two_phase_optimizer(
            model, phase=1,
            lr_decoder=float(phase1_cfg.get("lr", 1e-4)),
        )
        # Only load optimizer state if we're actually resuming phase 1.
        if resumed_phase == 1 and resume_state is not None:
            try:
                opt1.load_state_dict(resume_state["optimizer_state_dict"])
                print("    loaded phase-1 optimizer state from latest checkpoint")
            except Exception as e:
                print(f"    (optimizer state reload failed: {e}; starting fresh)")

        if p1_start_epoch <= args.epochs_phase1:
            best, _ = run_phase(
                model, opt1, train_loader, val_ds, val_ids,
                args.epochs_phase1, 1, device, scaler, best,
                val_every=args.val_every, val_subset=args.val_subset,
                start_epoch=p1_start_epoch,
                early_stop_patience=args.early_stop_patience,
                early_stop_min_delta=args.early_stop_min_delta,
                loss_kind=args.loss,
                model_out_path=model_out_path,
                latest_path=latest_path,
            )
        else:
            print(f"    phase 1 already finished ({p1_start_epoch - 1} / "
                  f"{args.epochs_phase1}), skipping")
        # Clear the phase-1 resume state — phase 2 uses a new optimizer
        # whose param groups don't match phase 1's, so reloading that
        # state would error.
        resume_state = None
        resumed_phase = None
        resumed_epoch = 0

    # --------- Phase 2 ----------
    if args.phase in (0, 2):
        p2_start_epoch = 1
        if resumed_phase == 2:
            p2_start_epoch = resumed_epoch + 1
        print(f"\n=== Phase 2 — full fine-tune, {args.epochs_phase2} epochs "
              f"(starting at epoch {p2_start_epoch}) ===")
        opt2 = make_two_phase_optimizer(
            model, phase=2,
            lr_decoder=float(phase2_cfg.get("lr_decoder", 1e-4)),
            lr_encoder_ratio=ratio,
        )
        if resumed_phase == 2 and resume_state is not None:
            try:
                opt2.load_state_dict(resume_state["optimizer_state_dict"])
                print("    loaded phase-2 optimizer state from latest checkpoint")
            except Exception as e:
                print(f"    (optimizer state reload failed: {e}; starting fresh)")

        if p2_start_epoch <= args.epochs_phase2:
            best, _ = run_phase(
                model, opt2, train_loader, val_ds, val_ids,
                args.epochs_phase2, 2, device, scaler, best,
                val_every=args.val_every, val_subset=args.val_subset,
                start_epoch=p2_start_epoch,
                early_stop_patience=args.early_stop_patience,
                early_stop_min_delta=args.early_stop_min_delta,
                loss_kind=args.loss,
                model_out_path=model_out_path,
                latest_path=latest_path,
            )
        else:
            print(f"    phase 2 already finished ({p2_start_epoch - 1} / "
                  f"{args.epochs_phase2}). Pass a larger --epochs-phase2 "
                  f"to continue training.")

    print(f"\nDone. Best val Dice (whole-volume, sliding window): {best:.4f}")
    print(f"Checkpoint: {model_out_path}")
    # Write checkpoint path as project-relative if possible, otherwise absolute
    # (relative_to() raises for paths outside the project root — happens in
    # tests that redirect MODEL_OUT_DIR to tmp_path).
    try:
        ckpt_str = str(model_out_path.relative_to(PROJECT))
    except ValueError:
        ckpt_str = str(model_out_path)
    with open(state_path, "w") as f:
        json.dump({
            "best_val_dice": best,
            "checkpoint": ckpt_str,
            "fold": args.fold,
            "train_cases": len(train_ids),
            "val_cases": len(val_ids),
        }, f, indent=2)


if __name__ == "__main__":
    main()
