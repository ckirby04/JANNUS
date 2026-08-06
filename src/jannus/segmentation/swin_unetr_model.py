"""
SwinUNETR wrapper for brain metastasis segmentation.

Thin shim around `monai.networks.nets.SwinUNETR` that:

  1. Configures the network for our 4-channel MRI input
     (T1, T1ce/t1_gd, T2, FLAIR) and 2-class (background / metastasis)
     output at img_size (96, 96, 96) with feature_size=48.
  2. Exposes helpers to load the MONAI-model-zoo BTCV checkpoint
     (swin_unetr_btcv_segmentation.pt) as the warm-start backbone, while
     gracefully tolerating our 4-channel stem and 2-channel head mismatch.
  3. Provides `freeze_encoder()` / `unfreeze_encoder()` and
     `make_two_phase_optimizer()` for the required two-phase fine-tune
     schedule:
         Phase 1: encoder frozen, decoder only.
         Phase 2: full fine-tune with encoder LR = decoder LR / 10.

This module is intentionally stateless — training loops live in
`scripts/training/` and consume these helpers.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

# MONAI is already a hard dependency (requirements.txt / pyproject.toml).
from monai.networks.nets import SwinUNETR


@dataclass
class SwinUNETRConfig:
    """Configuration for the SwinUNETR ensemble member.

    Mirrors the `models[swin_unetr]` entry in `configs/models.yaml`; anything
    that would otherwise be hardcoded here lives in the ensemble config.
    """

    in_channels: int = 4
    out_channels: int = 2
    img_size: tuple[int, int, int] = (96, 96, 96)
    feature_size: int = 48
    use_checkpoint: bool = True
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    dropout_path_rate: float = 0.0


def build_swin_unetr(cfg: SwinUNETRConfig | None = None) -> SwinUNETR:
    """Instantiate SwinUNETR from a config dict or SwinUNETRConfig.

    Note: MONAI >=1.4 removed the `img_size` argument from SwinUNETR's
    constructor — the model now infers spatial dims from the input
    tensor at forward time. `cfg.img_size` is still honoured at the
    ensemble layer for sliding-window inference (see SwinUNETRAdapter).
    """
    cfg = cfg or SwinUNETRConfig()
    return SwinUNETR(
        in_channels=cfg.in_channels,
        out_channels=cfg.out_channels,
        feature_size=cfg.feature_size,
        use_checkpoint=cfg.use_checkpoint,
        drop_rate=cfg.drop_rate,
        attn_drop_rate=cfg.attn_drop_rate,
        dropout_path_rate=cfg.dropout_path_rate,
    )


def build_swin_unetr_from_dict(d: dict) -> SwinUNETR:
    """Factory used by the model registry / ensemble.

    Accepts the dict form of the `swin_unetr` entry in
    `configs/models.yaml`.
    """
    return build_swin_unetr(SwinUNETRConfig(
        in_channels=int(d.get("in_channels", 4)),
        out_channels=int(d.get("out_channels", 2)),
        img_size=tuple(d.get("img_size", [96, 96, 96])),
        feature_size=int(d.get("feature_size", 48)),
        use_checkpoint=bool(d.get("use_checkpoint", True)),
    ))


# ---------------------------------------------------------------------------
# Pretrained-weight loading (MONAI model zoo BTCV checkpoint)
# ---------------------------------------------------------------------------

# TODO(manual): The MONAI BTCV pretrained weights file
# (swin_unetr_btcv_segmentation.pt) cannot be downloaded reliably in every
# environment — place it manually at `model/pretrained/swin_unetr_btcv.pt`
# before training. See configs/models.yaml -> models.swin_unetr.pretrained.

def load_pretrained_backbone(
    model: SwinUNETR,
    checkpoint_path: str | Path,
    strict: bool = False,
) -> tuple[list[str], list[str]]:
    """Load MONAI BTCV SwinUNETR weights into our 4-channel/2-class model.

    The BTCV checkpoint was trained with 1 input channel and 14 output
    classes; we re-initialise the first conv and final head from scratch and
    warm-start everything in between. Returns the (missing_keys,
    unexpected_keys) list from `load_state_dict` for logging.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        # Intentional: smoke tests and fresh clones should not crash here.
        return ([], [f"<checkpoint not found: {checkpoint_path}>"])

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    # Drop keys whose shapes conflict with the 4-channel stem or 2-class head;
    # those layers will train from scratch during phase 1.
    model_state = model.state_dict()
    filtered = {
        k: v for k, v in state.items()
        if k in model_state and v.shape == model_state[k].shape
    }
    missing, unexpected = model.load_state_dict(filtered, strict=strict)
    return list(missing), list(unexpected)


# ---------------------------------------------------------------------------
# Two-phase fine-tuning helpers
# ---------------------------------------------------------------------------

def _encoder_parameters(model: SwinUNETR) -> Iterable[nn.Parameter]:
    """Parameters we consider part of the SwinUNETR encoder.

    MONAI's SwinUNETR exposes `swinViT` (the Swin transformer trunk) plus a
    series of `encoder{1..10}` side-blocks that feed the UNet decoder; we
    treat the trunk and the encoder side-blocks as "encoder" and everything
    else (`decoder*`, `out`) as "decoder".
    """
    for name, p in model.named_parameters():
        if name.startswith("swinViT") or name.startswith("encoder"):
            yield p


def _decoder_parameters(model: SwinUNETR) -> Iterable[nn.Parameter]:
    for name, p in model.named_parameters():
        if not (name.startswith("swinViT") or name.startswith("encoder")):
            yield p


def freeze_encoder(model: SwinUNETR) -> None:
    """Phase 1: disable gradients on the transformer trunk + encoder blocks."""
    for p in _encoder_parameters(model):
        p.requires_grad = False


def unfreeze_encoder(model: SwinUNETR) -> None:
    """Phase 2: re-enable encoder gradients."""
    for p in _encoder_parameters(model):
        p.requires_grad = True


def make_two_phase_optimizer(
    model: SwinUNETR,
    phase: int,
    lr_decoder: float = 1e-4,
    lr_encoder_ratio: float = 0.1,
    weight_decay: float = 1e-5,
) -> torch.optim.Optimizer:
    """Build the correct optimizer for phase 1 or phase 2.

    Phase 1: AdamW over decoder parameters only (encoder is frozen).
    Phase 2: AdamW with two param groups — decoder at `lr_decoder`, encoder
             at `lr_decoder * lr_encoder_ratio` (default 1/10).
    """
    if phase == 1:
        freeze_encoder(model)
        return torch.optim.AdamW(
            list(_decoder_parameters(model)),
            lr=lr_decoder,
            weight_decay=weight_decay,
        )
    if phase == 2:
        unfreeze_encoder(model)
        return torch.optim.AdamW(
            [
                {"params": list(_decoder_parameters(model)), "lr": lr_decoder},
                {"params": list(_encoder_parameters(model)),
                 "lr": lr_decoder * lr_encoder_ratio},
            ],
            weight_decay=weight_decay,
        )
    raise ValueError(f"Unknown fine-tune phase: {phase} (expected 1 or 2)")
