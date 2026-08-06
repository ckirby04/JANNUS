"""
Stacking classifier for brain metastasis segmentation.

Production model (loaded by `src/segmentation/pipeline.py` and
`scripts/inference/run_inference.py`):

    architecture = StackingClassifierV2  (~1.36M params, SE attention,
                                          multi-scale branch, residual)
    base models  = [nnU-Net 3D, nnU-Net 2D,
                    patch_8, patch_12, patch_24, patch_36 (LightweightUNet3D),
                    SwinUNETR 150ep+]
    in_channels  = 9   (7 predictions + variance + range)
    checkpoint   = model/stacking_classifier_production.pth
    val Dice     = 0.7858 on 84-case held-out

Legacy (kept for back-compat): StackingClassifier (~25K params), 3 bases,
5-channel input, checkpoint model/stacking_v5_classifier.pth — Variant 1
in docs/stacking_architectures_explored.md (Dice 0.7744). Selected by
passing in_channels=5 to load_stacking_model().
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import label as ndimage_label
from scipy.ndimage import zoom
from torch.amp import autocast

from ..core import paths as _paths

# =============================================================================
# CONFIG
# =============================================================================

# ----- Production 7-model ensemble (StackingClassifierV2) -----
# Order is canonical and must match the order the pipeline feeds base
# predictions to `build_stacking_features_from_preds`.
STACKING_MODEL_NAMES_V2 = [
    'nnunet_3d', 'nnunet_2d',
    'patch_8', 'patch_12', 'patch_24', 'patch_36',
    'swin_unetr',
]
# 7 predictions + variance + (max - min) range = 9 feature channels.
STACKING_IN_CHANNELS_V2 = 9

# ----- Legacy 3-model ensemble (StackingClassifier, v2 hybrid) -----
# Retained because StackingClassifier is the smaller architecture and the
# 5-channel checkpoint at model/stacking_v5_classifier.pth is still on
# disk. Variant 1 in docs/stacking_architectures_explored.md (Dice 0.7744).
STACKING_MODEL_NAMES_LEGACY = [
    'nnunet_3d', 'nnunet_2d', 'swin_unetr',
]
STACKING_IN_CHANNELS_LEGACY = 5

# Default to the production constants. Legacy callers that still want the
# 3-model layout should import the *_LEGACY constants explicitly.
STACKING_MODEL_NAMES = STACKING_MODEL_NAMES_V2
STACKING_IN_CHANNELS = STACKING_IN_CHANNELS_V2

STACKING_THRESHOLD = 0.5
STACKING_PATCH_SIZE = 32
STACKING_OVERLAP = 0.5

DISPLAY_NAMES = {
    'nnunet_3d': 'nnU-Net 3D',
    'nnunet_2d': 'nnU-Net 2D',
    'patch_8': 'LightweightUNet3D patch_8',
    'patch_12': 'LightweightUNet3D patch_12',
    'patch_24': 'LightweightUNet3D patch_24',
    'patch_36': 'LightweightUNet3D patch_36',
    'swin_unetr': 'SwinUNETR',
}


# =============================================================================
# STACKING CLASSIFIER
# =============================================================================

class StackingClassifier(nn.Module):
    """
    3D CNN meta-learner with residual connections.
    Input: 8 channels (6 model predictions + variance + max-min range)
    Output: 1 channel (final segmentation logits)
    ~25K trainable parameters
    """
    def __init__(self, in_channels=8, mid_channels=32):
        super().__init__()
        self.in_channels = in_channels
        self.entry = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.block1 = nn.Sequential(
            nn.Conv3d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.1),
            nn.Conv3d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(mid_channels),
        )
        self.block2 = nn.Sequential(
            nn.Conv3d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.1),
            nn.Conv3d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(mid_channels),
        )
        self.head = nn.Conv3d(mid_channels, 1, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.entry(x)
        x = self.relu(x + self.block1(x))
        x = self.relu(x + self.block2(x))
        return self.head(x)


class _SEBlock3D(nn.Module):
    """Squeeze-Excitation channel attention for 3D feature maps."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        B, C, *_ = x.shape
        w = self.pool(x).view(B, C)
        w = self.fc(w).view(B, C, 1, 1, 1)
        return x * w


class _ResBlockSE3D(nn.Module):
    """Two 3x3x3 convs + BN + ReLU + SE attention, with residual add."""

    def __init__(self, channels: int, dropout: float = 0.1,
                 se_reduction: int = 8):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(channels)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(channels)
        self.dropout = nn.Dropout3d(dropout)
        self.se = _SEBlock3D(channels, reduction=se_reduction)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return self.relu(out + residual)


class StackingClassifierV2(nn.Module):
    """Higher-capacity meta-learner: input bottleneck + wider+deeper trunk +
    multi-scale context branch + SE channel attention.

    Changes vs StackingClassifier:
      1. 1x1x1 input bottleneck that learns linear combinations of base preds
         before any spatial processing.
      2. Wider (mid_channels default 64 vs 32) and deeper (n_blocks default 4
         vs 2) full-resolution trunk.
      3. Multi-scale branch: AvgPool3d(2) -> n_blocks_lowres residual blocks
         -> trilinear upsample, concatenated with the full-res output. Gives
         the stacker spatial context beyond the 32^3 patch's local receptive
         field (helps reject far-away FPs that hurt HD95/MSD).
      4. SE channel attention in every residual block, so the model learns
         which base to trust where.

    Parameter count at defaults (in_channels=9, mid=64, n_blocks=4,
    n_blocks_lowres=2): ~700k. Still small by segmentation standards.
    """

    def __init__(self, in_channels: int = 9, mid_channels: int = 64,
                 n_blocks: int = 4, n_blocks_lowres: int = 2,
                 dropout: float = 0.1, se_reduction: int = 8):
        super().__init__()
        self.in_channels = in_channels

        # (1) Input bottleneck: learn channel mixing before spatial conv
        self.input_bottleneck = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True),
        )

        # (2) Entry conv + wider+deeper residual trunk
        self.entry = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.full_res_blocks = nn.Sequential(*[
            _ResBlockSE3D(mid_channels, dropout=dropout, se_reduction=se_reduction)
            for _ in range(n_blocks)
        ])

        # (3) Multi-scale branch: downsample -> process -> upsample
        self.downsample = nn.AvgPool3d(kernel_size=2, stride=2)
        self.lowres_blocks = nn.Sequential(*[
            _ResBlockSE3D(mid_channels, dropout=dropout, se_reduction=se_reduction)
            for _ in range(n_blocks_lowres)
        ])
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

        # Fusion of full-res and upsampled low-res
        self.fuse = nn.Sequential(
            nn.Conv3d(mid_channels * 2, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
        )

        # Head
        self.head = nn.Conv3d(mid_channels, 1, kernel_size=1)

    def forward(self, x):
        x = self.input_bottleneck(x)
        f_full = self.full_res_blocks(self.entry(x))
        f_low = self.upsample(self.lowres_blocks(self.downsample(f_full)))
        fused = self.fuse(torch.cat([f_full, f_low], dim=1))
        return self.head(fused)


# =============================================================================
# POST-PROCESSING
# =============================================================================

def postprocess_prediction(binary_mask, min_size=0):
    """Remove connected components smaller than min_size voxels.

    Default changed from 20 → 0 on 2026-04-19: a per-bucket sweep showed
    min_size=20 was discarding real tiny metastases (Dice 0.3871 vs 0.4500
    with no postprocessing on the <100-voxel bucket).
    """
    labeled, n_components = ndimage_label(binary_mask)
    if n_components == 0:
        return binary_mask
    result = np.zeros_like(binary_mask)
    for i in range(1, n_components + 1):
        component = (labeled == i)
        if component.sum() >= min_size:
            result[component] = 1
    return result


# =============================================================================
# SLIDING WINDOW INFERENCE
# =============================================================================

def sliding_window_inference(model, volume, patch_size, device, overlap=0.25):
    """
    Run inference on full volume using sliding window with overlap.
    Returns probability map (not thresholded).
    """
    model.eval()
    C, H, W, D = volume.shape
    p = patch_size
    stride = max(int(p * (1 - overlap)), 1)

    # Dynamic batch size based on patch size
    if p <= 8:
        batch_size = 512
    elif p <= 12:
        batch_size = 256
    elif p <= 24:
        batch_size = 64
    else:
        batch_size = 32

    # Pad volume if needed
    pad_h = (p - H % p) % p if H % stride != 0 else 0
    pad_w = (p - W % p) % p if W % stride != 0 else 0
    pad_d = (p - D % p) % p if D % stride != 0 else 0

    orig_H, orig_W, orig_D = H, W, D

    if pad_h > 0 or pad_w > 0 or pad_d > 0:
        volume = np.pad(volume, ((0, 0), (0, pad_h), (0, pad_w), (0, pad_d)), mode='constant')
        C, H, W, D = volume.shape

    output = np.zeros((H, W, D), dtype=np.float32)
    count = np.zeros((H, W, D), dtype=np.float32)

    coords = []
    for h in range(0, H - p + 1, stride):
        for w in range(0, W - p + 1, stride):
            for d in range(0, D - p + 1, stride):
                coords.append((h, w, d))

    with torch.no_grad():
        for i in range(0, len(coords), batch_size):
            batch_coords = coords[i:i + batch_size]
            patches = []
            for h, w, d in batch_coords:
                patches.append(volume[:, h:h+p, w:w+p, d:d+p])

            batch = torch.from_numpy(np.stack(patches)).float().to(device)
            # Use cuda autocast if available, otherwise run without
            if device.type == 'cuda':
                with autocast('cuda'):
                    preds = torch.sigmoid(model(batch)).cpu().numpy()
            else:
                preds = torch.sigmoid(model(batch)).cpu().numpy()

            for j, (h, w, d) in enumerate(batch_coords):
                output[h:h+p, w:w+p, d:d+p] += preds[j, 0]
                count[h:h+p, w:w+p, d:d+p] += 1

    output = output / np.maximum(count, 1)
    return output[:orig_H, :orig_W, :orig_D]


# =============================================================================
# FEATURE BUILDING
# =============================================================================

def build_stacking_features(cache_file, model_names=None):
    """
    Build full-volume stacking features from cached predictions.

    Returns:
        features: (N+2, H, W, D) array — N model predictions + variance + range
        preds: (N, H, W, D) array — individual model predictions
        mask: (H, W, D) array — ground truth mask
    """
    if model_names is None:
        model_names = STACKING_MODEL_NAMES

    data = np.load(cache_file)
    mask = data['mask']

    preds = []
    for name in model_names:
        preds.append(data[name])
    preds = np.stack(preds, axis=0)

    variance = preds.var(axis=0, keepdims=True)
    range_map = preds.max(axis=0, keepdims=True) - preds.min(axis=0, keepdims=True)

    features = np.concatenate([preds, variance, range_map], axis=0)
    return features, preds, mask


def build_stacking_features_from_preds(preds: np.ndarray) -> np.ndarray:
    """Build the (N+2, H, W, D) stacking feature tensor from an in-memory
    stack of N base-model predictions (shape (N, H, W, D)).

    Used by the live inference pipeline (`src/segmentation/pipeline.py`),
    which never touches the on-disk stacking cache format.
    """
    if preds.ndim != 4:
        raise ValueError(
            f"preds must have shape (N, H, W, D); got {preds.shape}"
        )
    variance = preds.var(axis=0, keepdims=True)
    range_map = preds.max(axis=0, keepdims=True) - preds.min(axis=0, keepdims=True)
    return np.concatenate([preds, variance, range_map], axis=0)


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_stacking_model(
    model_dir=None,
    device=None,
    version: str = "v5",
    in_channels: int | None = None,
    allow_untrained: bool = False,
    checkpoint_name: str | None = None,
):
    """
    Load the stacking classifier from checkpoint.

    The architecture (StackingClassifier vs StackingClassifierV2) and
    default checkpoint are selected by `in_channels`:

        in_channels == 9 -> StackingClassifierV2  (production champion,
                            7-base, model/stacking_classifier_production.pth)
        in_channels == 5 -> StackingClassifier    (legacy v2 hybrid 3-base,
                            model/stacking_v5_classifier.pth)

    Args:
        model_dir: Path to model directory (default: project model/ dir)
        device: torch device (default: cuda if available)
        version: kept for back-compat; ignored — architecture is chosen by
            in_channels
        in_channels: 5 or 9 (default: STACKING_IN_CHANNELS = 9)
        allow_untrained: if True and no checkpoint is on disk, return a
            randomly-initialised classifier instead of None. Used by the
            smoke test and by fresh clones.
        checkpoint_name: explicit checkpoint filename inside model_dir.
            If None, derived from in_channels.
    """
    if model_dir is None:
        model_dir = _paths.model_dir()
    else:
        model_dir = Path(model_dir)

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if in_channels is None:
        in_channels = STACKING_IN_CHANNELS

    if in_channels == STACKING_IN_CHANNELS_V2:
        cls = StackingClassifierV2
        default_name = "stacking_classifier_production.pth"
    elif in_channels == STACKING_IN_CHANNELS_LEGACY:
        cls = StackingClassifier
        default_name = "stacking_v5_classifier.pth"
    else:
        raise ValueError(
            f"Unsupported stacker in_channels={in_channels}; "
            f"expected {STACKING_IN_CHANNELS_LEGACY} (legacy) or "
            f"{STACKING_IN_CHANNELS_V2} (production)."
        )

    checkpoint_path = model_dir / (checkpoint_name or default_name)

    def _fresh():
        if not allow_untrained:
            return None
        m = cls(in_channels=in_channels).to(device)
        m.eval()
        return m

    if not checkpoint_path.exists():
        if allow_untrained:
            return _fresh()
        raise FileNotFoundError(
            f"load_stacking_model: stacker checkpoint not found at "
            f"{checkpoint_path}. Pass allow_untrained=True only for "
            f"smoke tests / fresh clones; production callers must "
            f"have the stacker checkpoint on disk."
        )

    model = cls(in_channels=in_channels).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = checkpoint.get('model_state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
    try:
        model.load_state_dict(state)
    except RuntimeError:
        # Shape mismatch: a checkpoint trained for a different ensemble
        # size is sitting under the expected path. Fall back to a fresh
        # classifier rather than crashing.
        return _fresh()
    model.eval()
    return model


# =============================================================================
# INFERENCE PIPELINE
# =============================================================================

def run_stacking_inference(cache_path, model, device, target_size=None,
                           model_names=None, patch_size=None, overlap=None):
    """
    Full stacking inference pipeline: load cache → build features → sliding window → upsample.

    Args:
        cache_path: Path to stacking cache npz file
        model: StackingClassifier model
        device: torch device
        target_size: Optional (H, W, D) tuple to upsample output to (e.g., 256^3)
        model_names: List of model names (default: STACKING_MODEL_NAMES)
        patch_size: Stacking sliding window patch size (default: STACKING_PATCH_SIZE)
        overlap: Sliding window overlap (default: STACKING_OVERLAP)

    Returns:
        dict with:
            'fused': probability map (target_size or 128^3)
            'individual': {model_name: prob_map} for each base model
            'agreement': agreement map (how many models predict positive at their thresholds)
    """
    if model_names is None:
        model_names = STACKING_MODEL_NAMES
    if patch_size is None:
        patch_size = STACKING_PATCH_SIZE
    if overlap is None:
        overlap = STACKING_OVERLAP

    # Build features from cache
    features, preds, mask = build_stacking_features(cache_path, model_names)

    # Run stacking sliding window inference at 128^3
    stacking_prob = sliding_window_inference(
        model, features, patch_size, device, overlap=overlap
    )

    # Build agreement map: count models predicting positive (> 0.5 for each)
    agreement = np.zeros_like(stacking_prob)
    for i in range(preds.shape[0]):
        agreement += (preds[i] > 0.5).astype(np.float32)

    # Build individual predictions dict
    individual = {}
    for i, name in enumerate(model_names):
        individual[name] = preds[i]

    # Upsample to target size if needed
    if target_size is not None and tuple(target_size) != tuple(stacking_prob.shape):
        factors = [t / s for t, s in zip(target_size, stacking_prob.shape)]
        stacking_prob = zoom(stacking_prob.astype(np.float32), factors, order=1)
        agreement = zoom(agreement.astype(np.float32), factors, order=0)
        for name in individual:
            individual[name] = zoom(individual[name].astype(np.float32), factors, order=1)

    return {
        'fused': stacking_prob.astype(np.float32),
        'individual': {k: v.astype(np.float32) for k, v in individual.items()},
        'agreement': agreement.astype(np.float32),
    }
