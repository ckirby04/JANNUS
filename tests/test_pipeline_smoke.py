"""
Smoke test for the production BrainMetScan pipeline.

Verifies end-to-end wiring with zero real weights and zero real MRI data:

  1. The 9-channel StackingClassifierV2 instantiates and runs on a
     synthetic 9-channel input.
  2. `build_stacking_features_from_preds` produces the expected
     (N+2, H, W, D) shape from N base predictions.
  3. Each base adapter (nnU-Net 3D/2D, SwinUNETR, LightweightUNet3D)
     can be instantiated in stub mode and returns a (H, W, D) probability
     map matching its input.
  4. The full pipeline (7 stub bases + untrained StackingClassifierV2)
     runs on a synthetic 4-channel volume and returns valid output.

The test uses `stub=True`, which swaps every base model for a tiny
randomly-initialised 3D CNN, so no MONAI model-zoo download, no nnU-Net
checkpoint tree, and no patch-model weights are required.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from jannus.segmentation.model_adapters import build_adapter
from jannus.segmentation.pipeline import BrainMetPipeline, PipelineResult
from jannus.segmentation.stacking import (
    STACKING_IN_CHANNELS,
    STACKING_MODEL_NAMES,
    StackingClassifierV2,
    build_stacking_features_from_preds,
)

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "models.yaml"


# ---------------------------------------------------------------------------
# Stacker-only checks
# ---------------------------------------------------------------------------

def test_stacking_model_names_are_seven():
    assert list(STACKING_MODEL_NAMES) == [
        "nnunet_3d", "nnunet_2d",
        "patch_8", "patch_12", "patch_24", "patch_36",
        "swin_unetr",
    ]
    assert STACKING_IN_CHANNELS == 9  # 7 preds + variance + range


def test_stacking_classifier_v2_accepts_nine_channel_input():
    model = StackingClassifierV2(in_channels=STACKING_IN_CHANNELS)
    model.eval()
    x = torch.randn(1, STACKING_IN_CHANNELS, 32, 32, 32)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 1, 32, 32, 32)


def test_build_stacking_features_from_preds_has_correct_shape():
    preds = np.random.rand(7, 16, 16, 16).astype(np.float32)
    features = build_stacking_features_from_preds(preds)
    assert features.shape == (9, 16, 16, 16)
    # Last two channels are variance and range; both should be non-negative.
    assert (features[-2] >= 0).all()
    assert (features[-1] >= 0).all()


# ---------------------------------------------------------------------------
# Adapter smoke tests (stub mode)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "arch,extra_cfg",
    [
        ("nnunet_3d", {}),
        ("nnunet_2d", {}),
        ("swin_unetr", {"img_size": [32, 32, 32], "feature_size": 12}),
        ("lightweight_unet3d", {"patch_size": 8}),
        ("lightweight_unet3d", {"patch_size": 12}),
        ("lightweight_unet3d", {"patch_size": 24}),
        ("lightweight_unet3d", {"patch_size": 36}),
    ],
)
def test_adapter_stub_predicts_correct_shape(arch, extra_cfg):
    cfg = {"architecture": arch, "in_channels": 4, "out_channels": 1,
           "threshold": 0.5}
    cfg.update(extra_cfg)
    adapter = build_adapter(cfg, device=torch.device("cpu"), stub=True)
    volume = torch.randn(4, 24, 24, 24)
    prob = adapter.predict_crop(volume)
    assert prob.shape == (24, 24, 24)
    arr = prob.numpy() if isinstance(prob, torch.Tensor) else prob
    assert (arr >= 0).all() and (arr <= 1).all()


# ---------------------------------------------------------------------------
# Full pipeline end-to-end
# ---------------------------------------------------------------------------

def test_pipeline_from_config_stub_end_to_end():
    """The full production pipeline runs on a synthetic volume in stub mode."""
    pipeline = BrainMetPipeline.from_config(
        CONFIG_PATH, device=torch.device("cpu"), stub=True,
    )

    # 7 adapters in canonical order, plus an (untrained) stacker.
    assert len(pipeline.model_adapters) == 7
    assert [a.name for a in pipeline.model_adapters] == list(STACKING_MODEL_NAMES)
    assert pipeline.stacking_model is not None

    # Synthetic volume: small enough to run on CPU quickly.
    C, H, W, D = 4, 48, 48, 48
    rng = np.random.default_rng(0)
    volume = rng.standard_normal((C, H, W, D)).astype(np.float32)

    result = pipeline.predict_volume(volume)

    # Shape checks.
    assert isinstance(result, PipelineResult)
    assert result.probability_map.shape == (1, H, W, D)
    assert result.binary_mask.shape == (H, W, D)

    # Value range checks.
    prob = result.probability_map[0]
    assert np.isfinite(prob).all()
    assert (prob >= 0).all() and (prob <= 1).all()
    assert result.binary_mask.dtype != np.float64

    # Per-base full-volume probability maps were captured for diagnostics.
    assert set(result.base_probs.keys()) == set(STACKING_MODEL_NAMES)
    for name, p in result.base_probs.items():
        assert p.shape == (H, W, D), f"base {name!r} returned shape {p.shape}"

    # Legacy dict schema is preserved.
    d = result.as_dict()
    assert set(d.keys()) == {"probability_map", "binary_mask",
                              "lesion_count", "lesion_details"}
