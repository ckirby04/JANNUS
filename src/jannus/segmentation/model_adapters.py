"""
Segmentation model adapters — a uniform interface over nnU-Net 3D, nnU-Net
2D, SwinUNETR, and the 4 LightweightUNet3D patch variants so the production
pipeline can call them identically.

Design rationale
----------------
The seven base models we ensemble come from different runtimes:

  * nnU-Net 3D / 2D run through `nnUNetPredictor`, which owns its own
    sliding-window, preprocessing, and checkpoint management. It is not a
    plain `torch.nn.Module` we can call `.forward()` on.
  * SwinUNETR is a regular MONAI module that operates on 96^3 windows via
    MONAI's sliding_window_inference; it accepts any input spatial size.
  * The 4 patch models (patch_8/12/24/36) are LightweightUNet3D instances.
    Per the cache-build protocol used to produce the predictions the
    production stacker was trained against (see
    1.30/scripts/training/native_overnight.py), they are run on the full
    volume *resampled to 128^3*, with a sliding window at the model's
    native patch size and 0.5 overlap, then upsampled back to the target
    size. Mimicking this exactly preserves the training-distribution match
    that produced the documented Dice 0.7858.

Every base model is wrapped in a `SegmentationModelAdapter` that exposes
one method:

    predict_crop(volume_tensor: torch.Tensor) -> torch.Tensor
        # volume_tensor: (C, H, W, D) float on CPU or GPU
        # returns:      (H, W, D) float foreground probability in [0, 1]

The name `predict_crop` is historical: the production pipeline now calls
each adapter on the *full volume*, not on detection crops. (The legacy
detect-then-segment pipeline passed crops here.)

The pipeline then fuses the seven probability maps through the
StackingClassifierV2 meta-learner. All adapters accept a `stub` flag which
substitutes a small randomly-initialised 3D CNN — this is what makes the
smoke test runnable with no trained weights and no real MRI data.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from ..core import paths as _paths

# ---------------------------------------------------------------------------
# Base adapter
# ---------------------------------------------------------------------------

class SegmentationModelAdapter:
    """Uniform interface over every segmentation base model."""

    name: str = "base"

    def predict_crop(self, volume: torch.Tensor) -> torch.Tensor:
        """Return a (H, W, D) foreground probability map for a (C, H, W, D)
        crop. Implementations must handle their own preprocessing.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Stub backbone — used when real weights are unavailable (smoke tests, CI)
# ---------------------------------------------------------------------------

class _StubSegNet(nn.Module):
    """A very small 3D CNN with 4-channel input and 1-channel output.

    Used when nnU-Net/SwinUNETR weights are not available. The smoke test
    only checks shapes and end-to-end wiring, not accuracy.
    """

    def __init__(self, in_channels: int = 4, base: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, base, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(base, base, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(base, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _run_stub(stub: _StubSegNet, volume: torch.Tensor,
              device: torch.device) -> torch.Tensor:
    x = volume.unsqueeze(0).to(device)  # (1, C, H, W, D)
    with torch.no_grad():
        logits = stub(x)
        prob = torch.sigmoid(logits)[0, 0]  # (H, W, D)
    return prob.cpu()


# ---------------------------------------------------------------------------
# SwinUNETR adapter
# ---------------------------------------------------------------------------

class SwinUNETRAdapter(SegmentationModelAdapter):
    """Wraps a MONAI SwinUNETR instance with a sliding-window crop runner.

    When `stub=True` (no checkpoint / smoke test), a tiny placeholder network
    is used instead of SwinUNETR so the pipeline can still be exercised.
    """

    name = "swin_unetr"

    def __init__(
        self,
        cfg: dict[str, Any],
        device: torch.device | None = None,
        stub: bool = False,
    ):
        self.cfg = cfg
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.img_size = tuple(cfg.get("img_size", [96, 96, 96]))
        self.threshold = float(cfg.get("threshold", 0.5))
        self._stub = stub

        if stub:
            self.model: nn.Module = _StubSegNet(
                in_channels=int(cfg.get("in_channels", 4))).to(self.device)
            self.model.eval()
            return

        # Real path: build SwinUNETR and load weights in order of preference:
        #   1. Our own fine-tuned checkpoint at `cfg["path"]` (if it exists).
        #   2. MONAI BTCV warm-start at `cfg["pretrained"]` (if it exists).
        #   3. Fall through to random init.
        from .swin_unetr_model import (
            build_swin_unetr_from_dict,
            load_pretrained_backbone,
        )
        self.model = build_swin_unetr_from_dict(cfg).to(self.device)
        self.model.eval()

        import logging
        _log = logging.getLogger(__name__)

        ckpt = cfg.get("path")
        if ckpt:
            ckpt_path = Path(ckpt)
            if not ckpt_path.exists():
                raise FileNotFoundError(
                    f"SwinUNETRAdapter: configured checkpoint "
                    f"models.swin_unetr.path={ckpt!r} does not resolve "
                    f"to an existing file (resolved={ckpt_path.resolve()}). "
                    f"Relative paths are resolved against the current "
                    f"working directory; run from the project root or "
                    f"pass an absolute path."
                )
            state = torch.load(ckpt_path, map_location=self.device,
                               weights_only=False)
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            try:
                self.model.load_state_dict(state, strict=True)
            except RuntimeError as exc:
                result = self.model.load_state_dict(state, strict=False)
                _log.warning(
                    "SwinUNETRAdapter: strict load of %s failed (%s); "
                    "continuing with strict=False. missing_keys=%d "
                    "unexpected_keys=%d. missing_sample=%s "
                    "unexpected_sample=%s",
                    ckpt_path, exc.__class__.__name__,
                    len(result.missing_keys), len(result.unexpected_keys),
                    list(result.missing_keys)[:5],
                    list(result.unexpected_keys)[:5],
                )
        else:
            pretrained = cfg.get("pretrained")
            if pretrained:
                pretrained_path = Path(pretrained)
                if not pretrained_path.exists():
                    raise FileNotFoundError(
                        f"SwinUNETRAdapter: configured pretrained backbone "
                        f"models.swin_unetr.pretrained={pretrained!r} does "
                        f"not resolve to an existing file "
                        f"(resolved={pretrained_path.resolve()}). Relative "
                        f"paths are resolved against the current working "
                        f"directory; run from the project root or pass an "
                        f"absolute path."
                    )
                load_pretrained_backbone(self.model, pretrained_path)

    @torch.no_grad()
    def predict_crop(self, volume: torch.Tensor) -> torch.Tensor:
        if self._stub:
            return _run_stub(self.model, volume, self.device)

        # Real SwinUNETR forward: use MONAI sliding window if the crop is
        # larger than img_size, otherwise a direct forward on a padded crop.
        from monai.inferers import sliding_window_inference

        C, H, W, D = volume.shape
        x = volume.unsqueeze(0).to(self.device)
        logits = sliding_window_inference(
            inputs=x,
            roi_size=self.img_size,
            sw_batch_size=1,
            predictor=self.model,
            overlap=0.5,
            mode="gaussian",
        )
        # (1, 2, H, W, D) -> foreground channel probability
        prob = torch.softmax(logits, dim=1)[0, 1]
        return prob.cpu()


# ---------------------------------------------------------------------------
# nnU-Net 3D / 2D adapters
# ---------------------------------------------------------------------------

class _NnUNetAdapter(SegmentationModelAdapter):
    """Shared logic for nnU-Net 3D and nnU-Net 2D predictors.

    Because `nnUNetPredictor` owns its own sliding window + preprocessing,
    we surface a `predict_crop` call that passes the crop through the
    predictor's in-memory prediction API and grabs the foreground softmax
    channel. If nnU-Net is not installed *or* the checkpoint path is
    missing, a stub network is used instead.
    """

    configuration: str = ""  # "3d_fullres" | "2d"

    def __init__(
        self,
        cfg: dict[str, Any],
        device: torch.device | None = None,
        stub: bool = False,
    ):
        self.cfg = cfg
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = float(cfg.get("threshold", 0.5))

        self._stub = stub
        self._predictor = None
        if stub:
            self._stub_model = _StubSegNet(in_channels=4).to(self.device)
            self._stub_model.eval()
            return

        self._try_load_real_predictor()

    def _try_load_real_predictor(self) -> None:
        """Attempt to initialise a real nnUNetPredictor.

        TODO(manual): this requires nnU-Net to be installed AND a trained
        checkpoint tree under nnUNet/nnUNet_results/<dataset>/<trainer>/.
        If either is missing we fall back to the stub network so the
        pipeline remains runnable for development / smoke tests.
        """
        try:
            from nnunetv2.inference.predict_from_raw_data import (
                nnUNetPredictor,
            )
        except Exception:
            self._stub = True
            self._stub_model = _StubSegNet(in_channels=4).to(self.device)
            self._stub_model.eval()
            return

        # v1.50: honour nnU-Net's own environment variable before falling back
        # to the in-repo tree. v1.40 hardcoded the repo-relative path, so a site
        # with its trained models on a shared mount could not point at them.
        # `nnUNet_results` is nnU-Net's actual variable name; the mixed case is
        # required and is not a style slip.
        env_results = os.environ.get("nnUNet_results")  # noqa: SIM112
        results_root = (
            Path(env_results) if env_results
            else _paths.resource("nnUNet", "nnUNet_results")
        )
        model_dir = results_root / self.cfg.get("dataset_id", "") / self.cfg.get("trainer", "")
        if not model_dir.exists():
            raise FileNotFoundError(
                f"_NnUNetAdapter[{self.name}]: trained model directory "
                f"does not exist. Expected: {model_dir}. Configured via "
                f"models.{self.name}.dataset_id="
                f"{self.cfg.get('dataset_id')!r}, "
                f"trainer={self.cfg.get('trainer')!r}."
            )

        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=False,
            device=self.device,
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=False,
        )
        predictor.initialize_from_trained_model_folder(
            str(model_dir),
            use_folds=tuple(self.cfg.get("folds", (0,))),
            checkpoint_name=self.cfg.get("checkpoint", "checkpoint_final.pth"),
        )
        self._predictor = predictor

    @torch.no_grad()
    def predict_crop(self, volume: torch.Tensor) -> torch.Tensor:
        if self._stub or self._predictor is None:
            return _run_stub(self._stub_model, volume, self.device)

        # Real nnU-Net path: predictor expects a (C, X, Y, Z) numpy array plus
        # a minimal properties dict with spacing. We treat the crop as having
        # 1.0-mm isotropic spacing — spec says to preserve existing
        # preprocessing, which already handles real spacing upstream.
        props = {"spacing": [1.0, 1.0, 1.0]}
        arr = volume.cpu().numpy().astype(np.float32)
        # predict_single_npy_array returns (X, Y, Z) argmax by default; we
        # need the softmax foreground channel, so we enable save_probabilities.
        result = self._predictor.predict_single_npy_array(
            arr, props, None, None, True
        )
        # result = (seg, probabilities) when save_probabilities=True
        if isinstance(result, tuple):
            _, probs = result
        else:
            probs = result
        probs = np.asarray(probs)
        # Expected shape (n_classes, X, Y, Z); foreground is channel 1.
        if probs.ndim == 4 and probs.shape[0] >= 2:
            fg = probs[1]
        else:
            fg = probs  # already foreground-only
        return torch.from_numpy(fg.astype(np.float32))


class NnUNet3DAdapter(_NnUNetAdapter):
    name = "nnunet_3d"
    configuration = "3d_fullres"


class NnUNet2DAdapter(_NnUNetAdapter):
    name = "nnunet_2d"
    configuration = "2d"


# ---------------------------------------------------------------------------
# LightweightUNet3D patch-model adapter (patch_8 / patch_12 / patch_24 / patch_36)
# ---------------------------------------------------------------------------

class LightweightUNet3DAdapter(SegmentationModelAdapter):
    """Adapter for the 4 patch models (patch_8/12/24/36).

    Replicates the cache-build protocol used to produce the predictions
    the production stacker was trained against:

        1. Resample the input volume from native size to 128^3 (linear).
        2. Per-channel z-score normalise.
        3. Sliding-window inference with the model's native patch size
           (8 / 12 / 24 / 36) at 0.5 overlap.
        4. Upsample the resulting probability map from 128^3 back to the
           original spatial size (linear).

    Departing from this protocol meaningfully shifts the per-voxel
    probability distribution and degrades stacker accuracy, so this
    adapter does NOT support arbitrary inference resolutions.
    """

    name = "patch_unet"
    RESAMPLED_SIZE = (128, 128, 128)

    def __init__(
        self,
        cfg: dict[str, Any],
        device: torch.device | None = None,
        stub: bool = False,
    ):
        from .unet import LightweightUNet3D

        self.cfg = cfg
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.name = str(cfg.get("name", "patch_unet"))
        self.patch_size = int(cfg.get("patch_size"))
        self._stub = stub

        if stub:
            self.model: nn.Module = _StubSegNet(
                in_channels=int(cfg.get("in_channels", 4))).to(self.device)
            self.model.eval()
            return

        self.model = LightweightUNet3D(
            in_channels=int(cfg.get("in_channels", 4)),
            out_channels=int(cfg.get("out_channels", 1)),
            base_channels=int(cfg.get("base_channels", 20)),
            use_attention=bool(cfg.get("use_attention", True)),
            use_residual=bool(cfg.get("use_residual", True)),
            deep_supervision=bool(cfg.get("deep_supervision", True)),
        ).to(self.device)
        self.model.eval()

        ckpt_path = cfg.get("path")
        if not ckpt_path:
            raise FileNotFoundError(
                f"LightweightUNet3DAdapter[{self.name}]: config entry "
                f"has no 'path' field. Production patch models require "
                f"a trained checkpoint."
            )
        ckpt_file = Path(ckpt_path)
        if not ckpt_file.exists():
            raise FileNotFoundError(
                f"LightweightUNet3DAdapter[{self.name}]: configured "
                f"path={ckpt_path!r} does not resolve to an existing "
                f"file (resolved={ckpt_file.resolve()}). Relative paths "
                f"are resolved against the current working directory; "
                f"run from the project root or pass an absolute path."
            )
        ckpt = torch.load(ckpt_file, map_location=self.device,
                          weights_only=False)
        state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        self.model.load_state_dict(state)

    @torch.no_grad()
    def predict_crop(self, volume: torch.Tensor) -> torch.Tensor:
        """Despite the legacy method name, expects the FULL volume here.

        Returns a (H, W, D) probability map in original input dimensions.
        """
        from scipy.ndimage import zoom

        from .stacking import sliding_window_inference

        if self._stub:
            return _run_stub(self.model, volume, self.device)

        if volume.ndim != 4:
            raise ValueError(
                f"LightweightUNet3DAdapter expects (C, H, W, D); got {tuple(volume.shape)}"
            )

        vol_np = volume.detach().cpu().numpy().astype(np.float32)
        C, H, W, D = vol_np.shape
        target = self.RESAMPLED_SIZE

        # 1. Resample to 128^3 per-channel (skip if already that size)
        if target != (H, W, D):
            factors = [t / s for t, s in zip(target, (H, W, D))]
            resampled = np.stack([
                zoom(vol_np[c], factors, order=1) for c in range(C)
            ], axis=0)
        else:
            resampled = vol_np

        # 2. Per-channel z-score normalisation. The pipeline also normalises
        # upstream, but the cache-build protocol re-normalises after the
        # 128^3 resample, so we mirror that to stay distribution-faithful.
        for c in range(resampled.shape[0]):
            mean = resampled[c].mean()
            std = resampled[c].std()
            if std > 0:
                resampled[c] = (resampled[c] - mean) / std

        # 3. Sliding-window inference at the model's native patch size.
        prob_128 = sliding_window_inference(
            self.model, resampled, self.patch_size, self.device, overlap=0.5
        )

        # 4. Upsample back to native size.
        if prob_128.shape != (H, W, D):
            up_factors = [s / p for s, p in zip((H, W, D), prob_128.shape)]
            prob = zoom(prob_128, up_factors, order=1)
        else:
            prob = prob_128

        return torch.from_numpy(prob.astype(np.float32))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_adapter(
    cfg: dict[str, Any],
    device: torch.device | None = None,
    stub: bool = False,
) -> SegmentationModelAdapter:
    """Dispatch on `architecture` to build the right adapter."""
    arch = cfg.get("architecture", "")
    if arch == "swin_unetr":
        return SwinUNETRAdapter(cfg, device=device, stub=stub)
    if arch == "nnunet_3d":
        return NnUNet3DAdapter(cfg, device=device, stub=stub)
    if arch == "nnunet_2d":
        return NnUNet2DAdapter(cfg, device=device, stub=stub)
    if arch == "lightweight_unet3d":
        return LightweightUNet3DAdapter(cfg, device=device, stub=stub)
    raise ValueError(f"Unknown segmentation architecture: {arch!r}")
