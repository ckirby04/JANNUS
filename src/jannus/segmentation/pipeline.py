"""
Production BrainMetScan inference pipeline (full-volume 7-base + StackingClassifierV2).

Pipeline flow
-------------
    Full 4-channel MRI volume (T1, T1ce, FLAIR, T2; (4, H, W, D) at native res)
        |
        v
    7 base segmentation models, each producing a (H, W, D) probability map
    at the input native size:
        - nnU-Net 3D       (nnUNetPredictor, native preprocessing)
        - nnU-Net 2D       (nnUNetPredictor, native preprocessing)
        - patch_8 / patch_12 / patch_24 / patch_36
                           (LightweightUNet3D — cache-build protocol:
                            resample to 128^3, sliding-window at native ps,
                            upsample back to (H, W, D))
        - SwinUNETR        (MONAI sliding window, 96^3 ROI)
        |
        v
    Stack into a 9-channel feature cube
        (7 predictions + voxel-wise variance + (max - min) range)
        |
        v
    StackingClassifierV2 (~1.36M params) — 32^3 sliding window, 0.5 overlap,
    fully-convolutional residual CNN with SE attention and a multi-scale
    branch
        |
        v
    Threshold + postprocess (min_component_size = 0)
        |
        v
    Binary segmentation mask + per-lesion details

This faithfully replicates the cache-build protocol from
1.30/scripts/training/native_overnight.py — the same protocol used to
produce the predictions the production stacker was trained against.

Headline metrics (held-out 84-case eval; see docs/PRODUCTION.md):
    voxel Dice 0.7858, lesion-wise sens 0.810, FPs/case 1.68
    RANO-BM measurable-disease (>=10mm): sens 0.943, matched Dice 0.835

Design notes
------------
* Each adapter is a small, testable unit. The smoke test instantiates
  every adapter with `stub=True` (random-weight 3D CNN), runs a synthetic
  4-channel volume through the full pipeline, and checks output shapes.
* nnDetection is NOT part of this pipeline (it was never part of the
  workflow that produced the documented production metrics). The
  detection adapter remains available under `src/segmentation/detection.py`
  for any caller that wants the legacy detect-then-segment flow.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from .model_adapters import SegmentationModelAdapter, build_adapter
from .postprocessing import extract_lesion_details, full_postprocessing_pipeline
from .stacking import (
    STACKING_IN_CHANNELS,
    STACKING_MODEL_NAMES,
    STACKING_OVERLAP,
    STACKING_PATCH_SIZE,
    StackingClassifier,
    StackingClassifierV2,
    build_stacking_features_from_preds,
    load_stacking_model,
    sliding_window_inference,
)

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    """Output of `BrainMetPipeline.predict_volume`."""

    probability_map: np.ndarray      # (1, H, W, D) — fused stacker output
    binary_mask: np.ndarray          # (H, W, D)
    lesion_count: int
    lesion_details: list[dict]
    # Per-base full-volume probability maps, keyed by adapter name. Useful
    # for debugging / visualisation; stays empty when callers don't need it
    # (the pipeline always populates it though — cost is small relative to
    # the stacker run).
    base_probs: dict[str, np.ndarray] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return the legacy dict schema (same keys as SmartEnsemble)."""
        return {
            "probability_map": self.probability_map,
            "binary_mask": self.binary_mask,
            "lesion_count": self.lesion_count,
            "lesion_details": self.lesion_details,
        }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class BrainMetPipeline:
    """End-to-end production brain metastasis segmentation pipeline.

    Class name kept stable for back-compat; internals implement the
    full-volume 7-base + StackingClassifierV2 production architecture
    documented in docs/PRODUCTION.md.
    """

    def __init__(
        self,
        model_adapters: Sequence[SegmentationModelAdapter],
        stacking_model: StackingClassifier | StackingClassifierV2 | None,
        config: dict[str, Any],
        device: torch.device | None = None,
    ):
        self.model_adapters = list(model_adapters)
        self.stacking_model = stacking_model
        self.config = config
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # Tunables
        inf = config.get("inference", {})
        self.threshold = float(inf.get("threshold", 0.5))
        pp = inf.get("postprocessing", {}) or {}
        self.pp_min_size = int(pp.get("min_size", 0))
        self.pp_opening = int(pp.get("opening_size", 1))
        self.pp_closing = int(pp.get("closing_size", 1))

        stk = config.get("stacking", {}) or {}
        self.stacker_patch = int(stk.get("patch_size", STACKING_PATCH_SIZE))
        self.stacker_overlap = float(stk.get("overlap", STACKING_OVERLAP))

        # Validate that the adapters cover the canonical model-name list in
        # the order the stacker expects.
        adapter_names = [a.name for a in self.model_adapters]
        expected = list(STACKING_MODEL_NAMES)
        if adapter_names != expected:
            raise ValueError(
                f"Adapter ordering does not match stacker training order. "
                f"Expected {expected}; got {adapter_names}."
            )

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        config_path: str | Path,
        device: torch.device | None = None,
        stub: bool = False,
        verify: bool = False,
    ) -> BrainMetPipeline:
        """Build the production pipeline from `configs/models.yaml`.

        Args:
            config_path: path to models.yaml
            device: torch device (default: cuda if available)
            stub: if True, every base model is replaced by a small
                random-weight 3D CNN. Used by the smoke test and by fresh
                clones without trained weights.
            verify: if True (and stub is False), confirm every adapter
                has its real model loaded and the stacker is loaded
                before returning. Raises RuntimeError listing any silent
                fall-throughs detected. Off by default for back-compat
                with callers that did not previously opt into this check.
        """
        config_path = Path(config_path)
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}

        device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # Build base adapters in the canonical STACKING_MODEL_NAMES order.
        model_entries = {m.get("name"): m for m in cfg.get("models", [])}
        adapters: list[SegmentationModelAdapter] = []
        for name in STACKING_MODEL_NAMES:
            entry = model_entries.get(name)
            if entry is None:
                raise ValueError(
                    f"Config at {config_path} is missing required model "
                    f"entry '{name}'. Production pipeline expects all of "
                    f"{list(STACKING_MODEL_NAMES)}."
                )
            adapters.append(build_adapter(entry, device=device, stub=stub))

        # Stacking meta-learner.
        stacking_cfg = cfg.get("stacking", {}) or {}
        model_dir = config_path.parent.parent / "model"
        in_channels = int(stacking_cfg.get("in_channels", STACKING_IN_CHANNELS))
        ckpt_name = stacking_cfg.get("classifier_path")
        if ckpt_name:
            # Strip the leading "model/" if present so load_stacking_model
            # can join it with model_dir cleanly.
            ckpt_name = Path(ckpt_name).name
        stacking_model = load_stacking_model(
            model_dir=model_dir,
            device=device,
            in_channels=in_channels,
            allow_untrained=stub,
            checkpoint_name=ckpt_name,
        )

        pipeline = cls(
            model_adapters=adapters,
            stacking_model=stacking_model,
            config=cfg,
            device=device,
        )

        if verify and not stub:
            cls._verify_no_silent_fallback(pipeline)

        return pipeline

    @staticmethod
    def _verify_no_silent_fallback(pipeline: BrainMetPipeline) -> None:
        """Raise if any adapter is in stub mode or the stacker is missing.

        With the loader-hardening patch (raise-on-missing in every
        adapter and load_stacking_model), a missing checkpoint already
        fails at construction time. This check additionally surfaces the
        nnU-Net "import-fallback" stub, which fires when nnunetv2 is
        unavailable in the environment, plus any future adapter that
        introduces a silent stub mode.
        """
        issues: list[str] = []
        for adapter in pipeline.model_adapters:
            if getattr(adapter, "_stub", False):
                issues.append(
                    f"adapter[{adapter.name}]: in stub mode "
                    f"(no real weights — random-init forward pass)"
                )
            if (hasattr(adapter, "_predictor")
                    and getattr(adapter, "_predictor", None) is None
                    and not getattr(adapter, "_stub", False)):
                issues.append(
                    f"adapter[{adapter.name}]: nnUNetPredictor is None "
                    f"(initialise failed silently)"
                )
        if pipeline.stacking_model is None:
            issues.append(
                "stacking_model: not loaded (predict_volume would fall "
                "back to per-voxel base mean)"
            )
        if issues:
            raise RuntimeError(
                "BrainMetPipeline.from_config(verify=True) detected "
                "silent fall-throughs:\n  - " + "\n  - ".join(issues)
            )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_volume(
        self,
        volume: np.ndarray | torch.Tensor,
        voxel_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> PipelineResult:
        """Run the full production pipeline on a 4-channel MRI volume.

        Args:
            volume: (C, H, W, D) array — expected 4 channels (T1, T1ce,
                FLAIR, T2), z-score normalised per channel by the upstream
                loader.
            voxel_spacing: mm spacing for lesion-level stats.
        """
        if isinstance(volume, torch.Tensor):
            vol_np = volume.detach().cpu().numpy().astype(np.float32)
        else:
            vol_np = np.asarray(volume, dtype=np.float32)
        if vol_np.ndim != 4:
            raise ValueError(
                f"Expected (C, H, W, D) volume; got shape {vol_np.shape}"
            )
        C, H, W, D = vol_np.shape
        vol_t = torch.from_numpy(np.ascontiguousarray(vol_np))

        # =================================================================
        # Stage 1 — run every base model on the full volume.
        # Each adapter returns a (H, W, D) probability map. Order matches
        # STACKING_MODEL_NAMES, which is the order the stacker was trained
        # to expect.
        # =================================================================
        base_preds: list[np.ndarray] = []
        base_probs: dict[str, np.ndarray] = {}
        for adapter in self.model_adapters:
            pred = adapter.predict_crop(vol_t)
            if isinstance(pred, torch.Tensor):
                pred = pred.detach().cpu().numpy()
            pred = pred.astype(np.float32)
            if pred.shape != (H, W, D):
                raise RuntimeError(
                    f"Adapter {adapter.name!r} returned shape {pred.shape}, "
                    f"expected {(H, W, D)} (full input volume). Each adapter "
                    f"must produce a probability map matching the input "
                    f"spatial dims."
                )
            base_preds.append(pred)
            base_probs[adapter.name] = pred

        preds_stack = np.stack(base_preds, axis=0)  # (7, H, W, D)

        # =================================================================
        # Stage 2 — build 9-channel features and run StackingClassifierV2.
        # =================================================================
        features = build_stacking_features_from_preds(preds_stack)  # (9, H, W, D)
        if self.stacking_model is None:
            # No stacker available — degrade to per-voxel mean of the base
            # predictions so callers still get a usable map.
            fused = preds_stack.mean(axis=0)
        else:
            fused = sliding_window_inference(
                self.stacking_model, features,
                self.stacker_patch, self.device,
                overlap=self.stacker_overlap,
            ).astype(np.float32)

        probability_map = fused[np.newaxis, ...]  # (1, H, W, D)

        # =================================================================
        # Stage 3 — threshold + postprocess + lesion summarisation.
        # =================================================================
        binary_mask = full_postprocessing_pipeline(
            probability_map[0],
            threshold=self.threshold,
            min_size=self.pp_min_size,
            opening_size=self.pp_opening,
            closing_size=self.pp_closing,
        )
        lesion_details = extract_lesion_details(
            binary_mask, probability_map[0], voxel_spacing=voxel_spacing
        )

        return PipelineResult(
            probability_map=probability_map,
            binary_mask=binary_mask,
            lesion_count=len(lesion_details),
            lesion_details=lesion_details,
            base_probs=base_probs,
        )
