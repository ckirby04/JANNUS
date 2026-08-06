"""
SmartEnsemble module for multi-model brain metastasis segmentation.
Inference-only: loads multiple models and fuses their predictions.

Status (2026-04-13):
    This class predates the two-stage pipeline rewrite. It is still used by
    tests and by any code that wants to ensemble `LightweightUNet3D` /
    `DeepSupervisedUNet3D` checkpoints directly. For new work — especially
    anything involving nnU-Net 3D/2D or SwinUNETR — use
    `src/segmentation/pipeline.py:BrainMetPipeline`, which understands the
    new model architectures and the nnDetection candidate sweep.

When loading `configs/models.yaml`, SmartEnsemble will silently skip any
model whose `architecture` is not a supported in-repo UNet variant and
print a notice pointing at BrainMetPipeline.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from .enhanced_unet import DeepSupervisedUNet3D
from .postprocessing import extract_lesion_details, full_postprocessing_pipeline
from .tta import MinimalTTA, TestTimeAugmentation
from .unet import LightweightUNet3D


class SmartEnsemble(nn.Module):
    """
    Smart ensemble that combines models with different fusion strategies:
    - union: max probability across models (maximize sensitivity)
    - weighted: learned confidence weights
    - hybrid: union for detection, average for refinement
    """

    def __init__(self, model_configs: list, device: str = "cuda", fusion_mode: str = "union"):
        """
        Args:
            model_configs: list of dicts with keys:
                name, full_path, patch_size, threshold, architecture,
                base_channels, depth, use_attention, use_residual
            device: 'cuda' or 'cpu'
            fusion_mode: 'union', 'weighted', or 'hybrid'
        """
        super().__init__()
        self.device = device
        self.fusion_mode = fusion_mode
        self.models = nn.ModuleList()
        self.patch_sizes = []
        self.thresholds = []
        self.names = []

        for cfg in model_configs:
            # Silently skip model entries whose architecture is not a
            # supported in-repo UNet. The new 3-model pipeline
            # (nnU-Net / SwinUNETR) is handled by BrainMetPipeline —
            # SmartEnsemble only knows how to load LightweightUNet3D and
            # DeepSupervisedUNet3D from local checkpoints.
            arch = cfg.get("architecture", "lightweight")
            if arch not in ("lightweight", "deep_supervised"):
                print(
                    f"  Skipping {cfg.get('name', '?')} (architecture "
                    f"{arch!r}): not supported by SmartEnsemble. Use "
                    f"jannus.segmentation.pipeline.BrainMetPipeline instead."
                )
                continue

            model_path = cfg.get("full_path", cfg.get("path", ""))
            if not Path(model_path).exists():
                print(f"  Warning: {cfg['name']} not found at {model_path}, skipping")
                continue

            print(f"  Loading {cfg['name']} (patch {cfg.get('patch_size', '?')}, threshold {cfg.get('threshold', '?')})...")
            model = self._create_model(cfg)
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            model = model.to(device)
            model.eval()

            for param in model.parameters():
                param.requires_grad = False

            self.models.append(model)
            self.patch_sizes.append(cfg.get("patch_size", 96))
            self.thresholds.append(cfg.get("threshold", 0.5))
            self.names.append(cfg["name"])

        print(f"  Loaded {len(self.models)} models")

        if fusion_mode == "weighted" and len(self.models) > 0:
            self.fusion_weights = nn.Parameter(torch.ones(len(self.models)) / len(self.models))

    @staticmethod
    def _create_model(cfg: dict) -> nn.Module:
        arch = cfg.get("architecture", "lightweight")
        if arch == "deep_supervised":
            return DeepSupervisedUNet3D(
                in_channels=4,
                out_channels=1,
                base_channels=cfg.get("base_channels", 16),
                depth=cfg.get("depth", 4),
                dropout_p=0.0,
                deep_supervision=False,  # inference only
            )
        else:
            return LightweightUNet3D(
                in_channels=4,
                out_channels=1,
                base_channels=cfg.get("base_channels", 16),
                depth=cfg.get("depth", 3),
                dropout_p=0.0,
                use_attention=cfg.get("use_attention", True),
                use_residual=cfg.get("use_residual", True),
            )

    @classmethod
    def from_config(cls, config_path: str, device: str = "cuda") -> "SmartEnsemble":
        """Factory: load ensemble from configs/models.yaml."""
        config_path = Path(config_path)
        project_root = config_path.parent.parent

        with open(config_path) as f:
            config = yaml.safe_load(f)

        model_configs = []
        for m in config.get("models", []):
            full_path = project_root / m["path"]
            model_configs.append({**m, "full_path": str(full_path)})

        fusion_mode = config.get("ensemble", {}).get("fusion_mode", "union")
        return cls(model_configs, device=device, fusion_mode=fusion_mode)

    @classmethod
    def from_registry(cls, registry, device: str = "cuda") -> "SmartEnsemble":
        """Factory: load ensemble from a ModelRegistry instance."""
        config = registry.get_ensemble_config()
        fusion_mode = config.get("ensemble", {}).get("fusion_mode", "union")
        return cls(config["models"], device=device, fusion_mode=fusion_mode)

    def forward(self, x: torch.Tensor, target_size: int | None = None) -> torch.Tensor:
        """
        Forward pass with smart fusion.

        Args:
            x: input tensor [B, C, D, H, W]
            target_size: output spatial size (default: input size)

        Returns:
            Fused probability map [B, 1, D, H, W]
        """
        if target_size is None:
            target_size = x.shape[2]

        predictions = []
        for model, patch_size in zip(self.models, self.patch_sizes):
            if patch_size != x.shape[2]:
                x_resized = F.interpolate(x, size=(patch_size,) * 3,
                                          mode="trilinear", align_corners=False)
            else:
                x_resized = x

            with torch.no_grad():
                pred = model(x_resized)
                if isinstance(pred, (list, tuple)):
                    pred = pred[0]
                pred_prob = torch.sigmoid(pred)

            if patch_size != target_size:
                pred_prob = F.interpolate(pred_prob, size=(target_size,) * 3,
                                          mode="trilinear", align_corners=False)
            predictions.append(pred_prob)

        return self._fuse(predictions)

    def _fuse(self, predictions: list[torch.Tensor]) -> torch.Tensor:
        if not predictions:
            raise ValueError("No predictions to fuse")

        stacked = torch.stack(predictions, dim=0)

        if self.fusion_mode == "union":
            return stacked.max(dim=0)[0]
        elif self.fusion_mode == "weighted":
            weights = F.softmax(self.fusion_weights, dim=0)
            return sum(w * p for w, p in zip(weights, predictions))
        elif self.fusion_mode == "hybrid":
            union = stacked.max(dim=0)[0]
            average = stacked.mean(dim=0)
            confident = (union > 0.3).float()
            return confident * union + (1 - confident) * average
        else:
            return stacked.mean(dim=0)

    def predict_with_details(self, x: torch.Tensor, target_size: int | None = None) -> dict:
        """Get predictions with per-model details for debugging."""
        if target_size is None:
            target_size = x.shape[2]

        results = {"individual": {}, "fused": None}
        predictions = []

        for model, patch_size, threshold, name in zip(
            self.models, self.patch_sizes, self.thresholds, self.names
        ):
            if patch_size != x.shape[2]:
                x_resized = F.interpolate(x, size=(patch_size,) * 3,
                                          mode="trilinear", align_corners=False)
            else:
                x_resized = x

            with torch.no_grad():
                pred = model(x_resized)
                if isinstance(pred, (list, tuple)):
                    pred = pred[0]
                pred_prob = torch.sigmoid(pred)

            if patch_size != target_size:
                pred_prob = F.interpolate(pred_prob, size=(target_size,) * 3,
                                          mode="trilinear", align_corners=False)

            predictions.append(pred_prob)
            results["individual"][name] = {
                "prediction": pred_prob,
                "threshold": threshold,
            }

        results["fused"] = self._fuse(predictions)
        return results

    @torch.no_grad()
    def predict_volume(
        self,
        image: torch.Tensor,
        window_size: tuple = (96, 96, 96),
        overlap: float = 0.75,
        use_tta: bool = False,
        tta_mode: str = "minimal",
        threshold: float = 0.5,
        postprocess: bool = True,
        postprocess_params: dict = None,
        voxel_spacing: tuple = (1.0, 1.0, 1.0),
    ) -> dict:
        """
        Full-volume prediction: sliding window + ensemble fusion + postprocessing + lesion extraction.

        Args:
            image: Input tensor (C, H, W, D)
            window_size: Sliding window size
            overlap: Overlap ratio between windows
            use_tta: Whether to use test-time augmentation
            tta_mode: 'minimal' or 'full'
            threshold: Binarization threshold
            postprocess: Whether to apply postprocessing pipeline
            postprocess_params: Override postprocessing parameters
            voxel_spacing: Voxel dimensions in mm

        Returns:
            Dict with keys:
                probability_map: (1, H, W, D) numpy array
                binary_mask: (H, W, D) numpy array
                lesion_count: int
                lesion_details: list of dicts
        """
        C, H, W, D = image.shape
        wh, ww, wd = window_size

        # Set up TTA predictor if needed
        tta_predictor = None
        if use_tta and len(self.models) > 0:
            if tta_mode == "minimal":
                tta_predictor = MinimalTTA(self, self.device)
            else:
                tta_predictor = TestTimeAugmentation(self, self.device)

        # Sliding window inference
        output = torch.zeros((1, H, W, D), device="cpu")
        count = torch.zeros((1, H, W, D), device="cpu")

        sh = int(wh * (1 - overlap))
        sw = int(ww * (1 - overlap))
        sd = int(wd * (1 - overlap))

        h_starts = list(range(0, max(1, H - wh + 1), sh))
        if wh < H and h_starts[-1] + wh < H:
            h_starts.append(H - wh)

        w_starts = list(range(0, max(1, W - ww + 1), sw))
        if ww < W and w_starts[-1] + ww < W:
            w_starts.append(W - ww)

        d_starts = list(range(0, max(1, D - wd + 1), sd))
        if wd < D and d_starts[-1] + wd < D:
            d_starts.append(D - wd)

        for h_start in h_starts:
            for w_start in w_starts:
                for d_start in d_starts:
                    window = image[:, h_start:h_start + wh,
                                   w_start:w_start + ww,
                                   d_start:d_start + wd]
                    window = window.unsqueeze(0)  # (1, C, H, W, D)

                    if use_tta and tta_predictor is not None:
                        pred, _ = tta_predictor.predict(window, threshold=threshold)
                    else:
                        window = window.to(self.device)
                        pred = self.forward(window, target_size=wh)

                    output[:, h_start:h_start + wh,
                           w_start:w_start + ww,
                           d_start:d_start + wd] += pred[0].cpu()
                    count[:, h_start:h_start + wh,
                          w_start:w_start + ww,
                          d_start:d_start + wd] += 1

        probability_map = (output / (count + 1e-8)).numpy()

        # Postprocessing
        pp = postprocess_params or {}
        if postprocess:
            binary_mask = full_postprocessing_pipeline(
                probability_map[0],
                threshold=pp.get("threshold", threshold),
                min_size=pp.get("min_size", 15),
                opening_size=pp.get("opening_size", 1),
                closing_size=pp.get("closing_size", 1),
            )
        else:
            binary_mask = (probability_map[0] > threshold).astype(np.float32)

        # Lesion extraction
        lesion_details = extract_lesion_details(
            binary_mask, probability_map[0], voxel_spacing=voxel_spacing
        )

        return {
            "probability_map": probability_map,
            "binary_mask": binary_mask,
            "lesion_count": len(lesion_details),
            "lesion_details": lesion_details,
        }
