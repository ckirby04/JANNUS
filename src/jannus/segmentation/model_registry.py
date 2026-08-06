"""
Model Registry for managing trained segmentation model checkpoints.
Provides registration, validation, and loading of model files.
"""

import shutil
from pathlib import Path

import torch
import yaml

from ..core import paths as _paths


class ModelRegistry:
    """
    Registry for managing trained segmentation model checkpoints.

    Handles copying externally-trained models into the project,
    validating checkpoint integrity, and providing model configs
    for the ensemble inference pipeline.
    """

    def __init__(self, project_root: Path | None = None):
        if project_root is None:
            project_root = _paths.repo_root()
        self.project_root = Path(project_root)
        self.model_dir = self.project_root / "model"
        self.config_path = self.project_root / "configs" / "models.yaml"
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def register_model(
        self,
        source_path: str,
        name: str,
        patch_size: int,
        architecture: str = "lightweight",
        threshold: float = 0.5,
        base_channels: int = 16,
        depth: int = 3,
        use_attention: bool = True,
        use_residual: bool = True,
    ) -> Path:
        """
        Copy model checkpoint into the project and add to config.

        Args:
            source_path: Path to the externally-trained .pth file
            name: Model name identifier
            patch_size: Patch size used during training
            architecture: 'lightweight' or 'deep_supervised'
            threshold: Optimal inference threshold
            base_channels: Base channel count
            depth: Network depth
            use_attention: Whether model uses attention gates
            use_residual: Whether model uses residual connections

        Returns:
            Path to the registered model file
        """
        source = Path(source_path)
        if not source.exists():
            raise FileNotFoundError(f"Source model not found: {source}")

        # Validate checkpoint before copying
        self.validate_checkpoint(str(source), architecture)

        # Copy to model directory
        dest = self.model_dir / f"{name}_best.pth"
        shutil.copy2(source, dest)

        # Update config
        config = self._load_config()
        model_entry = {
            "name": name,
            "path": f"model/{dest.name}",
            "architecture": architecture,
            "patch_size": patch_size,
            "threshold": threshold,
            "base_channels": base_channels,
            "depth": depth,
            "use_attention": use_attention,
            "use_residual": use_residual,
        }

        # Replace existing entry with same name, or append
        existing = [m for m in config.get("models", []) if m["name"] != name]
        existing.append(model_entry)
        config["models"] = existing
        self._save_config(config)

        print(f"Registered model '{name}' -> {dest}")
        return dest

    # Architectures that ship their own state_dict key schemas and should NOT
    # be validated against the LightweightUNet3D layer names. nnU-Net 3D/2D
    # checkpoints are managed by nnUNetPredictor, not this registry, so they
    # are never validated here either.
    _EXTERNALLY_MANAGED_ARCHS = {"swin_unetr", "nnunet_3d", "nnunet_2d"}

    def validate_checkpoint(self, path: str, architecture: str = "lightweight") -> bool:
        """
        Validate that a checkpoint file is loadable and has expected keys.

        Args:
            path: Path to checkpoint file
            architecture: Expected architecture type

        Returns:
            True if valid

        Raises:
            ValueError: If checkpoint is invalid
        """
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as e:
            raise ValueError(f"Cannot load checkpoint: {e}") from e

        if "model_state_dict" not in checkpoint:
            raise ValueError("Checkpoint missing 'model_state_dict' key")

        state_dict = checkpoint["model_state_dict"]

        # Architectures with foreign state_dict schemas (SwinUNETR, nnU-Net)
        # get a lightweight check: loadable + non-empty. Layer-name inspection
        # is specific to our in-repo LightweightUNet3D/DeepSupervisedUNet3D.
        if architecture in self._EXTERNALLY_MANAGED_ARCHS:
            if not state_dict:
                raise ValueError(f"Empty state_dict for architecture '{architecture}'")
            return True

        # Check for expected LightweightUNet3D / DeepSupervisedUNet3D layers
        has_inc = any(k.startswith("inc.") for k in state_dict.keys())
        has_down = any(k.startswith("down_blocks.") for k in state_dict.keys())
        has_up = any(k.startswith("up_blocks.") for k in state_dict.keys())
        has_outc = any(k.startswith("outc.") for k in state_dict.keys())

        if not all([has_inc, has_down, has_up, has_outc]):
            raise ValueError("Checkpoint state_dict missing expected UNet layers")

        if architecture == "deep_supervised":
            has_bottleneck = any(k.startswith("bottleneck.") for k in state_dict.keys())
            if not has_bottleneck:
                raise ValueError("Deep supervised checkpoint missing bottleneck layer")

        return True

    def list_models(self) -> list[dict]:
        """List all registered models with their configs."""
        config = self._load_config()
        models = config.get("models", [])

        result = []
        for m in models:
            entry = {**m, "exists": self._entry_exists(m)}
            result.append(entry)

        return result

    def get_ensemble_config(self) -> dict:
        """
        Get ensemble configuration for loading models.

        Returns:
            Dict with 'models' list and 'ensemble' settings
        """
        config = self._load_config()
        models = config.get("models", [])

        # Filter to only models that exist on disk.
        available = []
        for m in models:
            if not self._entry_exists(m):
                continue
            entry = {**m}
            if m.get("path"):
                entry["full_path"] = str(self.project_root / m["path"])
            available.append(entry)

        return {
            "models": available,
            "ensemble": config.get("ensemble", {"fusion_mode": "union"}),
            "inference": config.get("inference", {}),
        }

    def _entry_exists(self, m: dict) -> bool:
        """Test whether the model files for an entry are on disk.

        nnU-Net entries don't have a single ``.pth`` path — they live under
        a trainer directory tree that nnUNetPredictor walks. For those
        entries, check the trainer dir under nnUNet/nnUNet_results/.
        Other entries use the standard ``path`` field.
        """
        if m.get("path"):
            return (self.project_root / m["path"]).exists()
        if m.get("dataset_id") and m.get("trainer"):
            trainer_dir = (self.project_root / "nnUNet" / "nnUNet_results"
                           / m["dataset_id"] / m["trainer"])
            return trainer_dir.exists()
        return False

    def _load_config(self) -> dict:
        if self.config_path.exists():
            with open(self.config_path) as f:
                return yaml.safe_load(f) or {}
        return {"ensemble": {"fusion_mode": "union"}, "models": [], "inference": {}}

    def _save_config(self, config: dict):
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
