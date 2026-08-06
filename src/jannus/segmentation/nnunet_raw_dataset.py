"""
PyTorch Dataset that reads an nnU-Net raw dataset directly.

Why this exists
---------------
`BrainMetDataset` in this repo expects per-case directories with
`t1_pre.nii.gz` / `t1_gd.nii.gz` / `flair.nii.gz` / `bravo.nii.gz` /
`seg.nii.gz`. That layout only covers the 105 original real
BrainMetShare cases under `data/train/`.

The real training pool is much bigger — 3,120 cases combining the
original real set, copy-paste synthetic, HBLI synthetic, and Yale
pseudo-labels — but it lives under
`nnUNet/nnUNet_raw/Dataset001_BrainMets/` in nnU-Net's flat
per-modality layout:

    imagesTr/{case}_0000.nii.gz   # T1_pre
    imagesTr/{case}_0001.nii.gz   # T1_Gd
    imagesTr/{case}_0002.nii.gz   # FLAIR
    imagesTr/{case}_0003.nii.gz   # T2
    labelsTr/{case}.nii.gz        # binary metastasis mask

This module reads that layout and yields 96^3 random patches with
foreground bias — the same training regime nnU-Net uses internally.
SwinUNETR needs every spatial dim divisible by 32, so 96^3 is the
natural patch size (and matches the `models.swin_unetr.img_size`
config).

Patch sampling strategy
-----------------------
* `foreground_ratio` (default 0.67): fraction of patches drawn from
  random foreground voxels (i.e. guaranteed to contain lesion). The
  rest are drawn uniformly from the brain region.
* Foreground voxel indices are computed once per case on first access
  and cached in memory. For 3,120 cases this is cheap — we store only
  up to `max_fg_sample` voxels per case and sample from them.
* Patches that would run off the volume edge are clipped inward; the
  returned patch is always exactly `patch_size`.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset


class NnUNetRawDataset(Dataset):
    """Reads an nnU-Net raw dataset and yields random 3D patches.

    Parameters
    ----------
    dataset_root : path to the Dataset001_BrainMets raw directory, i.e.
        the folder containing `imagesTr/` and `labelsTr/`.
    case_ids : list of case IDs to include (file stem of the label
        mask, e.g. 'SITE_0001'). If None, every label
        under labelsTr/*.nii.gz is included.
    patch_size : spatial dims of each yielded patch. Must be divisible
        by 32 for SwinUNETR.
    samples_per_case : number of random patches sampled per case per
        "epoch" — the returned `__len__` is `len(case_ids) *
        samples_per_case`, so a standard epoch hits every case roughly
        `samples_per_case` times.
    foreground_ratio : fraction of patches biased to contain lesion
        voxels. The remainder are uniformly random.
    normalize : per-volume, per-channel z-score normalisation. Matches
        BrainMetDataset behaviour.
    max_fg_sample : cap on foreground voxels cached per case.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        case_ids: Sequence[str] | None = None,
        patch_size: tuple[int, int, int] = (96, 96, 96),
        samples_per_case: int = 2,
        foreground_ratio: float = 0.67,
        normalize: bool = True,
        max_fg_sample: int = 20_000,
    ):
        self.root = Path(dataset_root)
        self.images_dir = self.root / "imagesTr"
        self.labels_dir = self.root / "labelsTr"
        if not self.images_dir.exists() or not self.labels_dir.exists():
            raise FileNotFoundError(
                f"nnU-Net raw dataset not found at {self.root}; "
                f"expected imagesTr/ and labelsTr/ subdirectories.")

        if case_ids is None:
            case_ids = sorted(
                p.name.replace(".nii.gz", "")
                for p in self.labels_dir.glob("*.nii.gz")
                if not p.stem.endswith("_provenance")
            )
        self.case_ids: list[str] = list(case_ids)
        if not self.case_ids:
            raise RuntimeError(f"No cases found under {self.labels_dir}")

        self.patch_size = tuple(patch_size)
        self.samples_per_case = int(samples_per_case)
        self.foreground_ratio = float(foreground_ratio)
        self.normalize = normalize
        self.max_fg_sample = int(max_fg_sample)

        # Foreground voxel cache: case_id -> (N, 3) int array, or None
        # if that case has no foreground voxels (synthetic-empty or
        # weirdly labelled). Populated lazily to keep __init__ fast.
        self._fg_cache: dict[str, np.ndarray | None] = {}

        print(f"NnUNetRawDataset: {len(self.case_ids)} cases, "
              f"patch {self.patch_size}, {self.samples_per_case} samples/case")

    # ------------------------------------------------------------------
    # Standard Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.case_ids) * self.samples_per_case

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        case_idx = idx // self.samples_per_case
        case_id = self.case_ids[case_idx]

        image, mask = self._load_case(case_id)   # (4, H, W, D), (H, W, D)
        image_patch, mask_patch = self._sample_patch(image, mask, case_id)

        # (4, pH, pW, pD), (1, pH, pW, pD)
        img_t = torch.from_numpy(np.ascontiguousarray(image_patch)).float()
        msk_t = torch.from_numpy(
            np.ascontiguousarray(mask_patch)[None, ...]
        ).float()
        return img_t, msk_t, case_id

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def _load_case(self, case_id: str) -> tuple[np.ndarray, np.ndarray]:
        """Load 4-channel image + binary mask for one case."""
        channels: list[np.ndarray] = []
        for mod in ("0000", "0001", "0002", "0003"):
            p = self.images_dir / f"{case_id}_{mod}.nii.gz"
            if not p.exists():
                raise FileNotFoundError(
                    f"Missing modality {mod} for case {case_id}: {p}")
            arr = np.asarray(
                nib.load(str(p)).dataobj, dtype=np.float32
            )
            if self.normalize:
                mean = arr.mean()
                std = arr.std()
                if std > 0:
                    arr = (arr - mean) / std
            channels.append(arr)
        image = np.stack(channels, axis=0)  # (4, H, W, D)

        mp = self.labels_dir / f"{case_id}.nii.gz"
        mask = np.asarray(
            nib.load(str(mp)).dataobj, dtype=np.float32
        )
        mask = (mask > 0).astype(np.float32)

        return image, mask

    # ------------------------------------------------------------------
    # Patch sampling
    # ------------------------------------------------------------------

    def _sample_patch(
        self, image: np.ndarray, mask: np.ndarray, case_id: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample one (patch, mask) pair with foreground bias."""
        _, H, W, D = image.shape
        pH, pW, pD = self.patch_size

        # If the volume is smaller than the patch on any axis, pad with
        # zeros up to the patch size. (Common on older data with
        # non-standard shapes.)
        pad_h = max(0, pH - H)
        pad_w = max(0, pW - W)
        pad_d = max(0, pD - D)
        if pad_h or pad_w or pad_d:
            image = np.pad(
                image,
                ((0, 0), (0, pad_h), (0, pad_w), (0, pad_d)),
                mode="constant",
            )
            mask = np.pad(
                mask, ((0, pad_h), (0, pad_w), (0, pad_d)), mode="constant",
            )
            _, H, W, D = image.shape

        rng = np.random.default_rng()

        fg_voxels = self._get_fg_voxels(case_id, mask)
        use_fg = (fg_voxels is not None
                  and rng.random() < self.foreground_ratio)

        if use_fg:
            centre = fg_voxels[rng.integers(len(fg_voxels))]
            z = int(np.clip(centre[0] - pH // 2, 0, H - pH))
            y = int(np.clip(centre[1] - pW // 2, 0, W - pW))
            x = int(np.clip(centre[2] - pD // 2, 0, D - pD))
        else:
            z = int(rng.integers(0, max(1, H - pH + 1)))
            y = int(rng.integers(0, max(1, W - pW + 1)))
            x = int(rng.integers(0, max(1, D - pD + 1)))

        image_patch = image[:, z:z + pH, y:y + pW, x:x + pD]
        mask_patch = mask[z:z + pH, y:y + pW, x:x + pD]
        return image_patch, mask_patch

    def _get_fg_voxels(
        self, case_id: str, mask: np.ndarray,
    ) -> np.ndarray | None:
        """Lazily cache foreground voxel indices for a case."""
        if case_id in self._fg_cache:
            return self._fg_cache[case_id]
        fg = np.argwhere(mask > 0.5)
        if len(fg) == 0:
            self._fg_cache[case_id] = None
            return None
        if len(fg) > self.max_fg_sample:
            idx = np.random.default_rng(0).choice(
                len(fg), size=self.max_fg_sample, replace=False)
            fg = fg[idx]
        self._fg_cache[case_id] = fg.astype(np.int64)
        return self._fg_cache[case_id]


# ---------------------------------------------------------------------------
# Split helpers
# ---------------------------------------------------------------------------

def load_nnunet_split(
    splits_json: str | Path, fold: int = 0,
) -> tuple[list[str], list[str]]:
    """Load the nnU-Net canonical 5-fold split for a dataset.

    Returns (train_case_ids, val_case_ids) for the requested fold.
    Using nnU-Net's split makes SwinUNETR's val Dice directly
    comparable to nnU-Net 3D's on the same held-out cases.
    """
    with open(splits_json) as f:
        splits = json.load(f)
    if fold >= len(splits):
        raise ValueError(f"Fold {fold} out of range for {len(splits)}-fold split")
    return list(splits[fold]["train"]), list(splits[fold]["val"])
