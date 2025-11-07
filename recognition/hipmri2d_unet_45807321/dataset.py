# dataset.py
# HipMRI_Study_open 2D dataloader:
# - auto-locates ./HipMRI_Study_open next to this file
# - pairs image slices (case_...) with masks (seg_...) via a canonical key
# - loads 2D Nifti slices using z-score normalization
# - applies simple train-time geometric augmentation
# - returns PyTorch DataLoaders for train/val/test

from __future__ import annotations
import os
import random
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
import nibabel as nib  # Added for Nifti loading
from tqdm import tqdm  # Import tqdm for progress bar

# ---------------------------
# Defaults / config
# ---------------------------
THIS_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = THIS_DIR / "HipMRI_Study_open"
DEFAULT_IMG_SIZE = None
DEFAULT_BATCH_SIZE = 64
DEFAULT_NUM_WORKERS = 1

# ---------------------------
# Simple helpers
# ---------------------------


def _canonical_key(p: Path) -> str:
    """
    Standardize filenames so image/mask match on the same key.
    Example:
      'case_001_slice_0.nii.gz' -> '001_slice_0'
      'seg_001-slice_0.nii.gz'  -> '001_slice_0'
    """
    name = p.stem.lower()
    if ".nii" in name:  # Handle double extensions like .nii.gz
        name = Path(name).stem

    if name.startswith("case_"):
        name = name[len("case_") :]
    elif name.startswith("seg_"):
        name = name[len("seg_") :]
    name = name.replace("-", "_")  # unify separators
    return name


class HipMRI2DSegDataset(Dataset):
    """
    Dataset for HipMRI 2D Nifti slices.

    Expected folder layout inside ./HipMRI_Study_open:
        keras_slices_train/
        keras_slices_validate/
        keras_slices_test/
        keras_slices_seg_train/
        keras_slices_seg_validate/
        keras_slices_seg_test/
    """

    def __init__(
        self,
        data_root: Path = DEFAULT_DATA_ROOT,
        split: str = "train",
        # normalize_meanstd: Optional[Tuple[float, float]] = None, # Removed
        train_augment: bool = False,
        max_rot_deg: float = 10.0,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        assert split in {"train", "validate", "test"}  # enforce valid split names
        self.split = split
        # self.normalize_meanstd = normalize_meanstd # Removed

        # Locate image/mask directories by split
        img_dir = (
            self.data_root
            / f"keras_slices_{'validate' if split=='validate' else split}"
        )
        seg_dir = (
            self.data_root
            / f"keras_slices_seg_{'validate' if split=='validate' else split}"
        )
        if not (img_dir.exists() and seg_dir.exists()):
            raise FileNotFoundError(
                f"Missing expected HipMRI_Study_open folders:\n{img_dir}\n{seg_dir}"
            )

        # List files and build a mask index keyed by canonical names
        images = sorted(
            [p for p in img_dir.iterdir() if p.is_file() and ".nii" in p.name]
        )
        masks = sorted(
            [p for p in seg_dir.iterdir() if p.is_file() and ".nii" in p.name]
        )
        mask_index: Dict[str, Path] = {_canonical_key(m): m for m in masks}

        # Pair image with mask via canonical key
        pairs: List[Tuple[Path, Path]] = []
        missing: List[str] = []
        for ip in images:
            key = _canonical_key(ip)
            mp = mask_index.get(key)
            if mp is not None:
                pairs.append((ip, mp))
            else:
                missing.append(f"{ip.name} (key: {key})")

        if not pairs:
            raise RuntimeError(
                "No image/mask pairs found. Check prefixes or directory names.\n"
                f"Example image: {images[0].name if images else 'None'}\n"
                f"Example mask : {masks[0].name if masks else 'None'}"
            )
        if missing:
            print(
                f"[HipMRI] {len(missing)} images had no mask match (showing first 5): {missing[:5]}"
            )

        self.pairs = pairs
        self.augment = None  # Placeholder for now

    def __len__(self):
        """Number of paired samples."""
        return len(self.pairs)

    # --- _normalize method removed ---

    def __getitem__(self, idx: int):
        """Load one (image, mask) pair; apply augments and preprocessing."""
        img_path, mask_path = self.pairs[idx]

        # --- MODIFIED: Load Nifti files using logic from load_data_2D ---

        # Load image
        img = nib.load(img_path).get_fdata(caching="unchanged")
        if len(img.shape) == 3:
            img = img[:, :, 0]  # Take first slice
        img = img.astype(np.float32)

        # Load mask
        mask = nib.load(mask_path).get_fdata(caching="unchanged")
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]  # Take first slice

        # Apply per-image z-score normalization (from load_data_2D)
        mean = img.mean()
        std = img.std()
        img = (img - mean) / (std + 1e-8)  # Add epsilon for safety

        # Convert to Tensors
        # Add channel dim to image: [H,W] -> [1,H,W]
        img_t = torch.from_numpy(img)[None, ...]
        # Mask should be LongTensor: [H,W]
        mask_t = torch.from_numpy(mask.astype(np.int64))

        # --- Data already contains labels [0, 1, 2, 3, 4, 5] ---

        # --- MODIFIED: Robust center-crop to (256, 128) ---
        # This handles (256, 144), etc.
        target_h, target_w = 256, 128
        _, current_h, current_w = img_t.shape

        if current_h == target_h and current_w > target_w:
            # Image is correct height but too wide. Center-crop width.
            crop_pixels = current_w - target_w
            start_w = crop_pixels // 2
            end_w = start_w + target_w

            img_t = img_t[:, :, start_w:end_w]
            mask_t = mask_t[:, start_w:end_w]

        # Apply augmentation (now on Tensors)
        if self.augment:
            img_t, mask_t = self.augment(img_t, mask_t)

        # self._normalize(img_t) call removed

        return {
            "image": img_t,
            "mask": mask_t,
            "image_path": str(img_path),
            "mask_path": str(mask_path),
        }
