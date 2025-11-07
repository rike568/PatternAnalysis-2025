# dataset.py
# HipMRI_Study_open 2D dataloader:
# - auto-locates ./HipMRI_Study_open next to this file
# - pairs image slices (case_...) with masks (seg_...) via a canonical key
# - loads 2D Nifti slices using z-score normalization
# - applies simple train-time geometric augmentation
# - returns PyTorch DataLoaders for train/val/test

from __future__ import annotations
import random
from pathlib import Path
from typing import List, Tuple, Dict

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


class RandomAugment2D:
    """Apply identical random flips/rotation to image and mask (train only)."""

    def __init__(
        self, max_rot_deg: float = 10.0, p_hflip: float = 0.5, p_vflip: float = 0.5
    ):
        self.max_rot_deg = max_rot_deg
        self.p_hflip = p_hflip
        self.p_vflip = p_vflip

    def __call__(self, img: torch.Tensor, mask: torch.Tensor):
        # Add channel dim to mask for transforms [H,W] -> [1,H,W]
        mask = mask[None, ...]

        # Horizontal flip
        if random.random() < self.p_hflip:
            img = F.hflip(img)
            mask = F.hflip(mask)
        # Vertical flip
        if random.random() < self.p_vflip:
            img = F.vflip(img)
            mask = F.vflip(mask)
        # Small random rotation; bilinear for image, nearest for mask (to preserve labels)
        if self.max_rot_deg > 0:
            angle = random.uniform(-self.max_rot_deg, self.max_rot_deg)
            img = F.rotate(
                img, angle, interpolation=F.InterpolationMode.BILINEAR, fill=0
            )
            # --- FIX 1: Corrected typo InterpolATIONMode -> InterpolationMode ---
            mask = F.rotate(
                mask, angle, interpolation=F.InterpolationMode.NEAREST, fill=0
            )

        # Remove channel dim from mask [1,H,W] -> [H,W]
        return img, mask.squeeze(0)


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
        self.augment = (
            RandomAugment2D(max_rot_deg=max_rot_deg)
            if train_augment and split == "train"
            else None
        )

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


def make_loaders(
    data_root: Path = DEFAULT_DATA_ROOT,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
    # normalize_meanstd: Optional[Tuple[float, float]] = None, # Removed
):
    """
    Convenience factory: returns (train_loader, val_loader, test_loader).
    - Shuffles only the training loader.
    - Leaves val/test deterministic.
    """
    train_ds = HipMRI2DSegDataset(data_root, "train", train_augment=True)
    val_ds = HipMRI2DSegDataset(data_root, "validate")
    test_ds = HipMRI2DSegDataset(data_root, "test")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # --- MODIFIED: Main block for shape and MASK VALUE verification ---
    print(f"[HipMRI] Using data_root: {DEFAULT_DATA_ROOT}")
    print("Running dataset shape verification...")

    all_shapes = set()
    all_mask_values = set()  # Check mask values

    def check_dataset_shapes(name: str, dataset: HipMRI2DSegDataset):
        print(f"\nChecking dataset: {name} ({len(dataset)} samples)")
        shapes = set()
        mask_vals = set()

        # Wrap dataset iteration with tqdm for a progress bar
        for i in tqdm(range(len(dataset)), desc=f"Scanning {name}"):
            try:
                # Use __getitem__ to load and process the data
                sample = dataset[i]
                img_shape = tuple(sample["image"].shape[1:])  # (H, W)
                mask_shape = tuple(sample["mask"].shape)  # (H, W)

                # Check mask values AFTER processing
                mask_vals.update(torch.unique(sample["mask"]).numpy().tolist())

                current_shape = img_shape

                if img_shape != mask_shape:
                    print(
                        f"  WARNING: Mismatch! Img {dataset.pairs[i][0].name} is {img_shape}, Mask {dataset.pairs[i][1].name} is {mask_shape}"
                    )

                shapes.add(current_shape)

            except Exception as e:
                print(f"  ERROR loading sample {i} ({dataset.pairs[i][0].name}): {e}")

        print(f"-> Found unique (H, W) shapes for {name}: {shapes}")
        print(f"-> Found unique mask values for {name}: {sorted(list(mask_vals))}")
        return shapes, mask_vals

    # Instantiate datasets
    try:
        train_ds = HipMRI2DSegDataset(DEFAULT_DATA_ROOT, "train")
        val_ds = HipMRI2DSegDataset(DEFAULT_DATA_ROOT, "validate")
        test_ds = HipMRI2DSegDataset(DEFAULT_DATA_ROOT, "test")

        train_shapes, train_mask_vals = check_dataset_shapes("train", train_ds)
        val_shapes, val_mask_vals = check_dataset_shapes("validate", val_ds)
        test_shapes, test_mask_vals = check_dataset_shapes("test", test_ds)

        all_shapes.update(train_shapes)
        all_shapes.update(val_shapes)
        all_shapes.update(test_shapes)

        all_mask_values.update(train_mask_vals)
        all_mask_values.update(val_mask_vals)
        all_mask_values.update(test_mask_vals)

        print("\n========================================")
        print(f"All unique (H, W) shapes found: {all_shapes}")
        print(f"All unique mask values found: {sorted(list(all_mask_values))}")

        # MODIFIED: Changed confirmation message to 256x128
        if len(all_shapes) == 1 and (256, 128) in all_shapes:
            print("Confirmation: All images are 256x128.")
        else:
            print("WARNING: Not all images are 256x128 or multiple sizes found.")

        # --- FIX 3: Updated check for 6 classes ---
        if all(v in [0, 1, 2, 3, 4, 5] for v in all_mask_values):
            print("Confirmation: All mask values are valid (0, 1, 2, 3, 4, 5).")
        else:
            print("WARNING: Invalid mask values found! Check the list above.")
        print("========================================")

    except Exception as e:
        print(f"\nFailed to initialize dataset. Check paths and folder names.")
        print(f"Error: {e}")
