# dataset.py
# OASIS 2D dataloader:
# - auto-locates ./OASIS next to this file
# - pairs image slices (case_...) with masks (seg_...) via a canonical key
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
from PIL import Image

# ---------------------------
# Defaults / config
# ---------------------------
THIS_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = THIS_DIR / "OASIS"  # expects six OASIS subfolders under this
DEFAULT_IMG_SIZE = None  # images are already 256x256; no resize needed
DEFAULT_BATCH_SIZE = 8
DEFAULT_NUM_WORKERS = max(os.cpu_count() - 1, 1) if os.cpu_count() else 4
DEFAULT_NORMALIZE_MEANSTD = (0.5, 0.5)  # normalize to roughly N(0,1): (x-0.5)/0.5

# ---------------------------
# Simple helpers
# ---------------------------


def _canonical_key(p: Path) -> str:
    """
    Standardize filenames so image/mask match on the same key.
    Example:
      'case_001_slice_0.png' -> '001_slice_0'
      'seg_001-slice_0.png'  -> '001_slice_0'
    """
    name = p.stem.lower()  # drop extension, lowercase
    if name.startswith("case_"):
        name = name[len("case_") :]
    elif name.startswith("seg_"):
        name = name[len("seg_") :]
    name = name.replace("-", "_")  # unify separators
    return name


def _pil_grayscale(path: Path) -> Image.Image:
    """Load an image and ensure single-channel grayscale (mode 'L')."""
    img = Image.open(path)
    if img.mode != "L":
        img = img.convert("L")
    return img


def _to_tensor01(img_pil: Image.Image) -> torch.Tensor:
    """
    Convert grayscale PIL image to FloatTensor in [0,1] with channel dim.
    Output shape: [1, H, W]
    """
    arr = np.asarray(img_pil, dtype=np.float32)[None, ...]  # add channel axis
    mn, mx = arr.min(), arr.max()
    arr = (arr - mn) / (mx - mn) if mx > mn else arr * 0.0  # safe min-max scale
    return torch.from_numpy(arr)


def _mask_to_tensor(mask_pil: Image.Image) -> torch.Tensor:
    """
    Convert mask to LongTensor of label IDs.
    OASIS masks typically use intensities {0,85,170,255}.
    """
    arr = np.asarray(mask_pil, dtype=np.int64)
    return torch.from_numpy(arr)


class RandomAugment2D:
    """Apply identical random flips/rotation to image and mask (train only)."""

    def __init__(
        self, max_rot_deg: float = 10.0, p_hflip: float = 0.5, p_vflip: float = 0.5
    ):
        self.max_rot_deg = max_rot_deg
        self.p_hflip = p_hflip
        self.p_vflip = p_vflip

    def __call__(self, img: Image.Image, mask: Image.Image):
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
            mask = F.rotate(
                mask, angle, interpolation=F.InterpolationMode.NEAREST, fill=0
            )
        return img, mask


class Oasis2DSegDataset(Dataset):
    """
    Dataset for OASIS 2D slices.

    Expected folder layout inside ./OASIS:
        keras_png_slices_train/
        keras_png_slices_validate/
        keras_png_slices_test/
        keras_png_slices_seg_train/
        keras_png_slices_seg_validate/
        keras_png_slices_seg_test/
    """

    def __init__(
        self,
        data_root: Path = DEFAULT_DATA_ROOT,
        split: str = "train",
        normalize_meanstd: Optional[Tuple[float, float]] = DEFAULT_NORMALIZE_MEANSTD,
        train_augment: bool = False,
        max_rot_deg: float = 10.0,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        assert split in {"train", "validate", "test"}  # enforce valid split names
        self.split = split
        self.normalize_meanstd = normalize_meanstd

        # Locate image/mask directories by split
        img_dir = (
            self.data_root
            / f"keras_png_slices_{'validate' if split=='validate' else split}"
        )
        seg_dir = (
            self.data_root
            / f"keras_png_slices_seg_{'validate' if split=='validate' else split}"
        )
        if not (img_dir.exists() and seg_dir.exists()):
            raise FileNotFoundError(
                f"Missing expected OASIS folders:\n{img_dir}\n{seg_dir}"
            )

        # List files and build a mask index keyed by canonical names
        images = sorted([p for p in img_dir.iterdir() if p.is_file()])
        masks = sorted([p for p in seg_dir.iterdir() if p.is_file()])
        mask_index: Dict[str, Path] = {_canonical_key(m): m for m in masks}

        # Pair image with mask via canonical key
        pairs: List[Tuple[Path, Path]] = []
        missing: List[str] = []
        for ip in images:
            mp = mask_index.get(_canonical_key(ip))
            if mp is not None:
                pairs.append((ip, mp))
            else:
                missing.append(ip.name)

        if not pairs:
            # Give a hint if nothing matched at all
            raise RuntimeError(
                "No image/mask pairs found. Check prefixes or directory names.\n"
                f"Example image: {images[0].name if images else 'None'}\n"
                f"Example mask : {masks[0].name if masks else 'None'}"
            )
        if missing:
            # Non-fatal: some images missing masks; show a few to aid debugging
            print(
                f"[OASIS] {len(missing)} images had no mask match (showing first 5): {missing[:5]}"
            )

        self.pairs = pairs
        # Only apply augmentation on the training split
        self.augment = (
            RandomAugment2D(max_rot_deg=max_rot_deg)
            if train_augment and split == "train"
            else None
        )

    def __len__(self):
        """Number of paired samples."""
        return len(self.pairs)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Optional mean/std normalization; pass None to skip."""
        if self.normalize_meanstd is None:
            return x
        mean, std = self.normalize_meanstd
        return (x - mean) / (std + 1e-8)

    def __getitem__(self, idx: int):
        """Load one (image, mask) pair; apply augments and preprocessing."""
        img_path, mask_path = self.pairs[idx]
        img = _pil_grayscale(img_path)
        mask = _pil_grayscale(mask_path)
        if self.augment:
            img, mask = self.augment(img, mask)
        img_t = _to_tensor01(img)  # FloatTensor [1,H,W] in [0,1]
        mask_t = _mask_to_tensor(mask)  # LongTensor  [H,W] with label IDs
        img_t = self._normalize(img_t)  # Optionally normalize to ~N(0,1)
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
    normalize_meanstd: Optional[Tuple[float, float]] = DEFAULT_NORMALIZE_MEANSTD,
):
    """
    Convenience factory: returns (train_loader, val_loader, test_loader).
    - Shuffles only the training loader.
    - Leaves val/test deterministic.
    """
    train_ds = Oasis2DSegDataset(
        data_root, "train", normalize_meanstd, train_augment=True
    )
    val_ds = Oasis2DSegDataset(data_root, "validate", normalize_meanstd)
    test_ds = Oasis2DSegDataset(data_root, "test", normalize_meanstd)

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
    # Sanity check: build loaders and print one batch summary
    print(f"[OASIS] Using data_root: {DEFAULT_DATA_ROOT}")
    tl, vl, te = make_loaders()
    batch = next(iter(tl))
    x, y = batch["image"], batch["mask"]
    print(
        f"Train batch image shape: {tuple(x.shape)}, dtype={x.dtype}, range=({x.min():.3f},{x.max():.3f})"
    )
    print(
        f"Train batch mask  shape: {tuple(y.shape)}, dtype={y.dtype}, labels(sample0)={torch.unique(y[0]).tolist()}"
    )
