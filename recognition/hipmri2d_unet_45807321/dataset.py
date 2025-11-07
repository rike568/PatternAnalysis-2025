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
# DEFAULT_NORMALIZE_MEANSTD = (0.5, 0.5) # No longer needed, using z-score

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
    pass # To be implemented