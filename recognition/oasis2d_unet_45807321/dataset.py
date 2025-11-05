# dataset.py
# Scaffold + canonical key for pairing case_/seg_ filenames

from __future__ import annotations
import os
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from PIL import Image

# ---------------------------
# Defaults
# ---------------------------
THIS_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = THIS_DIR / "OASIS"
DEFAULT_IMG_SIZE = None
DEFAULT_BATCH_SIZE = 8
DEFAULT_NUM_WORKERS = max(os.cpu_count() - 1, 1) if os.cpu_count() else 4
DEFAULT_NORMALIZE_MEANSTD = (0.5, 0.5)

# ---------------------------
# Simple helpers
# ---------------------------

def _canonical_key(p: Path) -> str:
    """
    Map 'case_001_slice_0.png' and 'seg_001-slice_0.png' -> '001_slice_0'
    """
    name = p.stem.lower()
    if name.startswith("case_"):
        name = name[len("case_"):]
    elif name.startswith("seg_"):
        name = name[len("seg_"):]
    name = name.replace("-", "_")
    return name

def _pil_grayscale(path: Path) -> Image.Image:
    img = Image.open(path)
    if img.mode != "L":
        img = img.convert("L")
    return img

def _to_tensor01(img_pil: Image.Image) -> torch.Tensor:
    arr = np.asarray(img_pil, dtype=np.float32)[None, ...]  # [1,H,W]
    mn, mx = arr.min(), arr.max()
    arr = (arr - mn) / (mx - mn) if mx > mn else arr * 0.0
    return torch.from_numpy(arr)

def _mask_to_tensor(mask_pil: Image.Image) -> torch.Tensor:
    arr = np.asarray(mask_pil, dtype=np.int64)
    return torch.from_numpy(arr)