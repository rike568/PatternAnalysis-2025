# predict.py
# Inference + visualisation for HipMRI 2D segmentation (Improved U-Net)

from __future__ import annotations
import os
from pathlib import Path
from typing import Tuple, List

import torch
import matplotlib.pyplot as plt
import numpy as np

from dataset import make_loaders
from modules import create_model
from utils import (
    # oasis_mask_to_class_ids, # No longer needed
    dice_per_class_from_logits,
    set_seed,
    to_device,
    load_checkpoint,
)

# ---------------------------
# Config
# ---------------------------
SEED = 42
NUM_CLASSES = 6  # MODIFIED: Changed from 4 to 6
IN_CHANNELS = 1

OUTDIR = Path("./outputs")
PRED_DIR = OUTDIR / "predictions"
CKPT_BEST = OUTDIR / "best.pt"
CKPT_LAST = OUTDIR / "last.pt"

N_VIS = 8  # number of samples to visualise/save


# ---------------------------
# Small helpers
# ---------------------------


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def colorize(mask_ids: np.ndarray) -> np.ndarray:
    """
    Colorize class-id mask to RGB for saving/plotting.
    Simple palette: background + 5 tissues.
    """
    # MODIFIED: Added 2 new colors for classes 4 and 5
    palette = np.array(
        [
            [0, 0, 0],  # 0: background - black
            [0, 114, 189],  # 1: blue
            [217, 83, 25],  # 2: orange
            [237, 177, 32],  # 3: yellow
            [126, 47, 142],  # 4: purple (NEW)
            [119, 172, 48],  # 5: green (NEW)
        ],
        dtype=np.uint8,
    )
    mask_ids = np.clip(mask_ids, 0, len(palette) - 1)
    return palette[mask_ids]


def tensor_to_uint8_img(x: torch.Tensor) -> np.ndarray:
    """
    x: [1,H,W] float in roughly [-1,1] or [0,1]
    Convert to uint8 grayscale [H,W].
    """
    x = x.detach().cpu().float()
    if x.ndim == 3 and x.size(0) == 1:
        x = x[0]
    # Try to map from z-score [-1,1] to [0,1] if necessary
    x_min, x_max = float(x.min()), float(x.max())
    if x_min < -0.1 or x_max > 1.1:  # Broadened range for z-score
        # Assumes z-score norm, map roughly -2..2 to 0..1
        x = (x + 2.0) / 4.0
    x = torch.clamp(x, 0.0, 1.0)
    return (x.numpy() * 255.0).astype(np.uint8)


def overlay(
    img_gray_u8: np.ndarray, mask_rgb: np.ndarray, alpha: float = 0.5
) -> np.ndarray:
    """
    Overlay RGB mask on grayscale image.
    """
    img_rgb = np.stack([img_gray_u8] * 3, axis=-1)
    out = (img_rgb * (1 - alpha) + mask_rgb * alpha).astype(np.uint8)
    return out


# ---------------------------
# Main
# ---------------------------


@torch.no_grad()
def main() -> None:
    print("==> HipMRI 2D — Inference & Visualisation")
    # (Implementation to be added)


if __name__ == "__main__":
    main()
