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
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data: only need test loader for predictions
    _, _, test_loader = make_loaders()

    # Model
    model = create_model(
        in_channels=IN_CHANNELS,
        num_classes=NUM_CLASSES,
    ).to(device)
    ckpt_path = CKPT_BEST if CKPT_BEST.exists() else CKPT_LAST
    if ckpt_path.exists():
        load_checkpoint(
            ckpt_path.as_posix(), model, optimizer=None, map_location=device
        )
        print(f"Loaded checkpoint: {ckpt_path}")
    else:
        print(
            "⚠️ No checkpoint found — running with random-initialized weights (metrics will be poor)."
        )

    model.eval()

    # Metrics over entire test set
    dice_sum = torch.zeros(NUM_CLASSES, device=device)
    n_batches = 0

    _ensure_dir(PRED_DIR)

    # (Visualization logic to be added here)
    saved_paths: List[Path] = []  # Added for later commit

    for batch_idx, batch in enumerate(test_loader):
        batch = to_device(batch, device)

        # MODIFIED: Get masks directly. Shape is [B,1,256,128]
        x, y_ids = (
            batch["image"],
            batch["mask"],
        )  # y_ids: [B,256,128] with {0,1,2,3,4,5}

        # y_ids = oasis_mask_to_class_ids(y_raw)  # No longer needed

        logits = model(x)
        dice_c = dice_per_class_from_logits(logits, y_ids)  # [C]
        dice_sum += dice_c
        n_batches += 1

        # (Visualization logic to be added here)

    # Report metrics
    dice_mean_c = (dice_sum / max(n_batches, 1)).detach().cpu().numpy()
    dice_mean = float(dice_mean_c.mean())
    print(
        "Per-class Dice:",
        "  ".join([f"C{c}:{dice_mean_c[c]:.3f}" for c in range(NUM_CLASSES)]),
    )
    print(f"Mean Dice: {dice_mean:.3f}")

    # (Preview grid logic to be added here)


if __name__ == "__main__":
    main()
