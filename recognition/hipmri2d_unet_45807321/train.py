# train.py
# End-to-end training script for HipMRI 2D segmentation with Improved U-Net.
from __future__ import annotations
import os
from pathlib import Path
from typing import Tuple, Dict, List
import argparse  # <-- ADDED
import csv

import torch
import torch.nn as nn
import torch.optim as optim

# --- NEW: headless plotting + csv logging ---
import matplotlib

matplotlib.use("Agg")  # safe for clusters / no display
import matplotlib.pyplot as plt

from dataset import make_loaders, DEFAULT_BATCH_SIZE
from modules import create_model, count_params
from utils import (
    set_seed,
    to_device,
    # oasis_mask_to_class_ids, # No longer needed
    CEDiceLoss,
    dice_per_class_from_logits,
    AvgMeter,
    save_checkpoint,
)

# ---------------------------
# Config (Default values)
# ---------------------------
SEED = 42
NUM_CLASSES = 6
IN_CHANNELS = 1

EPOCHS = 20
LR = 0.0005
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 1.0

AMP = True  # mixed precision
OUTDIR = Path("./outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)
CKPT_BEST = OUTDIR / "best.pt"
CKPT_LAST = OUTDIR / "last.pt"
HIST_CSV = OUTDIR / "history.csv"
CURVES_PNG = OUTDIR / "curves.png"

LOG_EVERY = 50  # steps


# ---------------------------
# Main
# ---------------------------


def main() -> None:
    print("==> HipMRI 2D — Improved U-Net training")
    # (Implementation to be added)


if __name__ == "__main__":
    main()