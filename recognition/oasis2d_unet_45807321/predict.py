# predict.py
# Inference + visualisation for OASIS 2D segmentation (Improved U-Net)

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
    oasis_mask_to_class_ids,
    dice_per_class_from_logits,
    set_seed,
    to_device,
    load_checkpoint,
)

# ---------------------------
# Config
# ---------------------------
SEED = 42
NUM_CLASSES = 4
IN_CHANNELS = 1
P_DROP = 0.0

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


# ---------------------------
# Main
# ---------------------------


@torch.no_grad()
def main() -> None:
    print("==> OASIS 2D — Inference & Visualisation")
    # (Implementation to be added)


if __name__ == "__main__":
    main()