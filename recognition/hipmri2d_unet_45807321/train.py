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
# Train / Val loops
# ---------------------------


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.amp.GradScaler | None,
    epoch: int,
    grad_clip_norm: float,  # <-- MODIFIED: Added arg
) -> float:
    model.train()
    loss_meter = AvgMeter()

    for step, batch in enumerate(loader, 1):
        batch = to_device(batch, device)
        # MODIFIED: Get masks directly
        x, y_ids = batch["image"], batch["mask"]
        # y_ids = oasis_mask_to_class_ids(y_raw)  # No longer needed

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                logits = model(x)  # [B,C,H,W]
                loss = criterion(logits, y_ids)  # CE + Dice
            scaler.scale(loss).backward()
            if grad_clip_norm > 0:  # <-- MODIFIED: Use arg
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), grad_clip_norm  # <-- MODIFIED: Use arg
                )
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = criterion(logits, y_ids)
            loss.backward()
            if grad_clip_norm > 0:  # <-- MODIFIED: Use arg
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), grad_clip_norm  # <-- MODIFIED: Use arg
                )
            optimizer.step()

        loss_meter.update(loss.item(), n=x.size(0))

        if step % LOG_EVERY == 0:
            print(
                f"Epoch {epoch:03d} | step {step:05d}/{len(loader):05d} | loss {loss_meter.avg:.4f}"
            )

    return loss_meter.avg


# ---------------------------
# Main
# ---------------------------


def main() -> None:
    print("==> HipMRI 2D — Improved U-Net training")
    # (Implementation to be added)


if __name__ == "__main__":
    main()
