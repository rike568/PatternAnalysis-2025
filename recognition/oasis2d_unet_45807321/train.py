# train.py
# End-to-end training script for OASIS 2D segmentation with Improved U-Net.
from __future__ import annotations
import os
from pathlib import Path
from typing import Tuple, Dict, List

import torch
import torch.nn as nn
import torch.optim as optim

# --- NEW: headless plotting + csv logging ---
import matplotlib

matplotlib.use("Agg")  # safe for clusters / no display
import matplotlib.pyplot as plt
import csv

from dataset import make_loaders, DEFAULT_BATCH_SIZE
from modules import create_model, count_params
from utils import (
    set_seed,
    to_device,
    oasis_mask_to_class_ids,
    CEDiceLoss,
    dice_per_class_from_logits,
    AvgMeter,
    save_checkpoint,
)

# ---------------------------
# Config (edit here if needed)
# ---------------------------
SEED = 42
NUM_CLASSES = 4
IN_CHANNELS = 1
BASE_CHANNELS = 64
P_DROP = 0.10

EPOCHS = 10
LR = 1e-3
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
) -> float:
    model.train()
    loss_meter = AvgMeter()

    for step, batch in enumerate(loader, 1):
        batch = to_device(batch, device)
        x, y_raw = batch["image"], batch["mask"]
        y_ids = oasis_mask_to_class_ids(y_raw)  # [B,H,W] in {0,1,2,3}

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                logits = model(x)  # [B,C,H,W]
                loss = criterion(logits, y_ids)  # CE + Dice
            scaler.scale(loss).backward()
            if GRAD_CLIP_NORM > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = criterion(logits, y_ids)
            loss.backward()
            if GRAD_CLIP_NORM > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()

        loss_meter.update(loss.item(), n=x.size(0))

        if step % LOG_EVERY == 0:
            print(
                f"Epoch {epoch:03d} | step {step:05d}/{len(loader):05d} | loss {loss_meter.avg:.4f}"
            )

    return loss_meter.avg

@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, torch.Tensor]:
    model.eval()
    loss_meter = AvgMeter()
    dice_sum = None
    n_batches = 0

    for batch in loader:
        batch = to_device(batch, device)
        x, y_raw = batch["image"], batch["mask"]
        y_ids = oasis_mask_to_class_ids(y_raw)

        logits = model(x)
        loss = criterion(logits, y_ids)
        loss_meter.update(loss.item(), n=x.size(0))

        dice_c = dice_per_class_from_logits(logits, y_ids)  # [C]
        dice_sum = dice_c if dice_sum is None else (dice_sum + dice_c)
        n_batches += 1

    dice_mean_c = dice_sum / max(n_batches, 1)  # [C]
    return loss_meter.avg, dice_mean_c  # val_loss, per-class dice

# ---------------------------
# Plotting / Logging helpers
# ---------------------------


def write_history_csv(rows: List[Dict], path: Path) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def plot_curves(history: List[Dict], png_path: Path, num_classes: int) -> None:
    # history: list of dicts with keys epoch, train_loss, val_loss, dice_c0..c{C-1}
    epochs = [h["epoch"] for h in history]
    tr = [h["train_loss"] for h in history]
    vl = [h["val_loss"] for h in history]
    dice_per_c = []
    for c in range(num_classes):
        dice_per_c.append([h[f"dice_c{c}"] for h in history])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss curves
    axes[0].plot(epochs, tr, label="train_loss")
    axes[0].plot(epochs, vl, label="val_loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curves")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Dice curves per class
    for c in range(num_classes):
        axes[1].plot(epochs, dice_per_c[c], label=f"Dice C{c}")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Dice")
    axes[1].set_title("Per-class Dice (Validation)")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(png_path, dpi=160)
    plt.close(fig)

# ---------------------------
# Main
# ---------------------------


def main() -> None:
    print("==> OASIS 2D — Improved U-Net training")
    # (Implementation to be added)


if __name__ == "__main__":
    main()