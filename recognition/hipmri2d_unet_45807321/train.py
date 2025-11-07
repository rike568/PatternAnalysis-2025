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
        # MODIFIED: Get masks directly
        x, y_ids = batch["image"], batch["mask"]
        # y_ids = oasis_mask_to_class_ids(y_raw) # No longer needed

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
    # --- ADDED: Argument Parser ---
    parser = argparse.ArgumentParser(description="HipMRI 2D U-Net Training")
    parser.add_argument(
        "--seed", type=int, default=SEED, help=f"Random seed (default: {SEED})"
    )
    parser.add_argument(
        "--lr", type=float, default=LR, help=f"Learning rate (default: {LR})"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=WEIGHT_DECAY,
        help=f"Adam weight decay (default: {WEIGHT_DECAY})",
    )
    parser.add_argument(
        "--grad_clip_norm",
        type=float,
        default=GRAD_CLIP_NORM,
        help=f"Gradient clipping norm, 0 to disable (default: {GRAD_CLIP_NORM})",
    )
    args = parser.parse_args()
    # ---------------------------------

    # MODIFIED: Changed print statement
    print("==> HipMRI 2D — Improved U-Net training")

    # --- ADDED: Print settings ---
    print("==> Settings:")
    print(f"  Seed: {args.seed}")
    print(f"  LR: {args.lr}")
    print(f"  Weight Decay: {args.weight_decay}")
    print(f"  Grad Clip Norm: {args.grad_clip_norm}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch Size: {DEFAULT_BATCH_SIZE}")
    print(f"  AMP: {AMP}")
    print(f"  Output Dir: {OUTDIR.as_posix()}")
    print("-" * 30)
    # -----------------------------

    # Repro
    set_seed(args.seed)  # <-- MODIFIED: Use arg

    # Device & AMP
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = torch.amp.GradScaler("cuda") if (AMP and device.type == "cuda") else None
    print(f"Device: {device} | AMP: {scaler is not None}")

    # Data
    train_loader, val_loader, test_loader = make_loaders(
        batch_size=DEFAULT_BATCH_SIZE,  # from dataset.py
        # num_workers=1,  # uncomment on Rangpur to avoid worker warnings
    )
    print(
        f"Train/Val/Test batches: {len(train_loader)}/{len(val_loader)}/{len(test_loader)}"
    )

    # Model
    model = create_model(
        in_channels=IN_CHANNELS,
        num_classes=NUM_CLASSES,
    ).to(device)
    print(f"Model params: {count_params(model):,}")

    # (Training loop to be added)


if __name__ == "__main__":
    main()
