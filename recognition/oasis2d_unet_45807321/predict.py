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


def colorize(mask_ids: np.ndarray) -> np.ndarray:
    """
    Colorize class-id mask to RGB for saving/plotting.
    Simple palette: background + 3 tissues.
    """
    palette = np.array(
        [
            [0, 0, 0],  # 0: background - black
            [0, 114, 189],  # 1: blue
            [217, 83, 25],  # 2: orange
            [237, 177, 32],  # 3: yellow
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
    # Try to map from [-1,1] to [0,1] if necessary
    x_min, x_max = float(x.min()), float(x.max())
    if x_min < 0.0 or x_max > 1.0:
        x = (x + 1.0) / 2.0
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
    print("==> OASIS 2D — Inference & Visualisation")
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data: only need test loader for predictions
    _, _, test_loader = make_loaders()

    # Model
    model = create_model(
        in_channels=IN_CHANNELS,
        num_classes=NUM_CLASSES,
        p_drop=P_DROP,
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

    # Gather a few samples for visualisation
    vis_count = 0
    saved_paths: List[Path] = []  # <-- ADD THIS

    for batch_idx, batch in enumerate(test_loader):
        batch = to_device(batch, device)
        x, y_raw = (
            batch["image"],
            batch["mask"],
        )  # x: [B,1,256,256], y_raw: [B,256,256] with {0,85,170,255}
        y_ids = oasis_mask_to_class_ids(y_raw)  # -> {0,1,2,3}

        logits = model(x)
        dice_c = dice_per_class_from_logits(logits, y_ids)  # [C]
        dice_sum += dice_c
        n_batches += 1

        # --- ADD THIS BLOCK ---
        # Visualise/save a few samples from the first batches
        if vis_count < N_VIS:
            # How many to take from this batch
            take = min(N_VIS - vis_count, x.size(0))
            for i in range(take):
                img_u8 = tensor_to_uint8_img(x[i])  # [H,W] uint8
                pred_ids = (
                    logits[i].argmax(dim=0).detach().cpu().numpy().astype(np.int32)
                )  # [H,W]
                gt_ids = y_ids[i].detach().cpu().numpy().astype(np.int32)

                pred_rgb = colorize(pred_ids)  # [H,W,3]
                gt_rgb = colorize(gt_ids)
                over_rgb = overlay(img_u8, pred_rgb, alpha=0.45)

                # Save individual panels
                base = Path(f"sample_{batch_idx:03d}_{i:02d}")
                paths = {
                    "input": PRED_DIR / f"{base}_input.png",
                    "gt": PRED_DIR / f"{base}_gt.png",
                    "pred": PRED_DIR / f"{base}_pred.png",
                    "over": PRED_DIR / f"{base}_overlay.png",
                }
                plt.imsave(paths["input"], img_u8, cmap="gray")
                plt.imsave(paths["gt"], gt_rgb)
                plt.imsave(paths["pred"], pred_rgb)
                plt.imsave(paths["over"], over_rgb)
                saved_paths.append(paths["over"])
                vis_count += 1

        # Early exit if we already have enough visualisations
        if vis_count >= N_VIS:
            # still continue metric accumulation for full test set
            pass
        # --- END OF BLOCK ---

    # Report metrics
    dice_mean_c = (dice_sum / max(n_batches, 1)).detach().cpu().numpy()
    dice_mean = float(dice_mean_c.mean())
    print(
        "Per-class Dice:",
        "  ".join([f"C{c}:{dice_mean_c[c]:.3f}" for c in range(NUM_CLASSES)]),
    )
    print(f"Mean Dice: {dice_mean:.3f}")

    # Quick preview grid (uses last N_VIS saved overlays + GT/pred/input for the last batch portion)
    if saved_paths:
        fig, axes = plt.subplots(
            nrows=min(N_VIS, 8), ncols=1, figsize=(6, 3 * min(N_VIS, 8))
        )
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])
        for ax, p in zip(axes, saved_paths[: len(axes)]):
            ax.imshow(plt.imread(p))
            ax.set_title(p.name)
            ax.axis("off")
        preview_path = PRED_DIR / "preview_overlays.png"
        fig.tight_layout()
        fig.savefig(preview_path, dpi=150)
        plt.close(fig)
        print(f"Saved {len(saved_paths)} sample overlays to: {PRED_DIR}")
        print(f"Preview grid: {preview_path}")


if __name__ == "__main__":
    main()
