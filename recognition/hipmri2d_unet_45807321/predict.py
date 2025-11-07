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
NUM_CLASSES = 6
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
    """
    Ensures that a directory exists, creating it if necessary.

    Args:
        p: The pathlib.Path of the directory to check/create.
    """
    p.mkdir(parents=True, exist_ok=True)


def colorize(mask_ids: np.ndarray) -> np.ndarray:
    """
    Maps a 2D array of class IDs to a 3D RGB color mask.

    Args:
        mask_ids: A 2D numpy array of integer class labels [H, W].

    Returns:
        A 3D numpy array (RGB image) [H, W, 3] of type uint8.
    """
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
    Converts a normalized image tensor to a uint8 grayscale image.

    Handles z-score normalized tensors by mapping a rough [-2, 2] range
    to [0, 1] before scaling to [0, 255].

    Args:
        x: A [1, H, W] or [H, W] image tensor, typically z-score normalized.

    Returns:
        A 2D numpy array (grayscale image) [H, W] of type uint8.
    """
    x = x.detach().cpu().float()
    if x.ndim == 3 and x.size(0) == 1:
        x = x[0]  # Squeeze channel dim

    # Check if tensor is z-score normalized (values outside [0, 1])
    x_min, x_max = float(x.min()), float(x.max())
    if x_min < -0.1 or x_max > 1.1:
        # Assumes z-score norm, map roughly -2..2 to 0..1
        x = (x + 2.0) / 4.0

    x = torch.clamp(x, 0.0, 1.0)  # Clamp to [0, 1] range
    return (x.numpy() * 255.0).astype(np.uint8)


def overlay(
    img_gray_u8: np.ndarray, mask_rgb: np.ndarray, alpha: float = 0.5
) -> np.ndarray:
    """
    Overlays a color RGB mask onto a grayscale image.

    Args:
        img_gray_u8: The base grayscale image [H, W] as uint8.
        mask_rgb: The color mask [H, W, 3] as uint8.
        alpha: The opacity of the mask (0.0 = transparent, 1.0 = opaque).

    Returns:
        A 3D numpy array (RGB image) [H, W, 3] of the blended overlay.
    """
    # Convert grayscale to 3-channel RGB
    img_rgb = np.stack([img_gray_u8] * 3, axis=-1)
    # Blend
    out = (img_rgb * (1 - alpha) + mask_rgb * alpha).astype(np.uint8)
    return out


# ---------------------------
# Main
# ---------------------------


@torch.no_grad()
def main() -> None:
    """
    Main function to run inference and visualization.

    - Loads the test dataset.
    - Loads the best trained model checkpoint.
    - Calculates and prints per-class Dice scores for the entire test set.
    - Saves N_VIS sample visualizations (input, gt, pred, overlay) to disk.
    - Creates a preview grid of the saved overlays.
    """
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

    # Load best checkpoint
    ckpt_path = CKPT_BEST if CKPT_BEST.exists() else CKPT_LAST
    if ckpt_path.exists():
        load_checkpoint(
            ckpt_path.as_posix(), model, optimizer=None, map_location=device
        )
        print(f"Loaded checkpoint: {ckpt_path}")
    else:
        print("⚠️ No checkpoint found — running with random-initialized weights.")

    model.eval()

    # Metrics over entire test set
    dice_sum = torch.zeros(NUM_CLASSES, device=device)
    n_batches = 0

    _ensure_dir(PRED_DIR)

    vis_count = 0
    saved_paths: List[Path] = []

    print(f"Running evaluation and saving {N_VIS} samples to {PRED_DIR}...")
    for batch_idx, batch in enumerate(test_loader):
        batch = to_device(batch, device)
        x, y_ids = batch["image"], batch["mask"]

        # Get model prediction
        logits = model(x)

        # --- Metric Calculation ---
        dice_c = dice_per_class_from_logits(logits, y_ids)
        dice_sum += dice_c
        n_batches += 1

        # --- Visualization Saving ---
        if vis_count < N_VIS:
            take = min(
                N_VIS - vis_count, x.size(0)
            )  # Num samples to take from this batch
            for i in range(take):
                # Convert tensors to numpy images
                img_u8 = tensor_to_uint8_img(x[i])
                pred_ids = logits[i].argmax(dim=0).cpu().numpy().astype(np.int32)
                gt_ids = y_ids[i].cpu().numpy().astype(np.int32)

                # Colorize masks
                pred_rgb = colorize(pred_ids)
                gt_rgb = colorize(gt_ids)

                # <--- MODIFIED SECTION START --->

                # --- Create combined plot (like your example) ---

                # NOTE: Your example image shows 3 panels.
                # If you also want the 'overlay' panel, change 'n_cols=3' to 'n_cols=4'
                # and uncomment the 4th panel (axes[3]) plotting lines below.

                # over_rgb = overlay(img_u8, pred_rgb, alpha=0.45) # Uncomment for 4 panels

                n_cols = 3  # Change to 4 if you want the overlay
                fig, axes = plt.subplots(
                    nrows=1,
                    ncols=n_cols,
                    figsize=(n_cols * 5, 5.5),  # 5x5 inch per panel + title space
                )

                # Ensure 'axes' is always an array for easy indexing
                if n_cols == 1:
                    axes = np.array([axes])
                else:
                    axes = axes.flat

                # Panel 1: Original Image
                axes[0].imshow(img_u8, cmap="gray")
                axes[0].set_title("Original Image")

                # Panel 2: Ground Truth Mask
                axes[1].imshow(gt_rgb)
                axes[1].set_title("Ground Truth Mask")

                # Panel 3: Predicted Mask
                axes[2].imshow(pred_rgb)
                axes[2].set_title("Predicted Mask")

                # # Panel 4: Overlay (Optional - uncomment lines below)
                # if n_cols >= 4:
                #   axes[3].imshow(over_rgb)
                #   axes[3].set_title("Overlay")

                # --- Clean up and Save ---
                for ax in axes:
                    ax.axis("off")

                fig.tight_layout()

                base = Path(f"sample_{batch_idx:03d}_{i:02d}")
                save_path = PRED_DIR / f"{base}_combined.png"

                fig.savefig(save_path, dpi=150, bbox_inches="tight")
                plt.close(fig)  # IMPORTANT: close fig in a loop

                # Add the path of the new combined image for the preview grid
                saved_paths.append(save_path)

                # <--- MODIFIED SECTION END --->

                vis_count += 1

    # --- Report final metrics ---
    print("\n==> Test Metrics:")
    dice_mean_c = (dice_sum / max(n_batches, 1)).cpu().numpy()
    dice_mean = float(dice_mean_c.mean())
    print(
        "Per-class Dice:",
        "  ".join([f"C{c}:{dice_mean_c[c]:.3f}" for c in range(NUM_CLASSES)]),
    )
    print(f"Mean Dice: {dice_mean:.3f}")

    # --- Create preview grid ---
    if saved_paths:
        print(f"\nCreating preview grid at {PRED_DIR / 'preview_overlays.png'}...")

        # <--- MODIFIED SECTION START --->
        # Adjust preview grid to better fit the new wide images
        n_rows = min(N_VIS, 8)

        # Set a fixed width (e.g., 15 inches) and calculate row height based on
        # the 3:1 aspect ratio of the new combined images (n_cols * 5, 5.5)
        aspect_ratio = (n_cols * 5) / 5.5
        fig_width = 15.0
        row_height = fig_width / aspect_ratio

        fig, axes = plt.subplots(
            nrows=n_rows, ncols=1, figsize=(fig_width, row_height * n_rows)
        )
        # <--- MODIFIED SECTION END --->

        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])

        for ax, p in zip(axes.flat, saved_paths[:n_rows]):
            ax.imshow(plt.imread(p))
            ax.set_title(p.name)
            ax.axis("off")

        preview_path = PRED_DIR / "preview_overlays.png"
        fig.tight_layout()
        fig.savefig(preview_path, dpi=150)
        plt.close(fig)
        print(f"Saved {len(saved_paths)} combined samples to: {PRED_DIR}")
        print(f"Preview grid saved: {preview_path}")


if __name__ == "__main__":
    main()
