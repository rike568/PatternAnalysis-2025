# utils.py
# Small utilities for training: one-hot, Dice metrics/loss, meters, seeding, checkpoints.

from __future__ import annotations
import os
import random
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------
# Reproducibility
# ---------------------------


def set_seed(seed: int = 42) -> None:
    """
    Sets the random seed for Python, NumPy, and PyTorch for reproducible results.

    Args:
        seed: The integer seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # no-op if CUDA not available
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------
# Tensor helpers
# ---------------------------


def to_device(
    batch: Dict[str, torch.Tensor], device: torch.device
) -> Dict[str, torch.Tensor]:
    """
    Moves a dictionary of tensors to the specified device (e.g., 'cuda' or 'cpu').

    Args:
        batch: A dictionary where keys are strings and values are tensors
               (or other data types which will be ignored).
        device: The target torch.device to move tensors to.

    Returns:
        A new dictionary with all tensor values moved to the specified device.
    """
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v  # Keep non-tensor items (like paths) as-is
    return out


def labels_to_onehot(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Converts a batch of integer label masks to a one-hot encoded format.

    Args:
        y: A tensor of integer labels with shape [B, H, W].
        num_classes: The total number of classes (C).

    Returns:
        A one-hot encoded tensor with shape [B, C, H, W].
    """
    # y expected long dtype; ensure safety.
    y = y.long()
    # F.one_hot creates [B, H, W, C], permute to [B, C, H, W]
    return F.one_hot(y, num_classes=num_classes).permute(0, 3, 1, 2).float()


# ---------------------------
# Dice metrics / Dice loss
# ---------------------------


@torch.no_grad()
def dice_per_class_from_logits(
    logits: torch.Tensor, y_true: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """
    Calculates the Dice score for each class from raw logits.

    Args:
        logits: The model's raw output tensor [B, C, H, W].
        y_true: The ground truth integer labels [B, H, W].
        eps: A small epsilon value to prevent division by zero.

    Returns:
        A 1D tensor of shape [C] containing the Dice score for each class.
    """
    num_classes = logits.size(1)
    probs = torch.softmax(logits, dim=1)  # [B, C, H, W]
    y_1h = labels_to_onehot(y_true, num_classes)  # [B, C, H, W]

    # Sum over batch (0) and spatial dims (2, 3)
    num = 2.0 * (probs * y_1h).sum(dim=(0, 2, 3))
    den = (probs * probs).sum(dim=(0, 2, 3)) + (y_1h * y_1h).sum(dim=(0, 2, 3)) + eps

    return num / den


def dice_loss_from_logits(
    logits: torch.Tensor, y_true_1h: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """
    Calculates the soft Dice loss from raw logits.
    This function expects the target to already be one-hot encoded.

    Args:
        logits: The model's raw output tensor [B, C, H, W].
        y_true_1h: The ground truth one-hot encoded labels [B, C, H, W].
        eps: A small epsilon value to prevent division by zero.

    Returns:
        A scalar tensor representing the mean Dice loss (1.0 - mean_dice).
    """
    probs = torch.softmax(logits, dim=1)

    # Sum over batch (0) and spatial dims (2, 3)
    num = 2.0 * (probs * y_true_1h).sum(dim=(0, 2, 3))
    den = (
        (probs * probs).sum(dim=(0, 2, 3))
        + (y_true_1h * y_true_1h).sum(dim=(0, 2, 3))
        + eps
    )

    dice_per_class = num / den
    return 1.0 - dice_per_class.mean()  # Mean loss across classes


# ---------------------------
# Combined loss (CE + Dice)
# ---------------------------


class CEDiceLoss(torch.nn.Module):
    """
    A combined loss function that is a weighted sum of Cross-Entropy
    and Dice loss.

    This is useful for segmentation tasks to balance pixel-wise accuracy (CE)
    with spatial overlap (Dice).
    """

    def __init__(
        self,
        num_classes: int,
        alpha_ce: float = 0.5,
        alpha_dice: float = 0.5,
        ignore_index: int | None = None,
    ):
        """
        Initializes the CEDiceLoss module.

        Args:
            num_classes: The number of classes for one-hot encoding.
            alpha_ce: The weight for the Cross-Entropy loss component.
            alpha_dice: The weight for the Dice loss component.
            ignore_index: An optional class index to ignore in CE loss.
        """
        super().__init__()
        self.num_classes = num_classes
        self.alpha_ce = alpha_ce
        self.alpha_dice = alpha_dice
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Calculates the combined loss.

        Args:
            logits: The model's raw output tensor [B, C, H, W].
            y_true: The ground truth integer labels [B, H, W].

        Returns:
            A scalar tensor representing the final combined loss.
        """
        # Calculate Cross-Entropy loss
        if self.ignore_index is None:
            loss_ce = F.cross_entropy(logits, y_true)
        else:
            loss_ce = F.cross_entropy(logits, y_true, ignore_index=self.ignore_index)

        # Calculate Dice loss
        y_1h = labels_to_onehot(y_true, num_classes=self.num_classes)
        loss_dice = dice_loss_from_logits(logits, y_1h)

        # Combine losses
        return self.alpha_ce * loss_ce + self.alpha_dice * loss_dice


# ---------------------------
# Running meters
# ---------------------------


@dataclass
class AvgMeter:
    """
    A simple data class to track the running average of a metric.
    """

    total: float = 0.0
    count: int = 0

    def update(self, val: float, n: int = 1) -> None:
        """
        Updates the meter with a new value and count.

        Args:
            val: The value to add (e.g., loss for a batch).
            n: The number of items this value represents (e.g., batch size).
        """
        self.total += float(val) * n
        self.count += n

    @property
    def avg(self) -> float:
        """Calculates the current average."""
        return self.total / max(self.count, 1)  # Avoid division by zero


# ---------------------------
# Checkpoint helpers
# ---------------------------


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int | None = None,
    extra: Dict | None = None,
) -> None:
    """
    Saves a model checkpoint to a .pt file.

    Args:
        path: The full path to save the checkpoint file (e.g., 'outputs/best.pt').
        model: The model to save (state_dict will be saved).
        optimizer: An optional optimizer to save (state_dict will be saved).
        epoch: An optional epoch number to save.
        extra: An optional dictionary of extra data to save (e.g., val_loss).
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    state = {"model": model.state_dict()}
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if epoch is not None:
        state["epoch"] = epoch
    if extra is not None:
        state["extra"] = extra
    torch.save(state, path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> Dict:
    """
    Loads a model checkpoint from a .pt file.

    Args:
        path: The full path to the checkpoint file.
        model: The model instance to load the state_dict into.
        optimizer: An optional optimizer instance to load the state_dict into.
        map_location: The device to load the checkpoint onto.

    Returns:
        The full checkpoint dictionary that was loaded.
    """
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt
