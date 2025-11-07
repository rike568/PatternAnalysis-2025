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
    """Set RNG seeds for Python, NumPy, and PyTorch."""
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
    """Move a dict of tensors (e.g., from dataset) to device."""
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def labels_to_onehot(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Convert integer labels [B,H,W] -> one-hot [B,C,H,W].
    """
    # y expected long dtype; ensure safety.
    y = y.long()
    return F.one_hot(y, num_classes=num_classes).permute(0, 3, 1, 2).float()


# ---------------------------
# Dice metrics / Dice loss
# ---------------------------


@torch.no_grad()
def dice_per_class_from_logits(
    logits: torch.Tensor, y_true: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """
    Dice per class given logits and integer labels.
    - logits: [B,C,H,W]
    - y_true: [B,H,W] with class ids 0..C-1
    Returns: [C] dice scores.
    """
    num_classes = logits.size(1)
    probs = torch.softmax(logits, dim=1)  # [B,C,H,W]
    y_1h = labels_to_onehot(y_true, num_classes)  # [B,C,H,W]
    num = 2.0 * (probs * y_1h).sum(dim=(0, 2, 3))
    den = (probs * probs).sum(dim=(0, 2, 3)) + (y_1h * y_1h).sum(dim=(0, 2, 3)) + eps
    return num / den


def dice_loss_from_logits(
    logits: torch.Tensor, y_true_1h: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """
    Soft Dice loss using one-hot target.
    - logits: [B,C,H,W]
    - y_true_1h: [B,C,H,W]
    """
    probs = torch.softmax(logits, dim=1)
    num = 2.0 * (probs * y_true_1h).sum(dim=(0, 2, 3))
    den = (
        (probs * probs).sum(dim=(0, 2, 3))
        + (y_true_1h * y_true_1h).sum(dim=(0, 2, 3))
        + eps
    )
    dice = num / den
    return 1.0 - dice.mean()


# ---------------------------
# Combined loss (CE + Dice)
# ---------------------------


class CEDiceLoss(torch.nn.Module):
    """
    CE + Dice combined loss.
    - alpha_ce: weight for CrossEntropy
    - alpha_dice: weight for Dice (targets one-hot internally)
    """

    def __init__(
        self,
        num_classes: int,
        alpha_ce: float = 0.5,
        alpha_dice: float = 0.5,
        ignore_index: int | None = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.alpha_ce = alpha_ce
        self.alpha_dice = alpha_dice
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # y_true expected as class ids [B,H,W]
        if self.ignore_index is None:
            loss_ce = F.cross_entropy(logits, y_true)
        else:
            loss_ce = F.cross_entropy(logits, y_true, ignore_index=self.ignore_index)
        y_1h = labels_to_onehot(y_true, num_classes=self.num_classes)
        loss_dice = dice_loss_from_logits(logits, y_1h)
        return self.alpha_ce * loss_ce + self.alpha_dice * loss_dice


# ---------------------------
# Running meters
# ---------------------------


@dataclass
class AvgMeter:
    total: float = 0.0
    count: int = 0

    def update(self, val: float, n: int = 1) -> None:
        self.total += float(val) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.total / max(self.count, 1)
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
    """Save model state dict (and optional optimizer/epoch/extra) to path."""
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
    """Load state dicts into model/optimizer; returns checkpoint dict."""
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt