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