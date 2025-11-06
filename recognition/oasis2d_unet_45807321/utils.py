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