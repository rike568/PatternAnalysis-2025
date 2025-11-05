# Scaffold for OASIS 2D dataloader (structure + imports + defaults)

from __future__ import annotations
import os
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from PIL import Image

# ---------------------------
# Defaults
# ---------------------------
THIS_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = THIS_DIR / "OASIS"
DEFAULT_IMG_SIZE = None  # images are already 256x256
DEFAULT_BATCH_SIZE = 8
DEFAULT_NUM_WORKERS = max(os.cpu_count() - 1, 1) if os.cpu_count() else 4
DEFAULT_NORMALIZE_MEANSTD = (0.5, 0.5)