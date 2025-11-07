# modules.py
# U-Net 2D model based on the provided diagram for HipMRI slices (256x128).
# Returns per-pixel logits [B, C, H, W] (no softmax in the model).

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F  # Added for potential future use, though not strictly in this model for now

__all__ = [
    "ImprovedUNet",  # Renamed to keep consistent with train.py, but it's the new arch
    "create_model",
    "count_params",
    "init_kaiming_normal_",  # Retaining for good practice
]


# Helper function for weight initialization
def init_kaiming_normal_(m: nn.Module) -> None:
    """
    He (Kaiming) init for conv/convtranspose; BatchNorm gamma=1, beta=0.
    Call with model.apply(init_kaiming_normal_).
    """
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if getattr(m, "bias", None) is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
