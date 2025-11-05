# modules.py
# Core blocks for Improved U-Net

from __future__ import annotations
import torch
import torch.nn as nn

__all__ = ["conv_block", "up_block", "ImprovedUNet"]

def conv_block(in_ch: int, out_ch: int, p_drop: float = 0.0) -> nn.Sequential:
    """
    Two 3x3 convs with BN + ReLU, optional Dropout2d.
    """
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),

        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),

        nn.Dropout2d(p_drop) if p_drop > 0 else nn.Identity(),
    )

def up_block(in_ch: int, out_ch: int) -> nn.ConvTranspose2d:
    """
    2x upsampling via transposed conv.
    """
    return nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)