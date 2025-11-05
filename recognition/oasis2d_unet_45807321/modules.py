# modules.py
# Improved U-Net with forward pass

from __future__ import annotations
import torch
import torch.nn as nn

__all__ = ["conv_block", "up_block", "ImprovedUNet"]

def conv_block(in_ch: int, out_ch: int, p_drop: float = 0.0) -> nn.Sequential:
    """
    Two 3x3 convs with BN + ReLU, optional Dropout2d.
    """
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Dropout2d(p_drop) if p_drop > 0 else nn.Identity(),
    )

def up_block(in_ch: int, out_ch: int) -> nn.ConvTranspose2d:
    """
    2x upsampling via transposed conv.
    """
    return nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)

class ImprovedUNet(nn.Module):
    """
    Improved U-Net (2D):
    - Encoder: 4 levels with BN + ReLU
    - Bottleneck
    - Decoder: transpose conv upsampling + skip connections
    - Head: 1x1 conv logits (no softmax)
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 4, base: int = 64, p_drop: float = 0.0):
        super().__init__()
        self.enc1 = conv_block(in_channels, base, p_drop); self.pool1 = nn.MaxPool2d(2)
        self.enc2 = conv_block(base, base*2, p_drop);       self.pool2 = nn.MaxPool2d(2)
        self.enc3 = conv_block(base*2, base*4, p_drop);     self.pool3 = nn.MaxPool2d(2)
        self.enc4 = conv_block(base*4, base*8, p_drop);     self.pool4 = nn.MaxPool2d(2)
        self.bott = conv_block(base*8, base*16, p_drop)
        self.up4  = up_block(base*16, base*8); self.dec4 = conv_block(base*16, base*8, p_drop)
        self.up3  = up_block(base*8, base*4);  self.dec3 = conv_block(base*8, base*4, p_drop)
        self.up2  = up_block(base*4, base*2);  self.dec2 = conv_block(base*4, base*2, p_drop)
        self.up1  = up_block(base*2, base);    self.dec1 = conv_block(base*2, base, p_drop)
        self.head = nn.Conv2d(base, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)            # 1x
        e2 = self.enc2(self.pool1(e1))  # 1/2
        e3 = self.enc3(self.pool2(e2))  # 1/4
        e4 = self.enc4(self.pool3(e3))  # 1/8
        b  = self.bott(self.pool4(e4))  # 1/16

        # Decoder with skip connections
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.head(d1)  # logits [B,C,H,W]