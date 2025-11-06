# modules.py
# Improved U-Net (2D) for HipMRI slices (256x256, 1-channel).
# Returns per-pixel logits [B, C, H, W] (no softmax in the model).

from __future__ import annotations
import torch
import torch.nn as nn

__all__ = [
    "conv_block",
    "up_block",
    "ImprovedUNet",
    "init_kaiming_normal_",
    "create_model",
    "count_params",
]


def conv_block(in_ch: int, out_ch: int, p_drop: float = 0.0) -> nn.Sequential:
    """
    Two 3×3 Conv2d layers with BatchNorm + ReLU.
    Optionally applies Dropout2d at the end (p_drop>0).
    """
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Dropout2d(p=p_drop) if p_drop > 0 else nn.Identity(),
    )


def up_block(in_ch: int, out_ch: int) -> nn.ConvTranspose2d:
    """2× upsampling via transposed convolution (kernel=2, stride=2)."""
    return nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)


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


class ImprovedUNet(nn.Module):
    """
    U-Net-style encoder–decoder with skip connections.
    MODIFIED: Uses fixed channel sizes (32, 64, 128, 256, 512).

    Architecture:
      - Encoder: 4 levels (ConvBlock ×2 per level) with MaxPool downsampling
      - Bottleneck: ConvBlock at deepest level
      - Decoder: transposed-conv upsampling + skip concat + ConvBlock
      - Head: 1×1 Conv2d to produce logits for num_classes

    Notes:
      - Designed for 1-channel 256x256 inputs (divisible by 16 → no size drift).
      - Output logits are returned without activation; apply softmax in loss/metrics if needed.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 4,
        p_drop: float = 0.0,
    ):
        super().__init__()

        # Encoder - MODIFIED to use fixed channels
        self.enc1 = conv_block(in_channels, 32, p_drop)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = conv_block(32, 64, p_drop)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = conv_block(64, 128, p_drop)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = conv_block(128, 256, p_drop)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck - MODIFIED to use fixed channels
        self.bott = conv_block(256, 512, p_drop)

        # Decoder - MODIFIED to use fixed channels
        self.up4 = up_block(512, 256)
        self.dec4 = conv_block(512, 256, p_drop) # 256 (from up4) + 256 (from enc4)

        self.up3 = up_block(256, 128)
        self.dec3 = conv_block(256, 128, p_drop) # 128 (from up3) + 128 (from enc3)

        self.up2 = up_block(128, 64)
        self.dec2 = conv_block(128, 64, p_drop) # 64 (from up2) + 64 (from enc2)

        self.up1 = up_block(64, 32)
        self.dec1 = conv_block(64, 32, p_drop) # 32 (from up1) + 32 (from enc1)

        # Per-pixel logits - MODIFIED
        self.head = nn.Conv2d(32, num_classes, kernel_size=1)

        # Weight init
        self.apply(init_kaiming_normal_)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)  # H,W
        e2 = self.enc2(self.pool1(e1))  # H/2,W/2
        e3 = self.enc3(self.pool2(e2))  # H/4,W/4
        e4 = self.enc4(self.pool3(e3))  # H/8,W/8
        b = self.bott(self.pool4(e4))  # H/16,W/16  <- Bottleneck shape changes

        # Decoder + skip connections
        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.head(d1)  # logits [B, C, H, W]


def create_model(
    in_channels: int = 1, num_classes: int = 4, p_drop: float = 0.0
) -> ImprovedUNet: # MODIFIED: Removed 'base' argument
    """Factory for quick construction (useful in train.py)."""
    return ImprovedUNet(
        in_channels=in_channels, num_classes=num_classes, p_drop=p_drop
    )


def count_params(model: nn.Module) -> int:
    """Return the number of trainable parameters (for logs/README)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick sanity check
    net = create_model(in_channels=1, num_classes=4, p_drop=0.1) 

    # MODIFIED: Test with 256x256 input
    x = torch.randn(2, 1, 256, 256)
    y = net(x)

    # MODIFIED: Expected output shape
    print("Output:", tuple(y.shape))  # expected: (2, 6, 256, 256)
    print("Params:", count_params(net))