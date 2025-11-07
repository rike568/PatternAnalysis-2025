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


# ---------------------------------------------
# Core building blocks from the diagram
# ---------------------------------------------


class _ContextModule(nn.Module):
    """
    Corresponds to 'context module' in the diagram.
    Two 3x3 Conv2d layers, each followed by BatchNorm and ReLU.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _DownsamplingModule(nn.Module):
    """
    Corresponds to '3x3x3 stride 2 convolution' in the diagram.
    For 2D, this is a 3x3 Conv2d with stride 2.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class _UpsamplingModule(nn.Module):
    """
    Corresponds to 'upsampling module' in the diagram.
    ConvTranspose2d with kernel_size=2, stride=2.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)


class _LocalizationModule(nn.Module):
    """
    Corresponds to 'localization module' in the diagram.
    Two 3x3 Conv2d layers, each followed by ReLU. BatchNorm is typically included.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
