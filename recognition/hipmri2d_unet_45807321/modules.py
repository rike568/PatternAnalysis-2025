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


class _SegmentationLayer(nn.Module):
    """
    Corresponds to 'segmentation layer' in the diagram.
    A single 1x1 Conv2d.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ---------------------------------------------
# The U-Net Model
# ---------------------------------------------


class ImprovedUNet(nn.Module):  # Keeping the name ImprovedUNet for compatibility
    """
    U-Net-style encoder–decoder with skip connections, based on the provided diagram.

    Architecture:
      - Encoder: Uses 'context modules' and '3x3 stride 2 convolutions' for downsampling.
      - Bottleneck: A 'context module'.
      - Decoder: Uses 'upsampling modules' and 'localization modules' with skip concatenations.
      - Segmentation Layers: 1x1 convolutions at the end of each decoder stage and final output.

    Notes:
      - Designed for 1-channel 256x128 inputs.
      - Output logits are returned without activation; apply softmax in loss/metrics if needed.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 6):
        super().__init__()

        # Encoder Path
        self.context1 = _ContextModule(in_channels, 16)
        self.down1 = _DownsamplingModule(16, 32)

        self.context2 = _ContextModule(32, 32)
        self.down2 = _DownsamplingModule(32, 64)

        self.context3 = _ContextModule(64, 64)
        self.down3 = _DownsamplingModule(64, 128)

        self.context4 = _ContextModule(128, 128)
        self.down4 = _DownsamplingModule(128, 256)

        # Bottleneck (deepest context module)
        self.bottleneck = _ContextModule(256, 256)

        # Decoder Path
        self.up4 = _UpsamplingModule(256, 128)
        self.loc4 = _LocalizationModule(
            128 + 128, 128
        )  # Concatenates upsampled with context4 output
        self.seg4 = _SegmentationLayer(128, num_classes)

        self.up3 = _UpsamplingModule(128, 64)
        self.loc3 = _LocalizationModule(
            64 + 64, 64
        )  # Concatenates upsampled with context3 output
        self.seg3 = _SegmentationLayer(64, num_classes)

        self.up2 = _UpsamplingModule(64, 32)
        self.loc2 = _LocalizationModule(
            32 + 32, 32
        )  # Concatenates upsampled with context2 output
        self.seg2 = _SegmentationLayer(32, num_classes)

        self.up1 = _UpsamplingModule(32, 16)
        self.loc1 = _LocalizationModule(
            16 + 16, 16
        )  # Concatenates upsampled with context1 output

        # Final output segmentation layer
        self.final_seg_layer = _SegmentationLayer(16, num_classes)

        # Apply Kaiming initialization
        self.apply(init_kaiming_normal_)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass  # To be implemented
