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
        # Encoder
        # Level 1
        x_c1 = self.context1(x)  # [B, 16, H, W]
        x_d1 = self.down1(x_c1)  # [B, 32, H/2, W/2]

        # Level 2
        x_c2 = self.context2(x_d1)  # [B, 32, H/2, W/2]
        x_d2 = self.down2(x_c2)  # [B, 64, H/4, W/4]

        # Level 3
        x_c3 = self.context3(x_d2)  # [B, 64, H/4, W/4]
        x_d3 = self.down3(x_c3)  # [B, 128, H/8, W/8]

        # Level 4
        x_c4 = self.context4(x_d3)  # [B, 128, H/8, W/8]
        x_d4 = self.down4(x_c4)  # [B, 256, H/16, W/16]

        # Bottleneck
        x_bottleneck = self.bottleneck(x_d4)  # [B, 256, H/16, W/16]

        # Decoder
        # Level 4 (decoding from bottleneck)
        x_up4 = self.up4(x_bottleneck)  # [B, 128, H/8, W/8]
        x_cat4 = torch.cat([x_up4, x_c4], dim=1)  # [B, 256, H/8, W/8]
        x_loc4 = self.loc4(x_cat4)  # [B, 128, H/8, W/8]
        s4 = self.seg4(x_loc4)  # [B, num_classes, H/8, W/8]

        # Level 3
        x_up3 = self.up3(x_loc4)  # [B, 64, H/4, W/4]
        x_cat3 = torch.cat([x_up3, x_c3], dim=1)  # [B, 128, H/4, W/4]
        x_loc3 = self.loc3(x_cat3)  # [B, 64, H/4, W/4]
        s3 = self.seg3(x_loc3)  # [B, num_classes, H/4, W/4]

        # Level 2
        x_up2 = self.up2(x_loc3)  # [B, 32, H/2, W/2]
        x_cat2 = torch.cat([x_up2, x_c2], dim=1)  # [B, 64, H/2, W/2]
        x_loc2 = self.loc2(x_cat2)  # [B, 32, H/2, W/2]
        s2 = self.seg2(x_loc2)  # [B, num_classes, H/2, W/2]

        # Level 1
        x_up1 = self.up1(x_loc2)  # [B, 16, H, W]
        x_cat1 = torch.cat([x_up1, x_c1], dim=1)  # [B, 32, H, W]
        x_loc1 = self.loc1(x_cat1)  # [B, 16, H, W]

        # Final segmentation layer
        s1 = self.final_seg_layer(x_loc1)  # [B, num_classes, H, W]

        # Element-wise sum of segmentation layers (after upscaling s4, s3, s2 to s1's size)
        # Note: F.interpolate is used for upscaling
        output = (
            s1
            + F.interpolate(s2, scale_factor=2, mode="bilinear", align_corners=False)
            + F.interpolate(s3, scale_factor=4, mode="bilinear", align_corners=False)
            + F.interpolate(s4, scale_factor=8, mode="bilinear", align_corners=False)
        )

        return output


# -----------------------------------------------------------------
# --- Factory and Parameter Counter (for compatibility) ---
# -----------------------------------------------------------------


def create_model(
    in_channels: int = 1,
    num_classes: int = 6,  # Removed base and p_drop as they are not used by this architecture
) -> ImprovedUNet:
    """Factory for quick construction (useful in train.py)."""
    return ImprovedUNet(in_channels=in_channels, num_classes=num_classes)


def count_params(model: nn.Module) -> int:
    """Return the number of trainable parameters (for logs/README)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
