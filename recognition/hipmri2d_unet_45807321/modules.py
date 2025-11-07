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
    Applies He (Kaiming) initialization to Conv2d and ConvTranspose2d layers
    and initializes BatchNorm2d layers.

    Call with `model.apply(init_kaiming_normal_)`.

    Args:
        m: The module to initialize.
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
    The 'context module' from the diagram.

    A block consisting of two sequential 3x3 convolutions, each followed
    by BatchNorm and ReLU. This block maintains the input resolution.
    """

    def __init__(self, in_channels: int, out_channels: int):
        """
        Initializes the context module.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
        """
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
        """
        Runs the forward pass for the context module.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        return self.block(x)


class _DownsamplingModule(nn.Module):
    """
    The 'downsampling module' from the diagram (3x3 stride 2 convolution).

    A 3x3 strided convolution block that halves the spatial dimensions (H, W)
    and increases the channel count.
    """

    def __init__(self, in_channels: int, out_channels: int):
        """
        Initializes the downsampling module.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
        """
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
        """
        Runs the forward pass for the downsampling module.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        return self.conv(x)


class _UpsamplingModule(nn.Module):
    """
    The 'upsampling module' from the diagram.

    A 2x2 transposed convolution that doubles the spatial dimensions (H, W)
    and halves the channel count.
    """

    def __init__(self, in_channels: int, out_channels: int):
        """
        Initializes the upsampling module.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels (typically in_channels // 2).
        """
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Runs the forward pass for the upsampling module.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        return self.up(x)


class _LocalizationModule(nn.Module):
    """
    The 'localization module' from the diagram, used in the decoder.

    A block consisting of two sequential 3x3 convolutions, each followed
    by BatchNorm and ReLU. This is structurally identical to the
    _ContextModule but is used on the decoder path.
    """

    def __init__(self, in_channels: int, out_channels: int):
        """
        Initializes the localization module.

        Args:
            in_channels: Number of input channels (from concatenated skip + upsample).
            out_channels: Number of output channels.
        """
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
        """
        Runs the forward pass for the localization module.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        return self.block(x)


class _SegmentationLayer(nn.Module):
    """
    The 'segmentation layer' from the diagram.

    A single 1x1 convolution used to map feature channels to the final
    number of classes.
    """

    def __init__(self, in_channels: int, out_channels: int):
        """
        Initializes the segmentation layer.

        Args:
            in_channels: Number of input feature channels.
            out_channels: Number of output classes.
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Runs the forward pass for the segmentation layer.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        return self.conv(x)


# ---------------------------------------------
# The U-Net Model
# ---------------------------------------------


class ImprovedUNet(nn.Module):
    """
    The complete U-Net style model for HipMRI segmentation, based on the
    provided architecture diagram.

    This model features an encoder-decoder structure with skip connections
    and deep supervision, where segmentation maps from multiple decoder
    levels are upscaled and summed for the final output.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 6):
        """
        Initializes the U-Net model.

        Args:
            in_channels: Number of input image channels (e.g., 1 for grayscale).
            num_classes: Number of output segmentation classes.
        """
        super().__init__()

        # --- Encoder Path ---
        # Level 1 (256x128)
        self.context1 = _ContextModule(in_channels, 16)
        self.down1 = _DownsamplingModule(16, 32)

        # Level 2 (128x64)
        self.context2 = _ContextModule(32, 32)
        self.down2 = _DownsamplingModule(32, 64)

        # Level 3 (64x32)
        self.context3 = _ContextModule(64, 64)
        self.down3 = _DownsamplingModule(64, 128)

        # Level 4 (32x16)
        self.context4 = _ContextModule(128, 128)
        self.down4 = _DownsamplingModule(128, 256)

        # --- Bottleneck --- (16x8)
        self.bottleneck = _ContextModule(256, 256)

        # --- Decoder Path ---
        # Level 4 (32x16)
        self.up4 = _UpsamplingModule(256, 128)
        self.loc4 = _LocalizationModule(
            128 + 128, 128
        )  # Concatenates upsampled with context4 output
        self.seg4 = _SegmentationLayer(128, num_classes)

        # Level 3 (64x32)
        self.up3 = _UpsamplingModule(128, 64)
        self.loc3 = _LocalizationModule(
            64 + 64, 64
        )  # Concatenates upsampled with context3 output
        self.seg3 = _SegmentationLayer(64, num_classes)

        # Level 2 (128x64)
        self.up2 = _UpsamplingModule(64, 32)
        self.loc2 = _LocalizationModule(
            32 + 32, 32
        )  # Concatenates upsampled with context2 output
        self.seg2 = _SegmentationLayer(32, num_classes)

        # Level 1 (256x128)
        self.up1 = _UpsamplingModule(32, 16)
        self.loc1 = _LocalizationModule(
            16 + 16, 16
        )  # Concatenates upsampled with context1 output

        # Final output segmentation layer
        self.final_seg_layer = _SegmentationLayer(16, num_classes)

        # Apply Kaiming initialization
        self.apply(init_kaiming_normal_)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Runs the forward pass of the U-Net.

        Args:
            x: The input batch of images [B, C_in, H, W].

        Returns:
            The raw logits for each class [B, C_out, H, W].
        """
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
    """
    Factory function for quick construction of the ImprovedUNet model.

    Args:
        in_channels: Number of input image channels.
        num_classes: Number of output segmentation classes.

    Returns:
        An instance of the ImprovedUNet model.
    """
    return ImprovedUNet(in_channels=in_channels, num_classes=num_classes)


def count_params(model: nn.Module) -> int:
    """
    Calculates the total number of trainable parameters in a model.

    Args:
        model: The PyTorch model (nn.Module).

    Returns:
        The integer count of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# -----------------------------------------------------------------
# --- Sanity Check ---
# -----------------------------------------------------------------

if __name__ == "__main__":
    """
    Runs a sanity check when the script is executed directly.

    Creates a model, passes a dummy tensor, and asserts that the
    output shape is correct.
    """
    print("Testing the U-Net model based on diagram (2D adaptation)...")
    net = create_model(in_channels=1, num_classes=6)

    # Test with 256x128 input
    x = torch.randn(2, 1, 256, 128)  # Batch size 2, 1 channel, 256x128
    print(f"Input shape: {x.shape}")
    y = net(x)

    # Expected output shape: (batch_size, num_classes, 256, 128)
    print(f"Output shape: {tuple(y.shape)}")
    print(f"Model params: {count_params(net):,}")

    expected_output_shape = (2, 6, 256, 128)
    assert (
        tuple(y.shape) == expected_output_shape
    ), f"Output shape mismatch! Expected {expected_output_shape}, got {tuple(y.shape)}"
    print("Output shape is correct!")
