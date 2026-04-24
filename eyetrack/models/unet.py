import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    summary:
    param in_channels: input
    param out_channels: output
    return: none
    """

    def __init__(self, in_channels: int, out_channels: int):
        """
        summary:
        param in_channels: input
        param out_channels: output
        return: none
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
        summary:
        param x: input
        return: output
        """
        return self.block(x)

    def fuse_model(self, is_qat: bool = True) -> None:
        """
        summary: Conv-BN-ReLU quantizationtraining inference
        param is_qat: QAT
        return: none
        """
        fusion_spec = [["0", "1", "2"], ["3", "4", "5"]]
        if is_qat and hasattr(torch.ao.quantization, "fuse_modules_qat"):
            torch.ao.quantization.fuse_modules_qat(self.block, fusion_spec, inplace=True)
            return

        torch.ao.quantization.fuse_modules(self.block, fusion_spec, inplace=True)


class Down(nn.Module):
    """
    summary: U-Net
    param in_channels: input
    param out_channels: output
    return: none
    """

    def __init__(self, in_channels: int, out_channels: int):
        """
        summary:
        param in_channels: input
        param out_channels: output
        return: none
        """
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        summary:
        param x: input
        return: output
        """
        x = self.pool(x)
        x = self.conv(x)
        return x


class Up(nn.Module):
    """
    summary: U-Net
    param in_channels: input
    param skip_channels:
    param out_channels: output
    return: none
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        """
        summary:
        param in_channels: input
        param skip_channels:
        param out_channels: output
        return: none
        """
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = DoubleConv(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        summary:
        param x: input
        param skip:
        return: output
        """
        x = self.up(x)

        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)

        x = F.pad(
            x,
            [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
        )

        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    """
    summary: U-Net
    param in_channels: input
    param num_classes: outputclass
    param base_channels:
    return: none
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 4, base_channels: int = 32):
        """
        summary: U-Net
        param in_channels: input
        param num_classes: outputclass
        param base_channels:
        return: none
        """
        super().__init__()

        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        self.down4 = Down(base_channels * 8, base_channels * 16)

        self.up1 = Up(base_channels * 16, base_channels * 8, base_channels * 8)
        self.up2 = Up(base_channels * 8, base_channels * 4, base_channels * 4)
        self.up3 = Up(base_channels * 4, base_channels * 2, base_channels * 2)
        self.up4 = Up(base_channels * 2, base_channels, base_channels)

        self.outc = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        summary:, output class logits
        param x: inputimage
        return: logits
        """
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)
        return logits

    def fuse_model(self, is_qat: bool = True) -> None:
        """
        summary: U-Net
        param is_qat: QAT
        return: none
        """
        self.inc.fuse_model(is_qat=is_qat)
        self.down1.conv.fuse_model(is_qat=is_qat)
        self.down2.conv.fuse_model(is_qat=is_qat)
        self.down3.conv.fuse_model(is_qat=is_qat)
        self.down4.conv.fuse_model(is_qat=is_qat)
        self.up1.conv.fuse_model(is_qat=is_qat)
        self.up2.conv.fuse_model(is_qat=is_qat)
        self.up3.conv.fuse_model(is_qat=is_qat)
        self.up4.conv.fuse_model(is_qat=is_qat)
