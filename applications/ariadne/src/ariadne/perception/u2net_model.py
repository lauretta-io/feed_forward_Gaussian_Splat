"""Official-checkpoint-compatible U²-Net architecture.

Adapted from the Apache-2.0 reference implementation at
https://github.com/xuebinqin/U-2-Net/blob/master/model/u2net.py. Only the full
U²-Net used by ARIADNE is included; U²-NetP is intentionally omitted.
"""

from __future__ import annotations

from typing import cast

import torch
from torch import nn
from torch.nn import functional as functional


class RebnConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int = 1) -> None:
        super().__init__()
        self.conv_s1 = nn.Conv2d(
            in_channels, out_channels, 3, padding=dilation, dilation=dilation
        )
        self.bn_s1 = nn.BatchNorm2d(out_channels)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.relu_s1(self.bn_s1(self.conv_s1(inputs))))


def _upsample_like(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return functional.interpolate(
        source, size=target.shape[2:], mode="bilinear", align_corners=False
    )


class Rsu(nn.Module):
    """Pooled residual U-block with reference-compatible parameter names."""

    def __init__(self, depth: int, in_channels: int, mid_channels: int, out_channels: int) -> None:
        super().__init__()
        if depth not in {4, 5, 6, 7}:
            raise ValueError("pooled RSU depth must be between four and seven")
        self.depth = depth
        self.rebnconvin = RebnConv(in_channels, out_channels)
        for level in range(1, depth):
            input_channels = out_channels if level == 1 else mid_channels
            setattr(self, f"rebnconv{level}", RebnConv(input_channels, mid_channels))
            if level < depth - 1:
                setattr(self, f"pool{level}", nn.MaxPool2d(2, stride=2, ceil_mode=True))
        setattr(self, f"rebnconv{depth}", RebnConv(mid_channels, mid_channels, dilation=2))
        for level in range(depth - 1, 0, -1):
            output_channels = out_channels if level == 1 else mid_channels
            setattr(
                self,
                f"rebnconv{level}d",
                RebnConv(mid_channels * 2, output_channels),
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.rebnconvin(inputs)
        value = residual
        skips: dict[int, torch.Tensor] = {}
        for level in range(1, self.depth):
            value = getattr(self, f"rebnconv{level}")(value)
            skips[level] = value
            if level < self.depth - 1:
                value = getattr(self, f"pool{level}")(value)
        value = getattr(self, f"rebnconv{self.depth}")(value)
        for level in range(self.depth - 1, 0, -1):
            value = getattr(self, f"rebnconv{level}d")(
                torch.cat((value, skips[level]), dim=1)
            )
            if level > 1:
                value = _upsample_like(value, skips[level - 1])
        return cast(torch.Tensor, value + residual)


class Rsu4f(nn.Module):
    """Dilated residual U-block with reference-compatible parameter names."""

    def __init__(self, in_channels: int, mid_channels: int, out_channels: int) -> None:
        super().__init__()
        self.rebnconvin = RebnConv(in_channels, out_channels)
        self.rebnconv1 = RebnConv(out_channels, mid_channels)
        self.rebnconv2 = RebnConv(mid_channels, mid_channels, dilation=2)
        self.rebnconv3 = RebnConv(mid_channels, mid_channels, dilation=4)
        self.rebnconv4 = RebnConv(mid_channels, mid_channels, dilation=8)
        self.rebnconv3d = RebnConv(mid_channels * 2, mid_channels, dilation=4)
        self.rebnconv2d = RebnConv(mid_channels * 2, mid_channels, dilation=2)
        self.rebnconv1d = RebnConv(mid_channels * 2, out_channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.rebnconvin(inputs)
        first = self.rebnconv1(residual)
        second = self.rebnconv2(first)
        third = self.rebnconv3(second)
        fourth = self.rebnconv4(third)
        third_decoder = self.rebnconv3d(torch.cat((fourth, third), dim=1))
        second_decoder = self.rebnconv2d(torch.cat((third_decoder, second), dim=1))
        first_decoder = self.rebnconv1d(torch.cat((second_decoder, first), dim=1))
        return cast(torch.Tensor, first_decoder + residual)


class U2Net(nn.Module):
    """Full 176 MB U²-Net architecture used by the official pretrained model."""

    def __init__(self, in_channels: int = 3, out_channels: int = 1) -> None:
        super().__init__()
        self.stage1 = Rsu(7, in_channels, 32, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage2 = Rsu(6, 64, 32, 128)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage3 = Rsu(5, 128, 64, 256)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage4 = Rsu(4, 256, 128, 512)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage5 = Rsu4f(512, 256, 512)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage6 = Rsu4f(512, 256, 512)

        self.stage5d = Rsu4f(1024, 256, 512)
        self.stage4d = Rsu(4, 1024, 128, 256)
        self.stage3d = Rsu(5, 512, 64, 128)
        self.stage2d = Rsu(6, 256, 32, 64)
        self.stage1d = Rsu(7, 128, 16, 64)

        self.side1 = nn.Conv2d(64, out_channels, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_channels, 3, padding=1)
        self.side3 = nn.Conv2d(128, out_channels, 3, padding=1)
        self.side4 = nn.Conv2d(256, out_channels, 3, padding=1)
        self.side5 = nn.Conv2d(512, out_channels, 3, padding=1)
        self.side6 = nn.Conv2d(512, out_channels, 3, padding=1)
        self.outconv = nn.Conv2d(6 * out_channels, out_channels, 1)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        first = self.stage1(inputs)
        second = self.stage2(self.pool12(first))
        third = self.stage3(self.pool23(second))
        fourth = self.stage4(self.pool34(third))
        fifth = self.stage5(self.pool45(fourth))
        sixth = self.stage6(self.pool56(fifth))

        fifth_decoder = self.stage5d(torch.cat((_upsample_like(sixth, fifth), fifth), dim=1))
        fourth_decoder = self.stage4d(
            torch.cat((_upsample_like(fifth_decoder, fourth), fourth), dim=1)
        )
        third_decoder = self.stage3d(
            torch.cat((_upsample_like(fourth_decoder, third), third), dim=1)
        )
        second_decoder = self.stage2d(
            torch.cat((_upsample_like(third_decoder, second), second), dim=1)
        )
        first_decoder = self.stage1d(
            torch.cat((_upsample_like(second_decoder, first), first), dim=1)
        )

        side1 = self.side1(first_decoder)
        side2 = _upsample_like(self.side2(second_decoder), side1)
        side3 = _upsample_like(self.side3(third_decoder), side1)
        side4 = _upsample_like(self.side4(fourth_decoder), side1)
        side5 = _upsample_like(self.side5(fifth_decoder), side1)
        side6 = _upsample_like(self.side6(sixth), side1)
        fused = self.outconv(torch.cat((side1, side2, side3, side4, side5, side6), dim=1))
        outputs = (fused, side1, side2, side3, side4, side5, side6)
        return tuple(torch.sigmoid(output) for output in outputs)
