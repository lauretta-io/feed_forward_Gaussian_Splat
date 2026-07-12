from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualUNestedBlock(nn.Module):
    """Small RSU-style block used by the compact local U2-Net implementation."""

    def __init__(self, in_ch: int, mid_ch: int, out_ch: int) -> None:
        super().__init__()
        self.in_conv = ConvBlock(in_ch, out_ch)
        self.down = nn.Sequential(nn.MaxPool2d(2, ceil_mode=True), ConvBlock(out_ch, mid_ch))
        self.bridge = ConvBlock(mid_ch, mid_ch)
        self.up = ConvBlock(mid_ch + out_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.in_conv(x)
        x1 = self.down(x0)
        x2 = self.bridge(x1)
        x2 = F.interpolate(x2, size=x0.shape[-2:], mode="bilinear", align_corners=False)
        return self.up(torch.cat([x2, x0], dim=1)) + x0


class U2NET(nn.Module):
    """Compact U2-Net-compatible saliency model.

    The public contract is intentionally simple: input [B, C, H, W], output
    sigmoid saliency [B, 1, H, W]. Checkpoints with either raw state_dict or a
    {"state_dict": ...} wrapper are supported by load_u2net_checkpoint.
    """

    def __init__(
        self,
        in_ch: int = 3,
        base_ch: int = 32,
        out_ch: int = 1,
    ) -> None:
        super().__init__()
        c1, c2, c3, c4 = base_ch, base_ch * 2, base_ch * 4, base_ch * 8
        self.enc1 = ResidualUNestedBlock(in_ch, c1, c1)
        self.enc2 = ResidualUNestedBlock(c1, c1, c2)
        self.enc3 = ResidualUNestedBlock(c2, c2, c3)
        self.enc4 = ResidualUNestedBlock(c3, c3, c4)
        self.pool = nn.MaxPool2d(2, ceil_mode=True)
        self.dec3 = ConvBlock(c4 + c3, c3)
        self.dec2 = ConvBlock(c3 + c2, c2)
        self.dec1 = ConvBlock(c2 + c1, c1)
        self.side1 = nn.Conv2d(c1, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(c2, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(c3, out_ch, 3, padding=1)
        self.fuse = nn.Conv2d(out_ch * 3, out_ch, 1)

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        d3 = F.interpolate(e4, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        s1 = self.side1(d1)
        s2 = F.interpolate(self.side2(d2), size=input_size, mode="bilinear", align_corners=False)
        s3 = F.interpolate(self.side3(d3), size=input_size, mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([s1, s2, s3], dim=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward_logits(x))


class U2NETP(U2NET):
    def __init__(self, in_ch: int = 3, out_ch: int = 1) -> None:
        super().__init__(in_ch=in_ch, base_ch=16, out_ch=out_ch)


def load_u2net_checkpoint(
    model: nn.Module,
    checkpoint: str | Path | None,
    strict: bool = False,
) -> str:
    if checkpoint is None or str(checkpoint) == "":
        return "untrained"
    path = Path(checkpoint)
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict):
        cleaned = {}
        for key, value in state.items():
            cleaned[key.removeprefix("module.").removeprefix("model.")] = value
        missing, unexpected = model.load_state_dict(cleaned, strict=strict)
        if missing or unexpected:
            return f"checkpoint_loaded_partial:{path}"
        return f"checkpoint_loaded:{path}"
    raise ValueError(f"Unsupported checkpoint format: {path}")

