from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class ResplatEmbeddingAdapter(nn.Module):
    """Convert saved ReSplat embeddings into saliency-decoder input maps."""

    def __init__(
        self,
        input_channels: int | None = None,
        output_channels: int = 3,
        hidden_channels: int = 128,
        token_grid_size: tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        self.token_grid_size = token_grid_size
        first: nn.Module
        if input_channels is None:
            first = nn.LazyConv2d(hidden_channels, 1)
        else:
            first = nn.Conv2d(input_channels, hidden_channels, 1)
        self.net = nn.Sequential(
            first,
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, output_channels, 1),
        )

    def _tokens_to_map(self, x: torch.Tensor) -> torch.Tensor:
        if self.token_grid_size is not None:
            h, w = self.token_grid_size
        else:
            side = int(math.sqrt(x.shape[1]))
            if side * side != x.shape[1]:
                raise ValueError(
                    "Token embeddings need --token-grid-size when token count is not square"
                )
            h = w = side
        return x.transpose(1, 2).reshape(x.shape[0], x.shape[2], h, w)

    def forward(
        self,
        embedding: torch.Tensor,
        output_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        if embedding.ndim == 3:
            embedding = self._tokens_to_map(embedding)
        elif embedding.ndim != 4:
            raise ValueError(f"Expected [B,C,H,W] or [B,N,C] embedding, got {tuple(embedding.shape)}")
        out = self.net(embedding.float())
        if output_size is not None and out.shape[-2:] != output_size:
            out = F.interpolate(out, size=output_size, mode="bilinear", align_corners=False)
        return out

