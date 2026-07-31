from __future__ import annotations

import torch
from torch import nn

from .resplat_embedding_adapter import ResplatEmbeddingAdapter
from .u2net_original import U2NET


class ResplatBackboneU2Net(nn.Module):
    def __init__(
        self,
        embedding_channels: int | None = None,
        adapter_channels: int = 128,
        decoder_base_channels: int = 32,
        token_grid_size: tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        self.adapter = ResplatEmbeddingAdapter(
            input_channels=embedding_channels,
            output_channels=3,
            hidden_channels=adapter_channels,
            token_grid_size=token_grid_size,
        )
        self.decoder = U2NET(in_ch=3, base_ch=decoder_base_channels)

    def forward(
        self,
        embedding: torch.Tensor,
        output_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        adapted = self.adapter(embedding, output_size=output_size)
        return self.decoder(adapted)

