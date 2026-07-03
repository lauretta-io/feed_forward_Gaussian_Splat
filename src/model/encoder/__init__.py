from typing import Optional

from .encoder import Encoder
from .cfg import EncoderCfg, EncoderCostVolumeCfg, EncoderReSplatCfg
from .visualization.encoder_visualizer import EncoderVisualizer


def get_encoder(cfg: EncoderCfg) -> tuple[Encoder, Optional[EncoderVisualizer]]:
    if cfg.name == "resplat":
        from .encoder_resplat import EncoderReSplat

        encoder = EncoderReSplat
        visualizer = None
    elif cfg.name == "costvolume":
        from .mvsplat.encoder_costvolume import EncoderCostVolume
        from .mvsplat.visualization.encoder_visualizer_costvolume import (
            EncoderVisualizerCostVolume,
        )

        encoder = EncoderCostVolume
        visualizer = EncoderVisualizerCostVolume
    else:
        raise ValueError(f"Unknown encoder: {cfg.name}")

    encoder = encoder(cfg)
    if visualizer is not None:
        visualizer = visualizer(cfg.visualizer, encoder)
    return encoder, visualizer


__all__ = [
    "Encoder",
    "EncoderCfg",
    "EncoderCostVolumeCfg",
    "EncoderReSplatCfg",
    "get_encoder",
]
