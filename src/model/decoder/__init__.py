from ...dataset import DatasetCfg
from .cfg import (
    DecoderCfg,
    GSplatDecoderSplattingCUDACfg,
    MVSplatDecoderSplattingCUDACfg,
    OpenSplatCPUDecoderCfg,
)
from .decoder import Decoder



def get_decoder(decoder_cfg: DecoderCfg, dataset_cfg: DatasetCfg) -> Decoder:
    if decoder_cfg.name == "gsplat":
        from .gsplat_decoder_splatting_cuda import GSplatDecoderSplattingCUDA

        decoder = GSplatDecoderSplattingCUDA
    elif decoder_cfg.name == "mvsplat_splatting_cuda":
        from .mvsplat.decoder_splatting_cuda import DecoderSplattingCUDA

        decoder = DecoderSplattingCUDA
    elif decoder_cfg.name == "opensplat_cpu":
        from .opensplat_cpu_decoder import OpenSplatCPUDecoder

        decoder = OpenSplatCPUDecoder
    else:
        raise ValueError(f"Unknown decoder: {decoder_cfg.name}")

    return decoder(decoder_cfg, dataset_cfg)


__all__ = [
    "Decoder",
    "DecoderCfg",
    "GSplatDecoderSplattingCUDACfg",
    "MVSplatDecoderSplattingCUDACfg",
    "OpenSplatCPUDecoderCfg",
    "get_decoder",
]
