from dataclasses import dataclass
from typing import Literal


@dataclass
class GSplatDecoderSplattingCUDACfg:
    name: Literal["gsplat"]
    scale_invariant: bool
    use_covariances: bool | None = True


@dataclass
class MVSplatDecoderSplattingCUDACfg:
    name: Literal["mvsplat_splatting_cuda"]


@dataclass
class OpenSplatCPUDecoderCfg:
    name: Literal["opensplat_cpu"]
    clip_thresh: float = 0.01
    sh_degree: int | None = None


DecoderCfg = (
    GSplatDecoderSplattingCUDACfg
    | MVSplatDecoderSplattingCUDACfg
    | OpenSplatCPUDecoderCfg
)
