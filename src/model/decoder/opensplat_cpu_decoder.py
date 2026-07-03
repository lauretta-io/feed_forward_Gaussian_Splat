from __future__ import annotations

from math import isqrt

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from ...dataset import DatasetCfg
from ..types import Gaussians
from .cfg import OpenSplatCPUDecoderCfg
from .decoder import Decoder, DecoderOutput, DepthRenderingMode


SH_C0 = 0.28209479177387814


class OpenSplatCPUDecoder(Decoder[OpenSplatCPUDecoderCfg]):
    background_color: Float[Tensor, "3"]

    def __init__(
        self,
        cfg: OpenSplatCPUDecoderCfg,
        dataset_cfg: DatasetCfg,
    ) -> None:
        super().__init__(cfg, dataset_cfg)
        self.register_buffer(
            "background_color",
            torch.tensor(dataset_cfg.background_color, dtype=torch.float32),
            persistent=False,
        )

    def forward(
        self,
        gaussians: Gaussians,
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        image_shape: tuple[int, int],
        depth_mode: DepthRenderingMode | None = None,
    ) -> DecoderOutput:
        if depth_mode is not None:
            raise NotImplementedError(
                "OpenSplat CPU decoder currently renders RGB only; depth output "
                "requires a backend exposing depth or alpha buffers."
            )

        from src.integrations.opensplat import load_backend

        backend = load_backend()
        h, w = image_shape

        means = gaussians.means.detach().to("cpu", dtype=torch.float32)
        scales = _require_tensor(gaussians.scales, "scales")
        rotations = _require_tensor(
            gaussians.rotations_unnorm
            if gaussians.rotations_unnorm is not None
            else gaussians.rotations,
            "rotations or rotations_unnorm",
        )
        harmonics = gaussians.harmonics.detach().to("cpu", dtype=torch.float32)
        opacities = gaussians.opacities.detach().to("cpu", dtype=torch.float32)
        extrinsics = extrinsics.detach().to("cpu", dtype=torch.float32)
        intrinsics = intrinsics.detach().to("cpu", dtype=torch.float32)
        near = near.detach().to("cpu", dtype=torch.float32)
        far = far.detach().to("cpu", dtype=torch.float32)
        background = self.background_color.detach().to("cpu", dtype=torch.float32)

        scales = scales.detach().to("cpu", dtype=torch.float32)
        rotations = rotations.detach().to("cpu", dtype=torch.float32)
        rotations = _xyzw_to_wxyz(rotations)

        batch_size, num_views = extrinsics.shape[:2]
        all_colors = []

        for batch_idx in range(batch_size):
            batch_colors = []
            batch_means = means[batch_idx]
            batch_scales = scales[batch_idx]
            batch_rotations = rotations[batch_idx]
            batch_harmonics = harmonics[batch_idx]
            batch_opacities = opacities[batch_idx]

            for view_idx in range(num_views):
                c2w = extrinsics[batch_idx, view_idx]
                viewmat = c2w.inverse()
                k = intrinsics[batch_idx, view_idx].clone()
                k[0] *= float(w)
                k[1] *= float(h)
                fx = float(k[0, 0].item())
                fy = float(k[1, 1].item())
                cx = float(k[0, 2].item())
                cy = float(k[1, 2].item())
                projmat = _projection_matrix(
                    float(near[batch_idx, view_idx].item()),
                    float(far[batch_idx, view_idx].item()),
                    fx,
                    fy,
                    cx,
                    cy,
                    h,
                    w,
                )

                xys, radii, conics, cov2d, cam_depths = backend.project_gaussians_cpu(
                    batch_means,
                    batch_scales,
                    batch_rotations,
                    viewmat,
                    projmat,
                    fx,
                    fy,
                    cx,
                    cy,
                    h,
                    w,
                    self.cfg.clip_thresh,
                )
                colors = _colors_for_view(
                    backend,
                    batch_harmonics,
                    batch_means,
                    c2w[:3, 3],
                    self.cfg.sh_degree,
                )
                image_hwc = backend.rasterize_gaussians_cpu(
                    xys,
                    radii,
                    conics,
                    colors,
                    batch_opacities,
                    cov2d,
                    cam_depths,
                    h,
                    w,
                    background,
                )
                if isinstance(image_hwc, (tuple, list)):
                    image_hwc = image_hwc[0]
                batch_colors.append(image_hwc.permute(2, 0, 1).contiguous())
            all_colors.append(torch.stack(batch_colors, dim=0))

        color = torch.stack(all_colors, dim=0)
        return DecoderOutput(color=color, depth=None, accumulated_alpha=None)


def _require_tensor(tensor: Tensor | None, name: str) -> Tensor:
    if tensor is None:
        raise ValueError(
            f"OpenSplat CPU decoder requires Gaussian `{name}`. Ensure the "
            "selected encoder preserves scales and rotations."
        )
    return tensor


def _xyzw_to_wxyz(rotations: Tensor) -> Tensor:
    return torch.cat([rotations[..., 3:], rotations[..., :3]], dim=-1)


def _projection_matrix(
    near: float,
    far: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    height: int,
    width: int,
) -> Tensor:
    proj = torch.zeros((4, 4), dtype=torch.float32)
    proj[0, 0] = 2.0 * fx / width
    proj[1, 1] = 2.0 * fy / height
    proj[0, 2] = 1.0 - 2.0 * cx / width
    proj[1, 2] = 2.0 * cy / height - 1.0
    proj[2, 2] = far / (far - near)
    proj[2, 3] = -(far * near) / (far - near)
    proj[3, 2] = 1.0
    return proj


def _colors_for_view(
    backend: object,
    harmonics: Tensor,
    means: Tensor,
    camera_position: Tensor,
    sh_degree: int | None,
) -> Tensor:
    degree = sh_degree
    if degree is None:
        degree = isqrt(harmonics.shape[-1]) - 1

    coeffs = harmonics.permute(0, 2, 1).contiguous()
    if hasattr(backend, "eval_sh_cpu"):
        viewdirs = F.normalize(means - camera_position[None], dim=-1)
        return backend.eval_sh_cpu(degree, viewdirs, coeffs).clamp(0.0, 1.0)

    return (harmonics[..., 0] * SH_C0 + 0.5).clamp(0.0, 1.0)
