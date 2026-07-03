# OpenSplat CPU Inference Integration

This repo keeps ReSplat, MVSplat, and CUDA `gsplat` behavior unchanged by
default. The OpenSplat path is an optional inference-only renderer for smoke and
debug runs on CPU:

```bash
python scripts/infer_colmap.py \
  --cpu \
  --model_preset dl3dv_8v_256x448 \
  --scene_path datasets/example_colmap_scene \
  --num_target 1 \
  --output_dir outputs/opensplat_cpu_smoke
```

`--cpu` maps to:

- `--device cpu`
- `runtime.cpu=true`
- `runtime.device=cpu`
- `model/decoder=opensplat_cpu`
- `--no_eval`

The CPU path renders RGB only. Depth, alpha buffers, CPU training, and OpenSplat
standalone training are out of scope for this integration.

## External Dependency

OpenSplat is licensed AGPL-3.0. To avoid redistributing AGPL source or binaries
inside this MIT-licensed codebase, this repo does not vendor OpenSplat. Instead,
the decoder expects an importable Python extension named `opensplat_cpu_ext`.

Build and install that extension outside this repo from
`pierotofy/OpenSplat`, using its CPU runtime:

```bash
# In an external checkout, not inside this repository.
git clone https://github.com/pierotofy/OpenSplat.git
cd OpenSplat

# Build OpenSplat with CPU libtorch and GPU_RUNTIME=CPU, then expose the
# Python extension functions listed below. Exact build flags depend on the
# local libtorch/Python extension wrapper you use.
```

The extension should wrap OpenSplat CPU implementations such as
`ProjectGaussiansCPU`, `RasterizeGaussiansCPU`, and the CPU spherical harmonics
helpers when available.

## Python Extension Contract

Required module name:

```python
import opensplat_cpu_ext
```

Required functions:

```python
project_gaussians_cpu(
    means,
    scales,
    quats_wxyz,
    viewmat,
    projmat,
    fx,
    fy,
    cx,
    cy,
    height,
    width,
    clip_thresh,
) -> xys, radii, conics, cov2d, cam_depths

rasterize_gaussians_cpu(
    xys,
    radii,
    conics,
    colors,
    opacities,
    cov2d,
    cam_depths,
    height,
    width,
    background,
) -> image_hwc
```

Optional function:

```python
eval_sh_cpu(degree, viewdirs, coeffs) -> colors
```

If `eval_sh_cpu` is missing, `OpenSplatCPUDecoder` falls back to DC color only.
All tensors passed to the extension are CPU `float32` tensors.

## ReSplat Usage

The Hydra decoder config is available as:

```bash
model/decoder=opensplat_cpu
```

The runtime controls are:

```yaml
runtime:
  device: auto  # auto | cuda | cpu | mps
  cpu: false
```

Setting `runtime.cpu=true` forces CPU execution and prevents Lightning from
requesting GPU devices. The CUDA decoder remains the default unless the CPU
decoder is selected explicitly.

## Limitations

- This path is for inference-smoke rendering of predicted Gaussians only.
- The encoder still needs to be usable on CPU for the selected checkpoint and
  experiment. CUDA-only encoder dependencies remain blockers for those specific
  model paths.
- `OpenSplatCPUDecoder` requires Gaussians with `means`, `scales`, `rotations`
  or `rotations_unnorm`, `harmonics`, and `opacities`.
- MVSplat CUDA decoder behavior is unchanged. The MVSplat encoder now preserves
  Gaussian scales and rotations so CPU-compatible decoders can consume them.
- Rendered depth and accumulated alpha are returned as `None` until the external
  extension exposes those buffers.

## Verification

Static checks:

```bash
python -m compileall src scripts/infer_colmap.py
python scripts/infer_colmap.py --help
```

Backend checks:

```bash
python - <<'PY'
from src.integrations.opensplat import load_backend
load_backend()
print("opensplat_cpu_ext is importable")
PY
```

CPU smoke run:

```bash
python scripts/infer_colmap.py \
  --cpu \
  --model_preset dl3dv_8v_256x448 \
  --scene_path <COLMAP scene> \
  --num_target 1 \
  --output_dir outputs/opensplat_cpu_smoke
```
