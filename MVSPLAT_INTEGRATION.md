# MVSplat Integration

This repository keeps ReSplat as the primary runtime and exposes MVSplat as a
side-by-side model path.

## Source snapshot

MVSplat source was imported from the GitHub `main` archive for
`donydchen/mvsplat` at commit `01f9a28edb5eb68416e7e63b01f8d90c3bdfbf01`.
The archive contents are tracked under `third_party/mvsplat-main/`; this is not
a clone, submodule, or second remote. See
`third_party/mvsplat-main/PROVENANCE.md` for the provenance and license note.

The runtime integration ports the required model pieces into namespaced
first-party modules:

- `src/model/encoder/mvsplat/` for the MVSplat cost-volume encoder and support
  code.
- `src/model/decoder/mvsplat/` for the MVSplat CUDA splatting decoder.
- `src/model/encodings/` and `assets/mvsplat/` for MVSplat-only support assets.

## Configs

The existing ReSplat configs remain unchanged. MVSplat can be selected with:

```bash
python -m src.main +experiment=mvsplat_re10k
python -m src.main +experiment=mvsplat_acid
python -m src.main +experiment=mvsplat_dtu
```

These experiments select:

- `model/encoder: costvolume`
- `model/decoder: mvsplat_splatting_cuda`

## Optional dependencies

The ReSplat environment targets Python 3.12, PyTorch 2.7.0, and CUDA 12.8.
MVSplat documents Python 3.10, PyTorch 2.1.2, and CUDA 11.8. No package
versions were changed for this integration.

The legacy MVSplat CUDA rasterizer is optional and only required when running
`model/decoder=mvsplat_splatting_cuda`:

```bash
pip install git+https://github.com/dcharatan/diff-gaussian-rasterization-modified
```

If it is missing, the decoder import still succeeds and raises a targeted error
only when the MVSplat CUDA rendering path is executed.

## Checkpoints

MVSplat pretrained weights are not included. Place or symlink weights manually
before running MVSplat inference:

- `checkpoints/re10k.ckpt`
- `checkpoints/acid.ckpt`
- `checkpoints/gmdepth-scale1-resumeflowthings-scannet-5d9d7964.pth`

The `gmdepth` file is the UniMatch depth checkpoint expected by the cost-volume
encoder when configured to load pretrained UniMatch weights.

## Data and outputs

`datasets/google_drive/` contains copied comparison and evaluation artifacts
from Drive. Treat those files as data artifacts, not source code. The normal
ReSplat output layout is preserved; test runs should continue writing under
`outputs/test`.
