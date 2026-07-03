# AnySplat Impact Integration

This repository keeps ReSplat as the primary training and runtime path. AnySplat
Impact is integrated only as a side-by-side inference command.

## Source Snapshot

AnySplat Impact source was imported from the GitHub `main` archive for
`QuangTran1608/AnySplat_impact` at commit
`fe25779ce0ec2747635f6555ecedadcdd565da9e`. The archive contents are tracked
under `third_party/anysplat-impact-main/`; this is not a clone, submodule, or
second remote. See `third_party/anysplat-impact-main/PROVENANCE.md`.

The inference runtime lives under `src/integrations/anysplat/` so its modules do
not collide with ReSplat or the MVSplat integration.

## Inference

Run AnySplat on a folder of images:

```bash
python scripts/infer_anysplat.py \
  --input-dir /path/to/images \
  --output-dir outputs/anysplat/my-scene \
  --save-ply \
  --save-video
```

If `--output-dir` is omitted, outputs are written to
`outputs/anysplat/<input-folder-name>/`.

The command writes:

- `predicted_poses.pt`
- `manifest.json`
- `gaussians.ply` when `--save-ply` is set
- `rgb.mp4` and `depth.mp4` when `--save-video` is set

## Weights

Weights are not downloaded during setup. The CLI loads `lhjiang/anysplat` from
Hugging Face only when `scripts/infer_anysplat.py` is executed. Use
`--model-id` to point at a different Hugging Face model id or local model
directory.

## Optional Dependencies

The ReSplat environment targets Python 3.12, PyTorch 2.7.0, and CUDA 12.8.
AnySplat documents Python 3.10+, PyTorch 2.2.0, and CUDA 12.1. No package
versions were changed for this integration.

AnySplat inference requires additional packages beyond the base ReSplat setup,
including `huggingface_hub`, `torch_scatter`, `gsplat`, `opencv-python`,
`scipy`, and `plyfile`. Video export also uses `matplotlib`, `imageio`, and
`sk-video`. The CLI checks for these modules at runtime and exits with a
targeted message when they are missing.

## Out of Scope

AnySplat training, datasets, losses, post-optimization tools, Gradio, FastAPI,
Qwen video analysis, and hard-coded service integrations are intentionally not
wired into ReSplat in this pass.
