# Smoke Test Datasets

This workspace uses small public datasets for installation and integration
checks. They are intentionally not tracked by git; `datasets*`, `pretrained*`,
`checkpoints*`, `outputs*`, and `results*` are ignored.

## Cross-System Data Requirements

The three GPU systems do not consume the same dataset format, so a fair
side-by-side check needs one shared scene source plus per-system adapters:

| System | Required input format | Smoke data used here | Required weights |
| --- | --- | --- | --- |
| ReSplat | COLMAP scene with `images_4` or `images_8` plus `sparse/0` cameras | `datasets/dl3dv-colmap-demo/<scene>` | `pretrained/resplat-base-dl3dv-256x448-view8-1934a04c.pth` and gmdepth symlink |
| MVSplat | ReSplat/pixelSplat-style `.torch` chunks with `index.json` | `datasets/re10k/test/*.torch` from the 720p two-scene subset | `checkpoints/re10k.ckpt` and `checkpoints/gmdepth-scale1-resumeflowthings-scannet-5d9d7964.pth` |
| AnySplat | Flat directory of `.png`, `.jpg`, or `.jpeg` images | `datasets/anysplat-smoke-images`, copied from the DL3DV COLMAP demo | Hugging Face `lhjiang/anysplat`; it also pulls `facebook/VGGT-1B` |

For quick compatibility checks, use the commands below as-is. For a stricter
three-way qualitative comparison on the same visual content, use the DL3DV
COLMAP demo as the shared source:

- Run ReSplat directly on the COLMAP scene.
- Run AnySplat on selected frames copied from the same scene into a flat image
  folder.
- Convert the same scene into chunk format before running MVSplat. The current
  MVSplat smoke command uses the RE10K two-scene subset because this repo does
  not currently include a DL3DV-COLMAP-to-`.torch` conversion adapter for the
  MVSplat evaluation path.

The RE10K subset is 720p. MVSplat's RE10K experiment defaults to 360p shape
validation, so smoke tests against this subset must pass `dataset.highres=true`.

## Downloaded Assets

DL3DV COLMAP demo:

```bash
wget -O /tmp/resplat_datasets/dl3dv-colmap-demo.zip \
  https://huggingface.co/datasets/haofeixu/depthsplat/resolve/main/dl3dv-colmap-demo.zip
unzip -q -o /tmp/resplat_datasets/dl3dv-colmap-demo.zip -d datasets
```

RealEstate10K 720p two-scene test subset:

```bash
wget -O /tmp/resplat_datasets/re10k_720p_test_subset.zip \
  https://huggingface.co/datasets/haofeixu/depthsplat/resolve/main/re10k_720p_test_subset.zip
unzip -q -o /tmp/resplat_datasets/re10k_720p_test_subset.zip -d datasets
ln -s re10k_720p_test_subset datasets/re10k
```

ReSplat low-resolution DL3DV checkpoint:

```bash
mkdir -p pretrained
wget -O pretrained/resplat-base-dl3dv-256x448-view8-1934a04c.pth \
  https://huggingface.co/haofeixu/resplat/resolve/main/resplat-base-dl3dv-256x448-view8-1934a04c.pth
ln -s ../checkpoints/gmdepth-scale1-resumeflowthings-scannet-5d9d7964.pth \
  pretrained/gmdepth-scale1-resumeflowthings-scannet-5d9d7964.pth
```

MVSplat checkpoint compatibility links:

```bash
ln -s mvsplat-download/re10k.ckpt checkpoints/re10k.ckpt
ln -s mvsplat-download/acid.ckpt checkpoints/acid.ckpt
```

AnySplat smoke image folder:

```bash
mkdir -p datasets/anysplat-smoke-images
cp datasets/dl3dv-colmap-demo/02267acf6fb98de36173bf4e7db9734c8c421dcb00267e42964dc15134cbb1be/images_4/frame_00001.png \
  datasets/anysplat-smoke-images/frame_00001.png
cp datasets/dl3dv-colmap-demo/02267acf6fb98de36173bf4e7db9734c8c421dcb00267e42964dc15134cbb1be/images_4/frame_00030.png \
  datasets/anysplat-smoke-images/frame_00030.png
cp datasets/dl3dv-colmap-demo/02267acf6fb98de36173bf4e7db9734c8c421dcb00267e42964dc15134cbb1be/images_4/frame_00060.png \
  datasets/anysplat-smoke-images/frame_00060.png
```

## Optional Dependencies

AnySplat needs `huggingface_hub`, `torch_scatter`, and `safetensors` in addition
to the base ReSplat environment. On Ubuntu 20.04, the PyG `torch_scatter`
binary wheel for PyTorch 2.7/CUDA 12.8 can fail with `GLIBC_2.32 not found`.
Build it locally instead:

```bash
pip install huggingface_hub safetensors
CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST="8.6" \
  pip install --no-build-isolation --no-binary torch-scatter torch-scatter
```

MVSplat CUDA rendering needs the legacy rasterizer:

```bash
CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST="8.6" \
  pip install --no-build-isolation \
  git+https://github.com/dcharatan/diff-gaussian-rasterization-modified
```

## Verified Commands

ReSplat COLMAP smoke test:

```bash
python scripts/infer_colmap.py \
  --model_preset dl3dv_8v_256x448 \
  --data_dir datasets/dl3dv-colmap-demo \
  --scene_name 02267acf6fb98de36173bf4e7db9734c8c421dcb00267e42964dc15134cbb1be \
  --output_dir outputs/smoke/resplat_colmap \
  --num_target 1 \
  --target_selection remaining \
  --no_eval \
  --max_save_images 0
```

Result: passed. It loaded 60 COLMAP frames, selected 8 context views and 1
target view, loaded
`pretrained/resplat-base-dl3dv-256x448-view8-1934a04c.pth`, rendered one target
view, and wrote outputs under `outputs/smoke/resplat_colmap/`.

MVSplat RE10K smoke test:

```bash
python -m src.main \
  +experiment=mvsplat_re10k \
  mode=test \
  checkpointing.load=checkpoints/re10k.ckpt \
  dataset.highres=true \
  data_loader.test.num_workers=0 \
  test.compute_scores=false \
  test.save_image=false \
  test.save_video=false \
  test.save_gt_image=false \
  test.save_input_images=false \
  output_dir=outputs/smoke/mvsplat_re10k_highres
```

Result: passed after compatibility fixes. The downloaded subset is 720p, so
`dataset.highres=true` is required; without it, the loader expects 360x640 and
skips both scenes as bad-shape examples.

AnySplat image-folder smoke test:

```bash
python scripts/infer_anysplat.py \
  --input-dir datasets/anysplat-smoke-images \
  --output-dir outputs/smoke/anysplat \
  --device cuda:0
```

Result: passed. It wrote `predicted_poses.pt` and `manifest.json` under
`outputs/smoke/anysplat/`. First run downloads Hugging Face weights for
`lhjiang/anysplat` and `facebook/VGGT-1B`; the local Hugging Face cache can be
tens of GB.

## Integration Fixes Exercised

The MVSplat smoke test exposed three compatibility gaps:

- `EncoderCostVolume` now inherits the shared
  `src.model.encoder.encoder.Encoder` base class.
- `EncoderVisualizerCostVolume` now inherits the shared
  `src.model.encoder.visualization.encoder_visualizer.EncoderVisualizer` base
  class.
- `EncoderCostVolumeCfg` now has `no_crop_image: false`, matching the field
  `ModelWrapper.test_step` expects on encoder configs.
