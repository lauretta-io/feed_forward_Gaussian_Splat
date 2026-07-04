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

## Technical Architecture Outlines

### ReSplat COLMAP

ReSplat is the native feed-forward Gaussian splatting path in this workspace.
`scripts/infer_colmap.py` converts a COLMAP scene into the shared
context/target batch format, then builds `EncoderReSplat` and the configured
decoder.

- Input: COLMAP images plus `sparse/0` cameras, with `images_4` or `images_8`
  supplying the resized frames used by the smoke tests.
- Encoder: `EncoderReSplat` runs `MultiViewUniMatch` over the context images to
  predict depth candidates, match probabilities, mono features, CNN features,
  and multi-view features.
- Gaussian head: image features, latent depth, match probability, and fused
  features are projected to latent 3D points. A KNN `PlainPointTransformer`
  predicts Gaussian scale, rotation, spherical harmonics, offset, and opacity
  channels.
- Refinement: when `num_refine` is enabled, the model renders the current
  Gaussian set, extracts render-error features, applies multi-view attention,
  and predicts per-Gaussian updates through a recurrent update module.
- Decoder: the default path uses the `gsplat` CUDA splatting decoder. The
  COLMAP script can also select the OpenSplat CPU decoder for smoke rendering.
- Logged evidence: target renders, context previews, per-scene metrics,
  aggregate PSNR/SSIM/LPIPS, JSON artifacts, and W&B run metadata.

### MVSplat RE10K

MVSplat is integrated as a side-by-side cost-volume Gaussian splatting path. The
Hydra experiment `mvsplat_re10k` overrides the encoder to `costvolume` and the
decoder to `mvsplat_splatting_cuda`.

- Input: RealEstate10K-style `.torch` chunks plus `index.json`; the downloaded
  smoke subset is 720p and therefore requires `dataset.highres=true`.
- Backbone: `EncoderCostVolume` uses `BackboneMultiview`, a UniMatch-style
  CNN/transformer stack with optional epipolar transformer support.
- Cost volume: `DepthPredictorMultiView` constructs a multi-view cost volume
  over 32 depth candidates, refines it with U-Net blocks, and predicts depths,
  densities, and raw Gaussian channels.
- Gaussian adapter: camera intrinsics/extrinsics unproject per-pixel depth
  samples into 3D. Learned pixel offsets adjust ray samples, density is mapped
  to opacity, and attributes are packed into the shared `Gaussians` dataclass.
- Decoder: `model/decoder=mvsplat_splatting_cuda` uses the legacy
  `diff_gaussian_rasterization` CUDA extension.
- Logged evidence: score JSON files, benchmark and memory JSON, rendered target
  views when enabled, aggregate PSNR/SSIM/LPIPS, and W&B artifacts.

### AnySplat Image Folder

AnySplat is wired as an image-folder inference integration for unordered source
images. It does not require COLMAP cameras or chunk metadata for the smoke path.

- Input: a flat directory of `.png`, `.jpg`, or `.jpeg` images.
- Geometry backbone: `EncoderAnySplat` loads VGGT-1B components from
  `facebook/VGGT-1B`, including the aggregator plus camera and depth or point
  heads.
- Pose and geometry: VGGT predicts pose encodings, intrinsics, depth or point
  maps, and confidence. The integration converts those outputs into dense 3D
  points.
- Gaussian head: `VGGT_DPT_GS_Head` fuses transformer tokens with dense
  geometry to predict opacity and Gaussian feature channels.
- Gaussian adapter: confident points can be filtered or voxelized before
  densities are mapped to opacities and converted into AnySplat Gaussian
  attributes.
- Logged evidence: predicted poses, manifest metadata, source image previews,
  Gaussian counts, opacity and scale statistics, and W&B artifacts. The current
  smoke path reports reconstruction statistics instead of PSNR/SSIM/LPIPS
  because no target-view ground truth adapter is wired for AnySplat.

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

All smoke and test commands should log to W&B unless explicitly debugging
offline. Load the local API key first:

```bash
set -a
source .env
set +a
```

The commands below use the `resplat-tests` W&B project, initialize Weave with
`weave.init("galvin/gaussiansplat test")`, and attach raw result JSON files plus
representative rendered media as W&B artifacts.

To run the wired 5-frame and 10-frame tests for ReSplat COLMAP, MVSplat RE10K,
and AnySplat image-folder inference:

```bash
scripts/run_wandb_frame_tests.sh
```

ReSplat COLMAP smoke test:

```bash
python scripts/infer_colmap.py \
  --model_preset dl3dv_8v_256x448 \
  --data_dir datasets/dl3dv-colmap-demo \
  --scene_name 02267acf6fb98de36173bf4e7db9734c8c421dcb00267e42964dc15134cbb1be \
  --output_dir outputs/smoke/resplat_colmap \
  --num_target 1 \
  --target_selection remaining \
  --max_save_images 1 \
  --wandb-project resplat-tests \
  --wandb-name smoke/resplat-colmap
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
  test.compute_scores=true \
  test.save_image=true \
  test.save_video=false \
  test.save_gt_image=false \
  test.save_input_images=false \
  wandb.project=resplat-tests \
  wandb.name=smoke/mvsplat-re10k \
  wandb.tags=[mvsplat,re10k,smoke] \
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
  --device cuda:0 \
  --wandb-project resplat-tests \
  --wandb-name smoke/anysplat
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
