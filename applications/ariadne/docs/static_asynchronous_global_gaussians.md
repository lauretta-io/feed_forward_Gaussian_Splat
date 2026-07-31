# Static asynchronous global Gaussian fusion

This stage combines dense feed-forward Gaussian PLY outputs in a shared spatial frame while making
no cross-Wingman time-alignment assumption. It is intentionally a static-scene first pass:
capture timestamps remain provenance, inputs may arrive out of timestamp order, and the fusion does
not estimate motion, synchronize clocks, or remove dynamic objects.

## Contract

Each contribution supplies:

- a dense Gaussian PLY with positions and `rot_0` through `rot_3` quaternion properties;
- a Wingman/source identifier and an optional original capture timestamp;
- an explicit rigid 4-by-4 local-to-global transform;
- the transform's registration method and whether independent evidence verifies it.

`DenseGaussianContribution` rejects non-rigid transforms. `fuse_static_gaussian_plys` preserves the
input order, transforms Gaussian means, normals, and orientations, leaves scale unchanged under the
rigid transform, removes non-finite or zero-quaternion primitives, and writes one standard binary
little-endian PLY. A sidecar `ariadne.static-asynchronous-global-gaussians.v1` manifest records
source hashes, transforms, timestamps, per-source output ranges, filtering, bounds, and claim state.
Directional spherical-harmonic coefficients are currently preserved rather than rotated into a new
basis. The manifest therefore closes its separate appearance claim when a contribution combines
non-identity rotation with non-zero directional harmonics; geometry remains available for
diagnostic inspection.

Time is not a fusion key. A later contribution can be applied before an earlier one, and unknown
timestamps remain `null`; neither condition blocks the static fusion. Spatial registration remains
mandatory. The manifest closes `global_metric_claim_eligible` whenever any source transform is
unverified.

## Attempts and results — 2026-07-31

### A. Existing-artifact atlas smoke

The format/provenance smoke used the two real dense ReSplat PLY artifacts already available for
DDOS neighbourhoods 105 and 102. They contain 245,760 Gaussians each and the same 62-property
Gaussian schema. The source artifacts do not contain trustworthy capture timestamps or a shared
global pose frame, so the run used an explicit 130 m X-axis atlas offset for neighbourhood 105 and
retained neighbourhood 102 as the atlas origin. Both transforms are marked unverified.

The run produced 491,519 finite Gaussians in about 117 MiB and removed one non-finite primitive
from neighbourhood 105. This proved the PLY, transform, filtering, hashing, and manifest mechanics,
but it was only an atlas: the inputs were not Wingmen and were not co-registered.

### B. Three-Wingman S3E reconstruction and fusion

The next attempt used actual S3E Alpha, Bob, and Carol camera data. Each Wingman supplied an
independently selected 80-frame window; their selected pivot times differ by about 8 to 18 seconds
and the fusion input order is deliberately not timestamp order. No images, trajectories, or clocks
were resampled across Wingmen.

The preparation stage exported the bounded bag windows, matched images to the saved ORB-SLAM3
poses, and fitted each complete local VIO trajectory to S3E position truth with an offline Sim(3).
That fit supplies a common diagnostic spatial frame, but uses evaluation truth and cannot be a
deployment transform. Each scene then ran the cached ReSplat DL3DV eight-view model at 320 by 384,
with uniform context selection, one target render, and zero refinement iterations. The three
61,440-Gaussian PLYs were fused using the pivot transform for their own capture time.

Preparation and inference commands:

```bash
PYTHONPATH=applications/ariadne/src applications/ariadne/.venv/bin/python \
  applications/ariadne/scripts/prepare_s3e_global_splat_inputs.py \
  --s3e-root datasets/ariadne/s3e/S3Ev1 \
  --output outputs/ariadne/s3e-global-gaussian-static \
  --resplat-output outputs/ariadne/s3e-global-gaussian-static/resplat \
  --agent-window Alpha 0 80 outputs/ariadne/real_vio/s3e-alpha/orbslam3-bf-1.2-fast-init-deterministic-repeat-3/f_ariadne.txt \
  --agent-window Bob 150 80 outputs/ariadne/real_vio/s3e-bob/orbslam3/f_ariadne.txt \
  --agent-window Carol 50 80 outputs/ariadne/real_vio/s3e-carol/orbslam3-auto-geometry-repeat-3/f_ariadne.txt

CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/infer_colmap.py \
  --data_dir outputs/ariadne/s3e-global-gaussian-static/scenes --scene_list all \
  --start_frame 0 --frame_distance 80 --images_dir images --sparse_dir sparse/0 \
  --model_preset dl3dv_8v_256x448 --num_refine 0 --num_context 8 --num_target 1 \
  --context_selection uniform --output_dir outputs/ariadne/s3e-global-gaussian-static/resplat \
  --no_save_images --save_ply --no_eval --wandb-mode disabled --no-weave --device cuda
```

Fusion command:

```bash
PYTHONPATH=applications/ariadne/src applications/ariadne/.venv/bin/python \
  applications/ariadne/scripts/fuse_global_gaussians.py \
  --spec outputs/ariadne/s3e-global-gaussian-static/fusion_input.json \
  --output outputs/ariadne/s3e-global-gaussian-static/unified_s3e_global_gaussians.ply \
  --manifest outputs/ariadne/s3e-global-gaussian-static/manifest.json
```

Result:

| Measure | Observed result |
|---|---:|
| Input Gaussians | 184,320, 61,440 per Wingman |
| Output Gaussians | 184,320 |
| Filtered corrupt primitives | 0 |
| Output PLY size | about 44 MiB |
| PLY properties | 62, preserved |
| Quaternion norm after fusion | 0.9999999 to 1.0 |
| Temporal alignment | none |
| Timestamp input order | out of order, preserved |
| Alpha pose-matched frames / alignment | 78 / 1.28 m RMSE, 1.005x scale |
| Bob pose-matched frames / alignment | 74 / 13.31 m RMSE, 4.492x scale |
| Carol pose-matched frames / alignment | 80 / 9.12 m RMSE, 0.133x scale |
| Global registration verified | no |
| Metric global claim eligible | no |
| Directional appearance claim eligible | no; rotated SH coefficients are not rebased |

An independent `plyfile` read confirmed the vertex count, all-finite XYZ coordinates, normalized
quaternions, 62 preserved properties, recorded bounds, and output SHA-256. This establishes that
independently timed Wingman images can reach one dense static PLY through the real ReSplat model and
the asynchronous fusion contract. The extreme Bob and Carol scale/alignment values agree with the
existing VIO failure evidence, so the artifact characterizes the current boundary rather than
hiding it.

## Planned next step

Recover Bob and Carol tracking, then replace the offline truth-fitted transforms with finalized
ARIADNE local-to-global poses evaluated at each contribution's own capture time; do not align or
resample capture times across Wingmen. Only mark a transform verified when its source pose is
claim-eligible and its frame convention, scale, and covariance gates pass. Rotate directional
spherical harmonics into the global basis, then inspect overlap consistency and duplicate geometry
before adding voxel consolidation or dynamic-object handling.

Until those gates pass, the generated S3E PLY is a viewable diagnostic static fusion and an
end-to-end multi-Wingman format/provenance result, not a metric or appearance-valid global scene.
