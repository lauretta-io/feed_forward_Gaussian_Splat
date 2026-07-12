# Phase 1 Model Pipeline

## Scope

The Phase 1 benchmark makes the first five model work-order steps executable without adding CUDA,
ROS, OpenCV, or GTSAM to the core package. It validates interfaces, data flow, acceptance metrics,
and deterministic replay before external production backends are installed.

## Pipeline

| Stage | Reference implementation | Production target | Primary metrics |
|---|---|---|---|
| Replay | bounded camera/IMU synchronization | ROS1/ROS2 and archive replay | drops, median/p95 sync error |
| VIO | IMU and complementary fusion | ORB-SLAM3, OpenVINS, DPVO | ATE, RPE, drift, uptime |
| Geometric features | gradient and grid patches | ALIKED/SuperPoint plus matcher | match recall, latency |
| Semantic features | histogram and spatial pyramid | DINOv2/DINOv3 | cosine separation, latency |
| Static filtering | temporal evidence smoothing | calibrated temporal classifier | false insertion, confirmation |
| Association | cosine and distance gates | FAISS plus graph assignment | global IDs, merge/split rates |
| Pose graph | robust translation IRLS | GTSAM or Ceres SE(3) | RMSE, rejected constraints |

The reference pose graph intentionally optimizes translations only. It tests incremental graph
construction and robust rejection. A production optimizer must implement full SE(3), covariance
propagation, disconnected subgraphs, and correction deltas.

## Running

```bash
ariadne benchmark --suite phase1 \
  --seed 7 \
  --output outputs/ariadne/phase1/benchmark.json \
  --wandb-mode online \
  --wandb-project gaussiansplat_test \
  --wandb-group ariadne-phase1
```

Raw frames and sensor messages remain local. W&B receives numeric metrics and the generated JSON
report artifact.

## Acceptance Gates

The deterministic reference benchmark fails unless:

- every synthetic frame is synchronized;
- visual-inertial ATE is below IMU-only ATE;
- at least one semantic feature pair separates positive and negative views;
- no dynamic observation is inserted into the static map;
- two Wingmen merge their confirmed landmark into one global object;
- global position RMSE is below 0.25 m; and
- the injected false graph constraint is rejected.

Production comparisons must add real-dataset ATE/RPE, tracking recovery, static/dynamic F1,
association precision/recall, p50/p95 latency, peak memory, and power. A model is not promotable
based only on the synthetic benchmark.

## Adapter Rules

- Keep geometric and semantic models separate.
- Preserve per-agent timestamps and model versions in every output.
- Never insert an `UNKNOWN`, `STATIC_CANDIDATE`, or `DYNAMIC` track into the persistent map.
- Convert model confidence into calibrated covariance before adding graph constraints.
- Keep local VIO continuous and apply global corrections through a separate transform.
- Run every backend on identical replay windows and seeds and publish the JSON artifact with W&B.
