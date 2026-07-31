# ARIADNE

ARIADNE is an isolated, CPU-testable application package for vision-first distributed UAV
autonomy. It includes the bootstrap and common types plus deterministic reference implementations
for synchronized replay, model interfaces, VIO evaluation, visual features, temporal static
filtering, cross-agent association, and robust translation-graph optimization from the
[build specification](../../documentation/ARIADNE_CODEX_BUILD_SPEC.md).

## Development setup

```bash
cd applications/ariadne
python3.12 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/pytest
```

No command imports model runtimes, initializes CUDA, or downloads artifacts.

## Commands

```bash
ariadne validate-config --config configs/wingman/default.yaml
ariadne wingman run --config configs/wingman/default.yaml
ariadne intelligence run --config configs/intelligence/default.yaml
ariadne simulate --scenario configs/simulation/two_node.yaml
ariadne benchmark --suite smoke
```

Without installing the package, set `PYTHONPATH=src` and invoke `python -m ariadne`.
Runtime artifacts are placed beneath the configured `output_dir`, never in package source.

## Module documentation

- [Architecture](docs/architecture.md)
- [Coordinate frames](docs/coordinate_frames.md)
- [Dataset evaluation](docs/datasets.md)
- [Real VIO backends](docs/real_vio.md)
- [VIO and global-pose experiment decision log](docs/vio_global_pose_experiment_log.md)
- [Static asynchronous global Gaussian fusion](docs/static_asynchronous_global_gaussians.md)
- [Module implementation plans](docs/implementation_plans.md)
- [Common types](src/ariadne/common/README.md)

## Dataset evaluation

```bash
python scripts/download_datasets.py
python scripts/run_dataset_sequence.py --wandb-mode offline
scripts/replicate_ignored_assets.sh --skip-pull --wandb-mode offline
```

Use `--wandb-mode online --wandb-project <project>` to publish scalar metrics and JSON report
artifacts. Raw datasets are never uploaded to W&B.

## Phase 1 model benchmark

Steps 1-5 of the model work order are available as an integrated deterministic benchmark:

```bash
ariadne benchmark --suite phase1 \
  --output outputs/ariadne/phase1/benchmark.json \
  --wandb-mode offline
```

The suite exercises synchronized camera/IMU replay, interchangeable VIO backends, separate
geometric and semantic features, temporal static filtering, cross-agent association, and robust
incremental pose optimization. The included NumPy implementations are reference backends for
interfaces, metrics, and regression testing; production ORB-SLAM3, DPVO, DINO, and GTSAM adapters
must be evaluated through the same contracts.

See [Phase 1 models](docs/phase1_models.md) for metrics and adapter boundaries.

The next reference gate carries a confirmed static observation from image preprocessing through
saliency, region formation, the local object store, compressed mesh uplink, and Intelligence-node
ingest. Runtime saliency defaults to the full official DUTS-trained U²-Net checkpoint; download and
verify it once from the repository root with:

```bash
.venv/bin/python applications/ariadne/scripts/download_u2net.py
```

The checkpoint is stored in the ignored `.cache/ariadne/models/u2net/u2net.pth` path. The
deterministic gradient/contrast backend remains available as the explicit `gradient_contrast`
fallback for offline contract tests.

```bash
ariadne benchmark --suite exchange \
  --output outputs/ariadne/exchange/benchmark.json \
  --wandb-mode offline
```

The global reference gate fuses two registered observations into a versioned Gaussian scene,
generates and bounds a correction delta, and exposes the result through unified context:

```bash
ariadne benchmark --suite global-scene \
  --output outputs/ariadne/global-scene/benchmark.json \
  --wandb-mode offline
```

Dense model-output PLYs use a separate static-asynchronous fusion contract. It requires an explicit
rigid local-to-global transform for every contribution but never requires capture-time alignment:

```bash
PYTHONPATH=src .venv/bin/python scripts/fuse_global_gaussians.py \
  --spec ../../outputs/ariadne/s3e-global-gaussian-static/fusion_input.json \
  --output ../../outputs/ariadne/s3e-global-gaussian-static/unified_s3e_global_gaussians.ply \
  --manifest ../../outputs/ariadne/s3e-global-gaussian-static/manifest.json
```

See [Static asynchronous global Gaussian fusion](docs/static_asynchronous_global_gaussians.md) for
the first real atlas attempt, the fail-closed registration claim, and the pose-derived next step.

The MILUV global-pose gate streams only the three mocap and three UWB CSV members needed from the
3.1 GB archive. It uses real three-UAV position and orientation truth to test the shared SE(3)
rationalizer, while keeping local odometry and cross-agent pose factors controlled:

```bash
ariadne benchmark --suite miluv-global-pose \
  --miluv-archive ../../datasets/ariadne/miluv/archives/default_3_random_0.zip \
  --output ../../outputs/ariadne/miluv-global-pose/benchmark.json \
  --wandb-mode offline
```

With seed 7, a 16.08-second fixed controlled-correction cadence reduces global ATE from 0.163 m to
0.0195 m and orientation RMSE from 0.1069 rad to 0.0069 rad. It sends five corrections to each UAV
(3.73 messages/min each) and reads only 0.147% of the compressed archive. Cross-agent pose factors
without later global anchors stop at 0.133 m ATE. The tested pose target survives 0.05 m
translation and 0.05 rad rotation correction noise.

The real UWB path now tests the measurement boundary directly using the published six-anchor
constellation and two tag moment arms per UAV. A 1.5-second delayed robust position solve reaches
0.182 m ATE; feeding every accepted estimate to the graph improves this to 0.143 m but costs up to
58.2 messages/min and still cannot reach 0.1 m.

The causal bias-aware path jointly optimizes a nine-sample/8.04-second sliding window every
1.005 seconds. Seed 7 reaches 0.0930 m fleet position ATE and 0.0851/0.0986/0.0946 m per UAV,
while thresholded correction traffic is 17.2–23.1 messages/min. Its dense CPU reference solve has
measured 93–346 ms at p95 on this host, inside the 1.005-second deadline. Two-second solves miss
narrowly at 0.1035 m and four-second solves reach
only 0.1277 m, establishing the timing boundary. Fleet ATE remains 0.0926–0.0943 m across seeds
7/17/29, although `ifo002` reaches 0.1016 m on seed 17, so per-Wingman robustness is not yet proven.

A separate robust full-batch upper bound jointly uses 6,709 real anchor/inter-agent range factors
and controlled odometry deltas. It reaches 0.0783 m position ATE, but remains non-causal. Both UWB
paths retain the controlled baseline orientation at 0.1069 rad RMSE. These results validate causal
Intelligence-node position rationalization with controlled odometry/orientation; they do not yet
validate end-to-end VIO or full SE(3).

The benchmark now inventories real factor availability and applies a fail-closed deployment claim
gate. MILUV has raw IMU but no independent local pose or attitude product, so the causal position
target remains a controlled regression pass rather than a production global-pose claim.

The complementary S3E global-pose gate uses all three Playground 2 agents, real trajectory timing
and translation geometry, and calibrated dataset metadata. It injects controlled local drift, then
sweeps Intelligence-node correction cadence and factor noise while measuring global ATE/RPE,
optimization latency, per-node message rate, payload bytes, and false-loop rejection:

```bash
ariadne benchmark --suite s3e-global-pose \
  --s3e-root ../../datasets/ariadne/s3e/S3Ev1 \
  --output ../../outputs/ariadne/s3e-global-pose/benchmark.json \
  --wandb-mode offline
```

With seed 7, controlled translation and rotation drift plus all-accepted-factor SE(3)
rationalization and adaptive per-Wingman scheduling reduce proxy global error from 12.640 m ATE
and 0.294 rad orientation RMSE to 0.080 m and 0.0046 rad. A bounded scheduler-demand sweep selects
the lowest-load point that keeps every Wingman below 0.1 m, rather than accepting fleet-average
ATE alone. The selected 0.75 m demand envelope uses 48 corrections instead of 69, a 30.4%
reduction: Alpha uses 8, Bob 23, and Carol 17, or 2.23, 6.42, and 4.75 messages/min. Their
controlled ATE is 0.044, 0.085, and 0.099 m respectively. The next tested 1.0 m envelope lowers
load to 46 but lets Carol reach 0.108 m even though fleet ATE is only 0.082 m, proving that fleet
averages can hide a weak Wingman.

The per-Wingman gate also tightens the measured vision-correction boundary. The selected schedule
survives 0.025 m translation noise and 0.005 rad rotation noise; larger tested perturbations fail
at least one Wingman even when a fleet aggregate may remain below target. The scheduler uses
predicted error, orientation error, translation/rotation covariance and drift, correction age,
link quality, and queue utilization. It needs only two capacity-override cycles instead of 19 at
the previous 0.05 m demand setting. Translation factors use S3E RTK positions; rotation uses a
controlled identity-frame reference because the files contain identity quaternion placeholders.
This is an optimizer and load regression, not an end-to-end visual localization or real
orientation result.

The cross-Wingman factor cadence sweep separates relative consistency from absolute localization.
At the densest controlled cadence, 12.84 relative factors/min reduce relative translation RMSE
from 14.261 m to 0.133 m, but absolute global ATE remains 6.028 m. Relative visual constraints can
tie the Wingmen together; they cannot create the absolute global information needed to satisfy
Alpha's measured 0.2 s live horizon. A measured global landmark or external observation source is
still required. At the same cadence, raising controlled relative-translation noise from 0.05 m to
0.20 m degrades relative RMSE from 0.149 m to 0.268 m while absolute ATE stays near 6.03 m.
The report now retains aggregate metrics and bounded sweeps but omits redundant per-pose
trajectories and scheduler traces, reducing this artifact from 58.1 KB to 18.9 KB even after adding
the four-point per-Wingman demand sweep.

The emitted claim evidence marks both S3E position and full-pose claims ineligible because the
estimator uses RTK evaluation truth and controlled odometry, cross-agent, and orientation factors.

Run production ORB-SLAM3 on the same bounded 500-frame S3E window for each Wingman:

```bash
for agent in Alpha Bob; do
  PYTHONPATH=src .venv/bin/python scripts/run_real_vio.py \
    --dataset s3e --backend orbslam3 --agent "$agent" \
    --max-frames 500 --wandb-mode disabled
done
PYTHONPATH=src .venv/bin/python scripts/run_real_vio.py \
  --dataset s3e --backend orbslam3 --agent Carol --max-frames 500 \
  --auto-stereo-geometry --wandb-mode disabled
```

The runner also bridges the bounded EuRoC window into a temporary ROS1 bag and generates an
OpenVINS camera/IMU model from the same S3E calibration:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_real_vio.py \
  --dataset s3e --backend openvins --agent Alpha \
  --max-frames 500 --wandb-mode disabled
```

The S3E path streams compressed camera payloads directly into a temporary EuRoC layout and removes
the export after the backend exits. It does not decode the whole ROS2 bag into RGB memory. The
current three-agent runs all fail the 0.1 m ATE gate, so the controlled global-pose result remains
a rationalization proxy rather than an end-to-end capability result.

Every S3E run now performs two cheap, independent preflights before launching a backend. The sensor
contract checks record/header time, stereo skew, IMU cadence and gravity magnitude, quaternion
validity, and AHRS/gyro agreement. Alpha, Bob, and Carol all pass: header error is 0 ms, stereo p95
skew is at most 2 ms, and AHRS/gyro scale is 0.994–1.001×. A five-frame SIFT/RANSAC geometry check
then verifies positive disparity. It finds 97.6% for Alpha and 100% for Bob, but only 0.95% for
Carol; swapping Carol reaches 98.9%. Carol's affine row model varies across the image, so the
automatic correction fits it from five frames without ground truth and reduces median row error
from −13.14 to −0.0025 px. Three identical one-flag replicates all remove lost frames, but expose
severe estimator variance: ATE spans 29.00–69.21 m with a 39.46 m median, Sim(3) ATE spans
9.12–10.58 m, resets span 1–4, and fitted scale spans 0.133–0.238×. Ideal correction demand is
consistently unschedulable at 464–491 messages/min. A configuration-fingerprinted reproducibility
gate rejects the global-pose claim. A coupled 0.42× baseline also loses 294 frames, confirming
that static depth scaling is not transferable. Carol must relocalize.

The matched Alpha OpenVINS run initializes after the S3E-specific static excitation threshold is
raised from 0.45 to 1.0. It then diverges to 563.20 m rigid-aligned ATE, 8.72 m
similarity-aligned ATE, and a 0.0225× fitted scale correction. An ideal translation envelope would
require 539.1 corrections/min and peak at 10/s. The evaluator marks this geometry implausible and
requires relocalization; OpenVINS is evidence against an ORB-specific failure, not a viable S3E
fallback.

A matched Alpha diagnostic shows stereo-only tracking at 8.73 m ATE versus 2.18 m for
stereo-inertial, so IMU fusion is beneficial. A controlled 1.15× stereo-baseline multiplier cuts
the 500-frame result to 0.56 m, but a 1,000-frame run rises to 2.83 m with three map resets. The
fast-IMU-initialization ablation removes all resets/lost frames on that long window. Combining it
with the predicted 1.20× baseline produces 1.34 m ATE in its first run. Three identical runs retain
998 poses with no resets or lost frames, but ATE spans 1.34–1.64 m and similarity-aligned ATE spans
1.26–1.61 m. Alpha is consistently correction-eligible, though its strict trajectory reproducibility
and global-pose claim gates fail. Path shape—not scale—is now the limit. The first run's
first-quarter, middle-half, and last-quarter ATE are
1.02 m, 1.55 m, and 1.14 m; the peak occurs near 60% of the window rather than at shutdown. A
bounded timestamp sweep improves ATE by only 1.89%, ruling out a constant sensor-time offset as the
dominant cause. A 2,400-feature high-recall profile regresses ATE to 2.03 m, so the 1,600-feature
profile remains selected. On non-overlapping frames 1000–1999, the original and 1.20× baselines
reach 8.07 m and 4.96 m ATE; the scaled run still needs a 1.221× residual scale correction, so a
static multiplier is not a transferable calibration. Across the three identical runs, the ideal
position-only event envelope reaches the target at 69.2–72.8 messages/min, with a 0.100-second
minimum reaction interval and a five-correction-per-second maximum peak. The configured
one-second, two-correction Intelligence cycle has enough average throughput but fails reaction and
burst limits. A causal scale-aware follow-up separates Intelligence fitting from Wingman
transmission. The selected balanced sensitivity uses 0.2-second RTK-interpolated scoring anchors
and one shared 0.15 m
transmit hold: all three Alpha replicates reach 0.0906–0.0937 m ATE at 75.2–78.9 corrections/min
while Intelligence ingress falls to 294.8 anchors/min. Holds reach 1.99 seconds at p95 and
3.70 seconds maximum, with four/s bursts. A radio-minimum comparison lowers transmission to
68.6–72.2/min but raises ingress to 469 anchors/min and worst ATE to 0.0987 m. Both remain
ground-truth-derived lower bounds, so their deployment claims are closed.

A native-observation-only follow-up prevents interpolated scoring truth from masquerading as RTK
ingress. Native RTK positions remain exact while VIO is interpolated to their timestamps. The 99
real Alpha observations arrive at 60.06/min; across the three identical runs, online causal
native-anchor Sim(3) stops at 0.200–0.244 m and SE(3) at 0.320–0.334 m, with 58.85
corrections/min and no target passes. Bob and automatic-geometry Carol reach only 1.089 m and
1.798 m Sim(3). All three live production S3E Wingmen therefore remain relocalization cases.
The capacity gate now records why: Alpha is tracking-healthy but live-correction-ineligible,
whereas Bob and Carol fail tracking health. Fail-closed routing suppresses 170.03 candidate
corrections/min and a combined three/s peak instead of sending inaccurate pose deltas; Alpha's
delayed-map rationalization remains available independently.

A causal current-pose follow-up resets position at each measured RTK sample, then propagates
between samples with an exponentially weighted hold of the last ten observed segment transforms.
One fixed policy improves Alpha by 23.9–28.3%, from 0.200–0.244 m to 0.152–0.175 m across the
three replicates. It still produces zero target passes at 59.45 predictor updates/min. Bob and
Carol regress to 1.364 m and 2.309 m, and their held scales fail the plausibility gate. This
quantifies the best simple past-only live opportunity tested: anchor resets help Alpha, but past
segment geometry cannot predict the next second of VIO deformation accurately enough for 0.1 m.
Across all Alpha replicates, cumulative ATE is 0.007 m through 0.1 s and 0.072 m through 0.2 s,
then rises to 0.123 m through 0.5 s and 0.175 m through 1.0 s. The safe tested horizon is only
0.2 s, implying at least 300 global observations/min rather than the available 60.06/min. Bob and
Carol do not pass even the 0.1 s horizon after scale plausibility is enforced.

An Intelligence-only fixed-lag rationalization closes part of that gap without synthesizing
observations. Once the next native RTK endpoint arrives, it maps the intervening relative VIO
segment into the measured endpoint displacement. Alpha's three finalized trajectories reach
0.077–0.085 m Sim(3) ATE at 98.99% coverage and 59.45 finalizations/min, with 0.539 s mean,
0.989 s p95, and 0.997 s maximum delay. Bob and Carol still miss at 0.420 m and 0.447 m. This is
useful for delayed global-map and object-history updates, but it is explicitly not a live Wingman
pose correction or a full-pose deployment claim. Alpha's one-second segment scales all remain
inside a 0.5–2.0× plausibility gate, but their replicated p05–p95 envelope is 0.704–1.311×.
That time-varying deformation explains why a single online Sim(3) does not close the live gap.
Only 6.1% of Bob and 9.7% of Carol segments are plausible, reinforcing relocalization.

An adaptive fixed-lag policy retains every native RTK observation for deciding when to finalize
map history, but coalesces at most two completed intervals unless scale changes by 10% or rotation
changes by 0.1 rad. One fixed policy preserves all three Alpha passes at 0.091–0.097 m while
reducing Intelligence finalizations to 37.0–38.8/min, 34.7–37.8% below the full-rate path. The
tradeoff is 0.917 s worst-replicate mean delay, 1.890 s p95, and 1.998 s maximum. Native RTK
ingress remains about 60/min, and Bob and Carol still miss at 0.434 m and 0.447 m. This is a
delayed-map load reduction only; it does not open a live-pose or deployment claim.

S3E Playground 2 RTK files contain position plus identity quaternion placeholders and
do not publish the RTK antenna lever arm, so real orientation accuracy and a full SE(3) load envelope
are not claimed. The bag's 100 Hz IMU/AHRS quaternion provides a separate consistency diagnostic:
the best Alpha stereo-inertial run differs by 0.032 rad RMSE with 0.0022 rad rotational RPE, while
stereo-only differs by 0.299 rad. The stereo-only comparison is independent of that estimator; the
stereo-inertial comparison shares its IMU input and cannot serve as ground truth. Together with the
translation results, it shows that IMU fusion stabilizes Alpha rotation while path shape and metric
translation remain the limiting errors. A separate sensitivity fit allows at most a 1 m rotating
body-to-RTK offset: the best Alpha run saturates that bound, moves only from 1.34 m to an optimistic
1.27 m ATE, and worsens held-out ATE by 1.1%. Its unconstrained 6.22 m fitted offset is far outside
the conservative 1 m sensitivity bound, so this diagnostic rules out a bounded fixed antenna offset as the dominant
long-window error without modifying the scored ATE. Bob and Carol remain relocalization cases
rather than correction-load candidates.

A non-causal piecewise alignment sweep now bounds what an Intelligence-node fixed-lag optimizer
could recover from that time-varying error. Best Alpha reaches 0.073 m with a fitted local SE(3)
frame every 1 second (60 anchors/min) and 0.051 m with local Sim(3) every 2 seconds
(30 anchors/min); its 5-second window scale spans 0.839–1.190× from p05 to p95. These are offline
fits that use complete future windows. Stereo-only still misses at 0.103 m even with 0.5-second
Sim(3), and Bob misses at
0.125 m; Carol reaches 0.093 m but remains tracking-unhealthy. This leaves Alpha as the only
candidate for causal scale-aware rationalization and keeps Bob/Carol on the relocalization path.

The causal trailing-anchor follow-up removes the apparent load advantage. Best Alpha only passes
at 0.2-second anchors: SE(3) reaches 0.070 m and Sim(3) 0.041 m, both at 294.8 anchors/min. Their
p95 correction jumps are 0.147 m and 0.154 m, and Sim(3) changes scale by about 20.2% at p95.
Because this idealized test still assumes zero latency and RTK-derived anchors, anchor ingress is
not a deployable sensor schedule. Threshold-held transmission nevertheless offers a balanced
replicated point at no more than 78.9 corrections/min and 294.8 anchor fits/min while retaining the
target. Sim(3) correction
should remain experimental until independent global observations replace RTK truth.

Planning, health, 60-second network partition recovery, registry, trust, and deployment gates run
independently or as part of the composed reference system:

```bash
ariadne benchmark --suite operations \
  --output outputs/ariadne/operations/benchmark.json
ariadne benchmark --suite end-to-end \
  --output outputs/ariadne/end-to-end/benchmark.json
```

The operations gate now executes the ARIADNE-to-SKYLA boundary rather than presenting it only as
an architecture proposal. It creates and signs a versioned global-context hand-off, applies
mission priority and no-fly constraints, excludes unhealthy, low-link, or low-battery Wingmen, and
emits idempotent route requests that still require local flight-safety validation.

## Capability status website

From the repository root, regenerate the self-contained local status website from the current
JSON evaluation reports:

```bash
.venv/bin/python scripts/build_ariadne_status_site.py
```

The generated page is `reports/ariadne_status.html`. It summarizes validated capabilities,
reference components, dataset readiness, real VIO results, source paths, and known gaps. The page
also embeds one self-contained 20-frame TUM VI `corridor1` segment selected at ORB-SLAM3
initialization. Those frames are run through the real preprocessing and feature implementations,
the full pretrained U²-Net saliency model, region clustering, and production ORB-SLAM3 poses matched within
60 ms. The 20 Hz source segment spans 0.95 seconds and is replayed at 5 fps for inspection; stages
without meaningful camera output show saved benchmark metrics instead of decorative video.
