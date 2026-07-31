# Dataset Evaluation

ARIADNE uses a representative corpus because the complete upstream releases require more storage
than the current workspace can safely provide. The active corpus prioritizes multi-agent,
vision-inertial data that can exercise the ARIADNE perception and global-pose path.

For a concise process-by-process record of every VIO, correction, and rationalization experiment,
including its result and planned next step, see the
[VIO and global-pose experiment decision log](vio_global_pose_experiment_log.md).

## Local corpus

| Dataset | Selected data | Purpose |
|---|---|---|
| MILUV | `default_3_random_0` | Primary three-UAV vision, IMU, UWB, and mocap replay |
| D2SLAM | aligned TUM corridor set | Five-agent stereo/IMU ROS1 bag replay |
| S3E | v1 Playground 2 and v2 Playground 3 | Three-agent visual-inertial, global-pose, and network stress |

QDrone was removed from the active corpus because its single-agent IMU/UWB-only streams cannot
exercise ARIADNE's vision or multi-agent global-pose path. D2SLAM remains as the production VIO
baseline, while MILUV and S3E cover multi-agent fusion.

The registry at `configs/datasets/registry.yaml` records upstream URLs, licenses, expected sizes,
and publisher checksums. Dataset payloads remain under the ignored `datasets/ariadne` tree.

## Metrics

Each adapter emits a typed `DatasetEvaluation` with agent and modality inventories, message or
sample counts, duration, ground-truth coverage, warnings, and dataset-specific metrics. Visual-
inertial datasets calculate nearest camera-to-IMU timestamp errors in milliseconds. The CLI writes
the complete result to JSON and logs numeric metrics plus that JSON report as a W&B artifact.

```bash
ariadne evaluate \
  --dataset miluv \
  --path datasets/ariadne/miluv/archives/default_3_random_0.zip \
  --output outputs/ariadne/miluv.json \
  --wandb-mode online \
  --wandb-project gaussiansplat_test
```

Run the MILUV full-SE(3) global-pose regression without extracting its image payload:

```bash
ariadne benchmark \
  --suite miluv-global-pose \
  --miluv-archive datasets/ariadne/miluv/archives/default_3_random_0.zip \
  --output outputs/ariadne/miluv-global-pose/benchmark.json
```

The benchmark streams six CSV members, 4.74 MB compressed or 0.147% of the 3.1 GB archive, and
samples 81 poses per UAV across the 80.4-second common mocap window. Real position and orientation
truth validate the common rationalizer: seed 7 reduces 0.163 m ATE and 0.1069 rad orientation RMSE
to 0.0195 m and 0.0069 rad. The selected 16.08-second cadence uses 15 corrections total, five per
UAV or 3.73 messages/min each. Cross-agent factors alone stop at 0.133 m ATE, so a global reference
is still required. The current adaptive policy reaches 0.0109 m but uses 18 corrections and is not
a load improvement over fixed cadence.

MILUV also bounds the measurement layer. Corrections with 0.05 m translation or 0.05 rad rotation
noise still pass the 0.1 m/0.05 rad target in the controlled graph, while the archive's 6,751
deduplicated UWB ranges have 0.239 m RMSE and 0.600 m p95 absolute error against mocap. The
measurement implementation uses the official
[six-anchor constellation](https://github.com/decargroup/miluv/blob/main/config/uwb/anchors.yaml)
and [two-tag-per-UAV geometry](https://github.com/decargroup/miluv/blob/main/config/uwb/tags.yaml).

The causal bounded-window estimator accepts 84.8% of sampled positions and reaches 0.182 m position
ATE at the best tested 0.5-second processing-latency point. Applying every accepted position-only
factor reaches 0.143 m graph ATE at a one-second cadence, but costs 41.0–58.2 messages/min per
Wingman and none of the tested cadences reaches 0.1 m. This establishes the limit of independently
solved scalar-range corrections.

The joint causal fixed-lag reference carries learned transceiver biases across a nine-sample,
8.04-second window and uses no measurements newer than each emitted pose. One-second solves reach
0.0930 m fleet ATE and 0.0851/0.0986/0.0946 m per UAV on seed 7. Its event-triggered 0.05 m
position-delta envelope sends 28/31/23 corrections, or 20.9/23.1/17.2 messages/min. The dense CPU
solve has measured 93–346 ms at p95 on the current host, inside its 1.005-second deadline. Reducing
Intelligence
solve frequency to two seconds yields 0.1035 m and four seconds yields 0.1277 m, so the target
depends on approximately one-second state updates even though Wingman messages can be sparser.
Fleet ATE spans 0.0926–0.0943 m across seeds 7/17/29; `ifo002` reaches 0.1016 m on seed 17, leaving
the per-Wingman gate just outside robust closure.

The archive evidence inventory confirms that each UAV has raw PX4/camera IMU streams but no
independent VIO, odometry, pose, trajectory, or attitude product. Consequently the fixed-lag
position regression passes its controlled seed-7 target but is explicitly ineligible for a
production position or full-pose claim.

The bounded full-batch upper bound jointly optimizes all 243 sampled positions and per-transceiver
range biases using 4,033 real anchor factors, 2,676 real inter-agent factors, controlled odometry
deltas, and fixed controlled orientations for the tag lever arms. It converges to 0.0783 m position
ATE (0.0681/0.0798/0.0859 m per UAV); seeds 7, 17, and 29 span only 0.0781–0.0792 m. It does not
meet the full-pose target because orientation remains 0.1069 rad RMSE. A post-batch 0.05 m
position-frame residual envelope requires 19/27/18 messages by UAV, or 13.4–20.1 messages/min.
These rates exclude range ingress. Local odometry, orientation, and relative-pose factors remain
deterministic perturbations. The next valid test replaces controlled odometry and orientation with
production VIO factors and moves the dense fixed-lag reference to sparse marginalization on target
Intelligence hardware.

Run the S3E global-pose regression over real Playground 2 trajectory geometry:

```bash
ariadne benchmark \
  --suite s3e-global-pose \
  --s3e-root datasets/ariadne/s3e/S3Ev1 \
  --output outputs/ariadne/s3e-global-pose/benchmark.json
```

This benchmark injects deterministic local-odometry drift, derives noisy cross-agent and global
translation constraints from S3E RTK positions, adds a controlled identity-frame rotation
reference, and reports global ATE/RPE before and after fusion.
It sweeps correction intervals from no global correction through every sampled epoch and measures
the resulting Intelligence-node solve time and per-Wingman correction traffic.

The seed-7 proxy now injects controlled rotational as well as translational drift. It reduces
12.640 m global ATE and 0.294 rad orientation RMSE to 0.080 m and 0.0046 rad through all-factor
SE(3) rationalization. The adaptive scheduler selects the lowest-load bounded demand point that
keeps every node below the target: 48 corrections instead of the fixed cadence's 69, a 30.4%
reduction. Alpha, Bob, and Carol receive 8, 23, and 17 corrections, corresponding to 2.23, 6.42,
and 4.75 compact messages/min, and finish at 0.044, 0.085, and 0.099 m ATE. Cross-agent relative
factors alone stop at 6.028 m because they cannot observe common-mode pose drift.

At the next tested 1.0 m scheduler demand envelope, only 46 corrections are needed and fleet ATE
still reads 0.082 m, but Carol reaches 0.108 m. The per-Wingman gate rejects that false economy.
The 0.1 m/0.05 rad target remains satisfied at 0.025 m tested translation-correction noise and
0.005 rad tested rotation-correction noise; larger tested perturbations fail at least one Wingman.
Translation factors are RTK-position-derived, while rotation factors use the controlled reference
because S3E does not publish real orientations here. These figures isolate pose rationalization
and load; they are not end-to-end scores for visual association, VIO, or real rotation accuracy.

The S3E proxy is also explicitly claim-ineligible: controlled odometry and cross-agent factors are
combined with evaluation-truth-derived RTK corrections and controlled orientation. Its pass status
therefore means the optimizer regression passed, not that production global pose passed.

The production S3E VIO runner streams a selected Wingman window from the ROS2 SQLite bag without
materializing decoded stereo frames in memory. A 50-frame Alpha export reduced measured peak RSS
from approximately 453 MiB through `S3EReplaySource` to 63 MiB through compressed passthrough.
Temporary EuRoC images are deleted after each run unless `--keep-export` is requested.

Before backend launch, a bounded sensor contract verifies exact message/header timing, stereo
synchronization, IMU rate and gravity magnitude, quaternion validity, and gyro/AHRS agreement.
All three Playground 2 Wingmen pass, which moves the failure boundary away from basic timing and
IMU units. A five-frame SIFT/RANSAC contract separately checks stereo disparity direction. Alpha
and Bob are positive-disparity streams; Carol is reversed and must use `--swap-stereo-input`.
Carol's reversed order and remaining image-dependent vertical residual are detected and repaired
with `--auto-stereo-geometry`, without ground truth or a retained decoded dataset copy.

Identical 500-frame ORB-SLAM3 runs now cover Alpha, Bob, and Carol. None meets 0.1 m aligned ATE.
The evaluator interpolates the sparse S3E ground truth, records p95/max error, lost frames, map
resets, and positive error growth, then sweeps translation-correction intervals from 0.1 to 10
seconds. Alpha can recover 0.1 m only at a very high tested correction rate; Bob and Carol remain
outside the tested recovery envelope. These results make local visual tracking and calibration the
next constraint, rather than Intelligence-node pose-graph solve latency.

The Carol order/affine-row correction improves visual continuity but not global accuracy. Three
identical replicates all have zero lost frames, yet ATE spans 29.00–69.21 m, Sim(3) ATE spans
9.12–10.58 m, resets span 1–4, and ideal load spans 464–491 messages/min. A configuration-
fingerprinted reproducibility gate rejects the trajectory. A coupled 0.42× baseline reintroduces
294 lost frames. This separates visual observability from estimator stability and metric
calibration, keeping Carol on the relocalization path.

A matched 500-frame Alpha OpenVINS path now uses the same streamed image/IMU window through a
temporary ROS1 bag. Its generated calibration combines the published raw intrinsics, distortion,
rectification rotations, scalar baseline, and left-camera-to-IMU transform; calibration learning
is disabled during the run. Raising the static initializer excitation threshold from 0.45 to 1.0
captures the observed motion transition, but the trajectory diverges to 563.20 m rigid ATE and
8.72 m Sim(3) ATE with a 0.0225× fitted scale. It would demand 539.1 ideal corrections/min with a
10/s peak, so scale plausibility rejects correction scheduling and sends the Wingman to
relocalization. This shifts the next test from backend substitution to independent S3E
calibration/observability validation.

Matched Alpha ablations narrow that constraint. Stereo-only tracking reaches 8.73 m ATE and one map
reset, while stereo-inertial reaches 2.18 m. Scaling `Camera.bf` by the observed 1.15× metric bias
reduces the 500-frame ATE to 0.56 m, but the 1,000-frame result degrades to 2.83 m with three map
resets and 46 lost frames. Fast IMU initialization on the same long window removes all resets/lost
frames and lowers ATE to 1.76 m, but bypasses the normal acceleration gate and remains an explicit
experiment rather than a dataset-wide default. A predicted 1.20× baseline point improves the same
healthy long run to 1.34 m ATE. Three identical runs retain all 998 poses with no resets or lost
frames, but span 1.34–1.64 m ATE and 1.26–1.61 m similarity-aligned ATE. Tracking health is stable;
the strict trajectory-reproducibility and global-pose claim gates still fail, establishing path
shape as the remaining floor.

Across the three healthy 1,000-frame Alpha replicates, an ideal event-triggered translation
envelope reaches the target at 69.2–72.8 messages/min rather than the periodic sweep's
300 messages/min. It needs 0.100-second reactions and peaks at five corrections/s. The configured
120 messages/min average capacity is sufficient, but the one-second scheduler tick and
two-correction burst are not.

A causal threshold-held Sim(3) sensitivity selects a balanced 0.2-second
RTK-interpolated scoring-anchor cadence and one shared
0.15 m transmit threshold. All three Alpha replicates reach 0.0906–0.0937 m while Intelligence
ingress falls to 294.8 anchors/min and Wingman traffic remains 75.2–78.9 corrections/min. Bursts
reach four/s, p95 holds 1.99 seconds, and maximum holds 3.70 seconds. A radio-minimum comparison
uses 469 anchors/min to reach 68.6–72.2 corrections/min. This separates Intelligence compute/ingress
from radio traffic; both policies remain ground-truth-derived and claim-ineligible.

The native-observation control selects exact measured positions only at timestamps present in the
1 Hz RTK files and interpolates VIO—not truth—to those timestamps. Alpha provides 99 anchors, or
60.06/min, and the three online causal Sim(3) results span 0.200–0.244 m with 58.85
corrections/min; native SE(3) spans 0.320–0.334 m. Bob and automatic-geometry Carol reach
1.089 m and 1.798 m Sim(3), respectively. No live Wingman pose passes. This closes the apparent
ingress gap: interpolation is valid for scoring VIO samples, but those interpolated truth positions
are not independent global observations available to Intelligence.

The Intelligence node can still improve delayed map state. A fixed-lag native-endpoint model waits
for the next RTK observation, maps the intervening relative VIO displacement into that measured
segment, and then finalizes the past trajectory. All three Alpha runs reach 0.077–0.085 m Sim(3)
ATE at 98.99% coverage and 59.45 finalizations/min. Mean, p95, and maximum delay are 0.539,
0.989, and 0.997 seconds. Bob and Carol remain at 0.420 m and 0.447 m. The metric is causal when
emitted but uses an observation later than each finalized pose timestamp; it therefore supports
global-map/object-history optimization, not live correction scheduling or a deployment claim.
All Alpha segment scales fall inside the 0.5–2.0× gate, with a replicated 0.704–1.311× p05–p95
envelope. Bob's 2.200–15.895× and Carol's 0.152–9.561× p05–p95 envelopes leave only 6.1% and
9.7% plausible segments. Their low fixed-lag scores are therefore rejected as tracking divergence.
The error is transient rather than monotonic: first-quarter, middle-half, and last-quarter ATE are
1.02 m, 1.55 m, and 1.14 m, with peak error near 60% of the window. A ±500 ms timing sweep improves
ATE by only 1.89%, and a 2,400-feature high-recall profile regresses ATE to 2.03 m. On independent
frames 1000–1999, the original calibration reaches 8.07 m and the 1.20× baseline reaches 4.96 m
with a further 1.221× residual scale correction. Baseline scaling helps both windows but does not
generalize as a complete calibration.

Playground 2 ground-truth files are 1 Hz RTK positions with identity quaternion placeholders.
The evaluator now marks orientation unavailable and omits real orientation and combined SE(3)
load metrics. Calibration supplies camera-to-IMU and camera-to-LiDAR transforms but no RTK antenna
lever arm, so residual ATE may include an unmodeled position-frame offset. The bag does contain
smooth 100 Hz IMU/AHRS quaternions, but their zero covariance and shared use by stereo-inertial VIO
make them a consistency proxy rather than orientation truth. Against that proxy, the best healthy
Alpha stereo-inertial run is 0.032 rad RMSE and 0.0022 rad rotational RPE; the independent
stereo-only comparison is 0.299 rad RMSE. This supports rotation stabilization by IMU fusion but
does not make full SE(3) load observable. Real rotation validation requires another reference
source.

A bounded RTK lever-arm sensitivity test fits at most a 1 m rotating offset while preserving the
original ATE as the score. For the best healthy Alpha long run, the fit saturates that bound and
reduces same-window ATE only from 1.34 m to 1.27 m; applying the first-half fit to the second half
worsens ATE by 1.1%. The unconstrained solution requests a 6.22 m lever arm. This is strong evidence
that the missing published lever arm is not the dominant long-window failure and that the next VIO
work should target motion-dependent path distortion. Bob and Carol require relocalization because
their reset/lost frame evidence invalidates correction envelopes.

The piecewise alignment sweep quantifies that distortion. On best Alpha, local rigid SE(3) fits
reach 0.073 m ATE at 1 second but fail at 2 seconds with 0.122 m; allowing local scale reaches
0.051 m at 2 seconds and fails at 5 seconds with 0.117 m. The implied optimistic loads are 60 and
30 anchors/min respectively, while 5-second scale corrections vary from 0.839× to 1.190×
(p05–p95). Stereo-only cannot reach target even with 0.5-second Sim(3) at 0.103 m, and Bob is
0.125 m; Carol reaches 0.093 m but is invalid for scheduling because tracking is unhealthy. The
fits use complete future windows, so they motivate a causal fixed-lag Alpha experiment rather than
superseding the measured event-triggered load.

That causal sensitivity now fits trailing anchors only. Best Alpha reaches 0.070 m with SE(3) and
0.041 m with Sim(3) at 0.2-second interpolated scoring anchors. Threshold-held transmission shows
that fit cadence is not radio cadence: the selected replicated 0.2-second-anchor sensitivity needs
294.8 fitted anchor samples/min
and 75.2–78.9 Wingman corrections/min to stay below target. It is an optimistic capacity lower
bound, not a substitute for measured global observations. At exact native RTK timestamps, the
online causal Sim(3) approach misses the target at 0.200–0.244 m. The fixed-lag endpoint model
reaches 0.077–0.085 m only by delaying finalized history by up to one second.

Run the complete representative sequence with:

```bash
python applications/ariadne/scripts/run_dataset_sequence.py \
  --wandb-mode online \
  --wandb-project gaussiansplat_test
```

W&B receives metrics and reports only. It does not receive raw images, ROS bags, archives, or
credentials.

## Replicating another clone

The shell entry point below updates the current branch from `origin`, installs evaluation
dependencies, downloads and verifies every file in the representative corpus, extracts the
D2SLAM archive, and regenerates ignored evaluation outputs:

```bash
applications/ariadne/scripts/replicate_ignored_assets.sh \
  --wandb-mode online \
  --wandb-project gaussiansplat_test
```

Use `--skip-pull` when testing local uncommitted changes. The script is resumable and safe to run
again: complete files are revalidated by expected size and publisher checksum. Clone-local `.env`
credentials are loaded when present but are never copied or generated.
