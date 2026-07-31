# Real VIO backends

ARIADNE evaluates OpenVINS and ORB-SLAM3 out of process so their ROS/C++ dependencies do not
enter the Python package. The setup script clones pinned GPLv3 sources into the ignored
`.cache/ariadne/backends` directory and builds them with Docker.

```bash
applications/ariadne/scripts/setup_vio_backends.sh --build openvins
applications/ariadne/scripts/setup_vio_backends.sh --build orbslam3
PYTHONPATH=applications/ariadne/src .venv/bin/python \
  applications/ariadne/scripts/prepare_openvins_d2_config.py \
  --openvins-root .cache/ariadne/backends/openvins_ws/src/open_vins \
  --output .cache/ariadne/backends/openvins_d2_config
```

Run either backend against D2SLAM sequence 1 and log the evaluation to W&B:

```bash
PYTHONPATH=applications/ariadne/src .venv/bin/python \
  applications/ariadne/scripts/run_real_vio.py --backend openvins --sequence 1 --wandb-mode online
PYTHONPATH=applications/ariadne/src .venv/bin/python \
  applications/ariadne/scripts/run_real_vio.py --backend orbslam3 --sequence 1 --wandb-mode online
```

The report records rigid-aligned ATE, similarity-aligned ATE, the scale correction implied by
ground truth, relative position RMSE, final drift, elapsed time, command, logs, and trajectory path.
Dataset payloads and backend source/build trees remain clone-local. OpenVINS
runs inside its ROS Noetic image; ORB-SLAM3 consumes an exported EuRoC-style stereo/IMU window.
The current passing production baseline is D2SLAM because its TUM-VI camera/IMU model is supported
by both backends. Both production binaries run inside their pinned build images. S3E now has
matched ORB-SLAM3 and OpenVINS execution paths, but neither meets its accuracy gate. MILUV remains
available to Python consumers without a validated production VIO calibration.

S3E ORB-SLAM3 runs use the upstream per-agent calibration through a generated legacy-compatible
settings file. OpenCV matrices are converted to float because the ORB-SLAM3 legacy IMU parser reads
the transform buffer as float without converting a double matrix. Camera messages remain compressed
while they are streamed into a temporary EuRoC window, which is removed after the run by default.
The pinned ORB-SLAM3 stereo EuRoC example is patched at setup time to disable its GUI so visual-only
diagnostics run headlessly.

For OpenVINS, the same bounded EuRoC window is bridged into a temporary ROS1 bag without decoding
the images. The generated model uses raw intrinsics/distortion, the published rectification
rotations, scalar rectified baseline, and inverse left-camera-to-IMU transform. Online camera
intrinsic, extrinsic, and time-offset calibration are disabled to keep the comparison bounded.
S3E begins moving between OpenVINS' default static-initializer regimes, so its configuration
raises `init_imu_thresh` from 0.45 to 1.0.

```bash
for agent in Alpha Bob; do
  PYTHONPATH=applications/ariadne/src .venv/bin/python \
    applications/ariadne/scripts/run_real_vio.py \
    --dataset s3e --backend orbslam3 --agent "$agent" \
    --max-frames 500 --wandb-mode disabled
done
PYTHONPATH=applications/ariadne/src .venv/bin/python \
  applications/ariadne/scripts/run_real_vio.py \
  --dataset s3e --backend orbslam3 --agent Carol --max-frames 500 \
  --auto-stereo-geometry --wandb-mode disabled
```

Run the matched Alpha OpenVINS window with:

```bash
PYTHONPATH=applications/ariadne/src .venv/bin/python \
  applications/ariadne/scripts/run_real_vio.py \
  --dataset s3e --backend openvins --agent Alpha \
  --max-frames 500 --wandb-mode disabled
```

Compare visual-only tracking with the matched stereo-inertial path, or run a controlled
`Camera.bf` scale experiment without changing the source calibration:

```bash
PYTHONPATH=applications/ariadne/src .venv/bin/python \
  applications/ariadne/scripts/run_real_vio.py \
  --dataset s3e --backend orbslam3 --agent Alpha --vio-mode stereo --max-frames 500
PYTHONPATH=applications/ariadne/src .venv/bin/python \
  applications/ariadne/scripts/run_real_vio.py \
  --dataset s3e --backend orbslam3 --agent Alpha \
  --stereo-baseline-scale 1.15 --max-frames 500
PYTHONPATH=applications/ariadne/src .venv/bin/python \
  applications/ariadne/scripts/run_real_vio.py \
  --dataset s3e --backend orbslam3 --agent Alpha \
  --stereo-baseline-scale 1.20 --imu-fast-init --max-frames 1000
PYTHONPATH=applications/ariadne/src .venv/bin/python \
  applications/ariadne/scripts/run_real_vio.py \
  --dataset s3e --backend orbslam3 --agent Alpha \
  --stereo-baseline-scale 1.20 --imu-fast-init \
  --orb-feature-profile high-recall --max-frames 1000
PYTHONPATH=applications/ariadne/src .venv/bin/python \
  applications/ariadne/scripts/run_real_vio.py \
  --dataset s3e --backend orbslam3 --agent Alpha \
  --start-frame 1000 --max-frames 1000 --stereo-baseline-scale 1.20 --imu-fast-init
```

For a controlled reproducibility ablation, `--orb-deterministic-runtime` pins the ORB-SLAM3
container and common numeric libraries to one allowed CPU. It is deliberately opt-in and is part
of the report configuration fingerprint; its elapsed time is not production latency evidence.
Three Alpha runs still span 1.005–1.559 m ATE and 0.980–1.502 m Sim(3), so the 0.521 m Sim(3)
spread is worse than the normally paced 0.347 m spread. CPU affinity and numeric-library thread
limits are therefore insufficient.

`--orb-sync-local-mapping` is the follow-up controlled offline ablation. It waits after every frame
until the local-mapping queue is empty and the mapper accepts keyframes again, and it disables the
dataset's real-time sleep. Three Alpha runs retain 998 poses with zero losses or resets and span
1.393–1.635 m ATE and 1.347–1.522 m Sim(3). Relative to normal pacing, the ATE spread narrows by
18.5% and the Sim(3) spread by 49.7%, but median ATE worsens from 1.341 m to 1.608 m and the
reproducibility gate remains closed. Local-mapper overlap contributes to the variation but does not
explain it. Normal real-time pacing remains the production reference; the next controlled step is
to instrument map/keyframe state and isolate remaining loop-closing or map-state divergence.

The concise result/decision/next-step record for every tested process is maintained in
[VIO and global-pose experiment decision log](vio_global_pose_experiment_log.md).

Nonzero start frames automatically receive a distinct output-directory suffix so cross-window
experiments cannot overwrite the frame-zero baseline.

Evaluator-only changes can be applied to the retained trajectory and logs without scanning the bag,
exporting images, or launching Docker:

```bash
PYTHONPATH=applications/ariadne/src .venv/bin/python \
  applications/ariadne/scripts/run_real_vio.py \
  --dataset s3e --backend orbslam3 --agent Alpha \
  --stereo-baseline-scale 1.15 --reanalyze-existing --wandb-mode disabled
```

## Current correction boundary

The existing D2SLAM evidence shows that local VIO can already fall below 0.1 m ATE on a supported
single-agent sequence: OpenVINS reports 0.066 m and ORB-SLAM3 reports 0.011 m. Those runs use
different replay windows, so they do not establish a backend ranking or multi-agent global pose.
S3E calibration export and matched-window production runs now expose scale, visual tracking, and
long-window reset behavior separately.

The S3E global-pose proxy separates the remaining system problem. Relative cross-agent factors
cannot remove controlled common-mode translation/rotation drift and stop at 6.028 m global ATE.
They are still useful: the densest controlled 12.84-factor/min case reduces relative translation
RMSE from 14.261 m to 0.133 m, a 99.1% improvement. This proves that cross-Wingman vision can
synchronize the fleet but cannot supply a global gauge. Increasing relative-factor cadence is
therefore not a substitute for Alpha's required 5 Hz measured global source. Controlled
translation-noise sensitivity reaches 0.149 m relative RMSE at 0.05 m association noise and
0.268 m at 0.20 m, quantifying the visual-association quality boundary without opening a claim.
The bounded report no longer duplicates per-pose trajectories or scheduler trace rows already
represented by aggregate metrics. Intelligence-node SE(3) rationalization now selects a bounded
per-Wingman demand sweep on the worst node rather than fleet ATE alone. The selected point reaches
0.080 m fleet ATE and 0.0046 rad, with Alpha/Bob/Carol at 0.044/0.085/0.099 m. It uses 48
corrections instead of 69 fixed-cadence corrections, a 30.4% reduction, at 2.23/6.42/4.75
messages/min. The next 1.0 m scheduler envelope demonstrates the limit: fleet ATE remains 0.082 m,
but Carol fails at 0.108 m. Per-Wingman gating therefore prevents a fleet aggregate from masking
the weakest VIO path. The
S3E translation geometry is RTK-backed; the rotation reference is a controlled identity frame,
not measured S3E orientation. The
scheduler now overrides the nominal two-correction cycle capacity twice rather than 19 times at the
previous conservative demand point. With the stronger per-Wingman gate, the target holds only at
0.025 m translation noise and 0.005 rad rotation noise; larger tested perturbations fail at least
one Wingman even where fleet averages look acceptable. At 0.020 rad rotation noise, fleet ATE
already fails at 0.1002 m even though
orientation RMSE remains 0.018 rad. This makes the next optimization priorities explicit:

1. validate S3E camera/IMU calibration and metric observability independently of both production
   estimators, then harden visual relocalization and map continuity;
2. measure real static-object association error and covariance at the Intelligence node;
3. send correction deltas rather than images or full trajectories when a compact correction meets
   the uncertainty and freshness gates;
4. calibrate adaptive scheduling inputs and cycle capacity from real Wingman telemetry;
5. replace staged chordal-rotation/linear-translation rationalization with robust joint
   Lie-algebra SE(3) optimization while retaining orientation-aware correction deadlines.

## Measured S3E VIO boundary

The identical 500-frame production runs fail the 0.1 m gate for every Wingman. Alpha is the best
current run, while Bob and Carol show severe tracking divergence. The report distinguishes backend
process success from target success and records fixed-interval correction recovery:

| Wingman | Aligned ATE | RPE | p95 error | Observed correction recovery |
|---|---:|---:|---:|---|
| Alpha | 2.18 m | 0.044 m | 2.99 m | 0.095 m at 0.5 s, or 120 messages/min |
| Bob | 14.91 m | 0.180 m | 24.76 m | 0.1 s tested cadence still misses |
| Carol | 4.20 m | 0.384 m | 6.97 m | 0.1 s tested cadence still misses |

The backend-independent sensor preflight passes for all three Wingmen: record/header timestamps
agree exactly, stereo p95 skew is no more than 2 ms, median IMU cadence is approximately 10 ms,
and AHRS-derived angular rate agrees with gyro at 0.994–1.001× scale and at least 0.992
correlation. Basic timing, IMU units, and quaternion convention are therefore not the current
failure.

The visual preflight exposes a Carol-specific calibration boundary. Five sampled frames provide
906, 1,976, and 1,896 RANSAC inliers for Alpha, Bob, and Carol. Positive-disparity fractions are
97.6%, 100%, and 0.95%; swapping Carol raises its fraction to 98.9%. The swapped stream retains
image-dependent row error: a robust affine model fitted from five frames varies from roughly
−8.5 to −17.8 px at the image corners. Automatic row correction reduces the measured median from
−13.14 to −0.0025 px without ground truth. Three canonical replicates all have zero lost frames,
but the trajectories are not reproducible: ATE spans 29.00–69.21 m with a 39.46 m median and
37.1% coefficient of variation; Sim(3) ATE spans 9.12–10.58 m, poses span 327–426, resets span
1–4, and fitted scale spans 0.133–0.238×. Ideal correction demand remains 464–491 messages/min.
An earlier 7.81 m trajectory with equivalent geometry is not representative. Intelligence must
relocalize.

A coupled 0.42× baseline lowers ATE to 4.97 m but loses 294 frames, resets four times, and still
has a 4.88 m Sim(3) floor. As on Alpha, one static baseline correction changes estimator dynamics
and does not transfer as metric calibration. A tested reconstruction of the swapped right-camera
extrinsic worsened ATE to 20.50 m and was removed; the published `RIGHT.R/P` is not treated as a
physical raw-camera extrinsic.

Repeated-run evidence is summarized by `summarize_vio_replicates.py`. It rejects mixed
configurations using a canonical SHA-256 fingerprint and requires at least three runs. A global
pose claim requires every replicate to pass the ATE and tracking gates plus bounded ATE, Sim(3),
pose-count, reset, and lost-frame spread; a favorable single run cannot pass.

Three identical Alpha 1,000-frame fast-initialization, 1.20×-baseline runs each retain 998 poses
with zero lost frames and zero resets. Their ATE spans 1.338–1.635 m, Sim(3) ATE spans
1.262–1.609 m, and correction demand spans 69.2–72.8 messages/min. Although ATE coefficient of
variation is 9.7%, the 0.347 m Sim(3) spread exceeds the 0.1 m target, so trajectory
reproducibility and the global-pose claim both fail. Tracking health is repeatable; global
trajectory accuracy is not.

The identical Alpha window is now also executed through OpenVINS. Its default static initializer
misses the transition from stationary to moving, while dynamic initialization produces
rank-deficient covariance candidates. A bounded static threshold change from 0.45 to 1.0
initializes successfully and emits 457 matched poses, but the trajectory diverges to 563.20 m ATE,
6.43 m RPE, and 1,555.13 m final drift. Even similarity alignment leaves 8.72 m ATE and the fitted
metric-scale correction is only 0.0225×. The ideal translation correction envelope would require
539.1 messages/min and peak at 10/s, so the evaluator disables correction scheduling and requests
relocalization. The failure is not confined to ORB-SLAM3; shared calibration, initialization, and
motion observability are the next VIO boundary.

The periodic sweep is deliberately simple. A second, ideal event-triggered envelope applies a
zero-latency translation correction only when measured residual error crosses a swept threshold.
For calibrated 500-frame Alpha it reaches 0.100 m corrected ATE at 55.6 messages/min, 53.7% below
the 120 messages/min periodic result. The first healthy 1,000-frame fast-initialization run reaches
0.094 m at 70.4 messages/min, 76.5% below its 300 messages/min periodic requirement. Across three
identical runs, target-passing load spans 69.2–72.8 messages/min. These are
ground-truth-backed lower-bound
experiments, not deployable predictors: they include 0.1-second reaction intervals, and the longer
replicates peak at up to five corrections in one second.

The evaluator now distinguishes interpolated scoring positions from actual RTK messages. With
native RTK positions kept exact and VIO interpolated to their timestamps. With anchors restricted
to the 99 native Alpha observations (60.06/min), online causal Sim(3) reaches only 0.200–0.244 m
across the three long-run replicates; native SE(3) reaches 0.320–0.334 m. The fit would transmit
58.85 corrections/min with a one/s peak, but none of the runs meets 0.1 m. Native Bob and
automatic-geometry Carol Sim(3) are 1.089 m and 1.798 m. Correction bandwidth is therefore not
the immediate production constraint: S3E's native global-observation cadence and local VIO shape
error are insufficient, so all three live Wingman poses require relocalization.

A past-only segment-hold profile tests whether Intelligence can improve the current pose without
waiting for a future RTK sample. At each measured RTK position it resets translation exactly,
then propagates the intervening VIO displacement with an exponentially weighted average of up to
ten completed segment rotations and log scales. The same policy reduces Alpha ATE by
23.9–28.3% to 0.152–0.175 m with 98.38% coverage and 59.45 predictor updates/min, but none of the
three runs reaches 0.1 m. Bob and Carol reach 1.364 m and 2.309 m; their held-scale plausible
fractions are 0% and 16.1%. This is live-position-capable, past-only evidence, but it is
position-only and claim-ineligible. It shows that reset accuracy is not the remaining limit;
unobserved next-interval VIO shape change is. Worst-replicate cumulative Alpha ATE is 0.007 m at
0.1 s and 0.072 m at 0.2 s, then 0.123 m at 0.5 s and 0.175 m at 1.0 s. Consequently, the fixed
cross-replicate target horizon is 0.2 s and would require at least 300 global observations/min.
Bob begins at 0.163 m and Carol at 0.289 m within 0.1 s, so higher observation cadence alone
cannot rescue their current VIO.

A fixed-lag Intelligence model provides a narrower opportunity. After the next exact RTK endpoint
arrives, it maps relative VIO motion across the completed segment and finalizes past global
positions. Alpha reaches 0.077–0.085 m Sim(3) ATE in all three runs with 98.99% pose coverage,
59.45 segment finalizations/min, 0.539 s mean delay, 0.989 s p95 delay, and 0.997 s maximum delay.
Bob and Carol still miss at 0.420 m and 0.447 m. This can tighten delayed map and object-history
state, but it cannot steer the current Wingman pose and remains position-only, claim-ineligible
sensitivity evidence. Alpha's one-second scale estimates all remain inside 0.5–2.0×, while the
replicated p05–p95 envelope still spans 0.704–1.311×. That is direct evidence of time-varying VIO
deformation. Bob and Carol retain only 6.1% and 9.7% plausible segments, so their endpoint fits
do not qualify even as delayed map recovery.

The adaptive fixed-lag policy uses every native observation to evaluate the current completed
segment, but defers finalization for at most two RTK intervals when the observed transform stays
within fixed 10% log-scale and 0.1 rad rotation-change thresholds. Across the same Alpha
replicates it reaches 0.091–0.097 m, preserving three of three delayed-map target passes while
reducing Intelligence finalizations to 37.0–38.8/min (34.7–37.8%). Worst-replicate timing is
0.917 s mean, 1.890 s p95, and 1.998 s maximum delay. Native RTK ingress remains 60.06/min. Bob
and Carol remain above target at 0.434 m and 0.447 m, so the adaptive path is neither live-pose
capable nor claim eligible.

The resulting Intelligence timing/load boundary is:

| Native-observation policy | Alpha ATE | Bob ATE | Carol ATE | Intelligence updates/min |
|---|---:|---:|---:|---:|
| trailing global online Sim(3) | 0.200–0.244 m | 1.089 m | 1.798 m | 58.85 Alpha corrections |
| past-segment current-pose hold | 0.152–0.175 m | 1.364 m | 2.309 m | 59.45 / 57.73 / 57.05 |
| full-rate fixed lag | 0.077–0.085 m | 0.420 m | 0.447 m | 59.45 / 57.73 / 57.05 |
| adaptive fixed lag | 0.091–0.097 m | 0.434 m | 0.447 m | 37.01–38.83 / 54.23 / 57.05 |

Only Alpha's delayed rows pass 0.1 m. Bob and Carol should relocalize rather than consume nearly
the full native cadence for a correction that remains inaccurate. For Alpha, a 5 Hz global source
could plausibly support the measured 0.2 s live horizon; this is a sensor-rate requirement, not a
claim that interpolating the existing 1 Hz RTK creates new observations.

The correction-capacity gate now preserves the reason for each exclusion instead of overloading
tracking health with target capability. Alpha is tracking-healthy but live-correction-ineligible
because its causal native-RTK path misses 0.1 m and its repeated trajectory is not reproducible;
Bob and Carol are tracking failures. Routing all three to relocalization suppresses 170.03 candidate
corrections/min and a combined peak of three/s from the Wingman correction channel. These are
avoided transmissions, not recovered accuracy: Alpha's delayed-map finalizations remain available
at the Intelligence node, while live corrections stay closed until the causal and reproducibility
gates pass.

That translation-only result is the ground-truth-backed sensitivity envelope for Playground 2.
Its RTK files
contain positions and constant identity quaternion placeholders, not measured platform
orientation. The evaluator marks those samples as position-only and omits orientation and
combined SE(3) correction metrics. The S3E calibration also lacks the RTK antenna lever arm, so
the body-pose-to-RTK position relationship cannot yet be removed from ATE.

The bag's 100 Hz IMU messages do publish smooth AHRS quaternions, so the evaluator also reports a
clearly labeled orientation-consistency proxy after applying the RTK-derived world alignment and a
best-fit constant body-frame alignment. For the best healthy Alpha stereo-inertial run, its
orientation RMSE is 0.032 rad, p95 is 0.054 rad, and rotational RPE is 0.0022 rad. That comparison
is non-independent because ORB-SLAM3 consumes the same IMU. The stereo-only run provides an
independent estimator comparison and is materially worse at 0.299 rad RMSE. Neither comparison is
orientation ground truth, and neither is used to derive correction traffic or full SE(3) accuracy.
The evidence instead localizes the present Alpha limit: IMU fusion stabilizes rotation, while
metric translation and path-shape distortion dominate the missed ATE target.

The evaluator also quantifies the missing RTK antenna lever arm without treating a fitted parameter
as calibration. It alternates rigid body-trajectory alignment with a rotating lever-arm fit,
constrains the offset to at most 1 m, and reports both full-window and first-half-to-second-half
holdout changes. On the best healthy Alpha run, the fit saturates the 1 m bound, leaves 1.27 m
same-window ATE (only 5.0% below the 1.34 m score), and worsens held-out ATE by 1.1%. The
unconstrained solution asks for 6.22 m. A fixed body-to-antenna offset therefore cannot explain the
remaining long-window error. The 500-frame calibrated run sees a larger 19.2% optimistic reduction
but still stops at 0.445 m, so even its best-case sensitivity remains more than four times the
target. Correction-load calculations continue to use the unmodified RTK-scored residual.

A second offline sensitivity sweep fits independent local frames over 0.5, 1, 2, 5, and 10 second
windows. Best Alpha crosses the target with rigid SE(3) at 1 second (0.073 m ATE) but not 2 seconds
(0.122 m). Adding a local scale state extends the passing window to 2 seconds at 0.051 m, while
5 seconds reaches only 0.117 m. The corresponding lower-bound anchor rates are 60/min for SE(3)
and 30/min for Sim(3), and the fitted 5-second scale distribution spans 0.839×–1.190× from p05 to
p95. This is direct evidence of motion-dependent metric deformation rather than one static
baseline multiplier.

Those rates are not scheduler recommendations: each local transform sees the entire future window.
Stereo-only remains above target at 0.103 m even with 0.5-second Sim(3); Bob is 0.125 m at the same
window; Carol reaches 0.093 m but is tracking-unhealthy. Bob and Carol must relocalize before this
optimization is applicable.

The causal trailing-window implementation closes that question. It uses only current and prior
ideal position anchors, requires at least three, assumes zero latency, and leaves the scored ATE
unchanged. Best Alpha passes only at 0.2-second cadence: SE(3) reaches 0.070 m and Sim(3) reaches
0.041 m at 294.8 anchor messages/min. Their p95 output jumps are 0.147 m and 0.154 m, while the
Sim(3) p95 scale change is about 20.2%. At 0.5-second cadence, the tested Sim(3) result is 0.121 m
and fails.

The follow-up separates Intelligence ingress and fitting from Wingman transmission. The balanced
policy evaluates 0.2-second ideal position anchors and holds the last transmitted Sim(3) transform
until its current-pose effect exceeds one shared 0.15 m threshold. Across all three Alpha
replicates, corrected ATE is 0.0906–0.0937 m, Wingman traffic is 75.2–78.9 corrections/min, and
Intelligence ingress is 294.8 anchors/min. Correction holds reach 1.99 seconds at p95 and
3.70 seconds maximum; bursts reach four/s. A radio-minimum comparison uses 0.1-second anchors and a
0.17 m hold, lowering transmission to 68.6–72.2/min while increasing ingress to 469/min and worst
ATE to 0.0987 m.

This is the selected lower-load sensitivity, but it is not a deployment claim: the anchors are
zero-latency evaluation RTK truth, orientation remains unobservable, and the threshold was selected
against evaluation ATE. The configured 120 messages/min average capacity covers Alpha, while its
one-second evaluation period still misses the 0.100-second reaction boundary and its two/s burst
capacity misses the observed four/s peak. Use deadline-triggered wakes; keep Bob and Carol on the
relocalization path. VIO-side motion/initialization improvement remains the route to reducing the
294.8/min Intelligence ingress burden.

Correction application still has independent rotation controls. It smooths an accepted update to
at most 0.1 rad per application and requests a reset/relocalization when a correction exceeds
0.5 rad, instead of silently applying a discontinuity. Those guards are validated by controlled
SE(3) tests; S3E RTK cannot calibrate them without orientation truth.

On Alpha, disabling IMU worsens ATE to 8.73 m, RPE to 1.38 m, and causes a map reset, so inertial
fusion is helping rather than causing the primary error. A controlled 1.15× `Camera.bf` multiplier
reduces the 500-frame stereo-inertial ATE from 2.18 m to 0.56 m and leaves only a 1.007× residual
scale correction, but similarity-aligned ATE remains 0.55 m. Extending that calibrated run to 1,000
frames raises ATE to 2.83 m with three map resets and 46 lost frames. Enabling ORB-SLAM3 fast IMU
initialization on the same window removes every reset and lost frame, improves ATE to 1.76 m and
p95 error to 2.58 m, and emits 998 poses. Moving the predicted residual scale correction into the
baseline (1.20× total) further improves ATE to 1.34 m, p95 to 2.24 m, and final drift to 0.59 m,
while retaining zero resets/lost frames. Its similarity-aligned ATE is still 1.26 m and residual
scale is only 1.017×, so time-varying path-shape distortion is now the dominant translation error.
Phase metrics localize it: first-quarter, middle-half, and
last-quarter ATE are 1.02 m, 1.55 m, and 1.14 m, with peak error at 60.3% of the window. A bounded
±500 ms timing sweep improves ATE by only 1.89%, so a fixed clock offset is not dominant. Raising
the ORB budget from 1,600 to 2,400 features and lowering FAST thresholds increases map points but
regresses ATE to 2.03 m; the balanced profile remains selected. Fast initialization bypasses the
acceleration check and therefore remains an explicit S3E ablation rather than the production
default.

A non-overlapping frames-1000–1999 test rules out a window-specific static calibration. With
identical fast initialization and balanced features, the upstream baseline reaches 8.07 m ATE and
the 1.20× baseline reaches 4.96 m. Scaling remains beneficial, but the scaled run still implies a
1.221× residual scale correction versus 1.017× on frames 0–999. Both runs have zero resets and lost
frames. This window-dependent metric scale points to motion/initialization-dependent VIO error, not
a single physical-baseline constant that should be tuned again.

When a dataset supplies measured orientations, evaluation uses SLERP, the same global rotation as
position alignment, and one best-fit constant body-frame alignment. Playground 2 does not satisfy
that ground-truth contract, so ground-truth orientation metrics remain absent rather than inferred
from identity placeholders. The separately named AHRS proxy retains its independence and
covariance flags so it cannot be mistaken for truth. The lever-arm sensitivity likewise retains
fitted-data, bound-active, holdout, and non-calibration flags. The position correction envelope
still assumes zero network and optimizer latency and is therefore a lower bound on production load.

The exact values are runtime evidence and can change slightly because ORB-SLAM3 executes concurrent
mapping threads. The durable conclusion is stronger: the roughly 6 messages/min proxy cadence cannot
compensate for current production S3E VIO. Intelligence should reject or relocalize an unhealthy
Wingman instead of attempting to saturate the link with pose corrections.
