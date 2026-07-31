# Bootstrap Architecture

The core package is independent of ROS, CUDA, model weights, and the parent model runtimes.
Configuration and common geometry are safe to import in CPU-only processes. Runtime commands
provide deterministic component, exchange, global-scene, MILUV and S3E global-pose, operations,
and end-to-end gates behind one command boundary.

Configuration flows from YAML through Pydantic validation before logging or runtime setup. The
common package owns frame and transform conventions so downstream sensor, mapping, and correction
modules cannot silently reinterpret matrices.

The Phase 1 reference pipeline adds explicit boundaries beneath that bootstrap layer:

1. `ariadne.replay` aligns camera frames and per-agent IMU windows and drops frames outside the
   configured synchronization tolerance.
2. `ariadne.models` defines swappable VIO, geometric-feature, and semantic-embedding contracts.
3. `ariadne.tracking` applies temporal hysteresis before static observations can enter the global
   registry, then associates confirmed observations using geometry and embeddings.
4. `ariadne.optimization` provides legacy translation metrics and a covariance-aware SE(3)
   pose-graph forest with disconnected-component and robust-loop handling.
5. `ariadne.benchmarks.phase1` executes the chain and emits one versioned JSON/W&B report.

`ariadne.benchmarks.global_pose_rationalization` owns the controlled SE(3) graph construction,
error metrics, compact correction payload accounting, and adaptive scheduling used by both dataset
gates. `miluv_global_pose` streams only three mocap and three UWB CSV members from the archive,
using real full-SE(3) mocap truth while avoiding image extraction. `s3e_global_pose` samples the
common time window from Alpha, Bob, and Carol RTK positions and supplies a controlled
identity-frame rotation because S3E publishes no usable orientation truth.

Both gates inject deterministic local odometry drift and test multi-agent recovery, cadence/noise
limits, per-Wingman load, and false-loop rejection. The Intelligence-node rationalizer uses the
maximum-information forest as initialization, chordally averages rotations over all accepted
factors with one fixed gauge per component, then solves accepted weighted translation factors in
one bounded least-squares pass. MILUV validates this path against real position and orientation
truth. Its UWB path uses covariance-separated position-only factors so range-derived positions do
not inject artificial orientation certainty. Independent delayed estimates and their graph cadence
sweep establish the 0.143 m floor of solving each position independently. The joint causal
fixed-lag reference persists transceiver biases, uniformly bounds retained range factors, and emits
only the current state so later measurements cannot revise earlier outputs. A nine-sample window
with one-second solves reaches 0.0930 m; two-second solves miss at 0.1035 m. A separate full-batch
solver reaches 0.0783 m as the non-causal upper bound. Local odometry, orientation, and cross-agent
pose observations remain controlled in both gates, so neither claims production VIO or
visual-association accuracy.

The shared `GlobalPoseClaimEvidence` gate keeps that boundary executable. A position claim requires
production local odometry, independent measured position and cross-agent factors, causal inference,
no evaluation truth in the estimator, and both fleet and per-Wingman target closure. A full-pose
claim additionally requires independent measured orientation and orientation-target closure.
Missing or controlled evidence fails closed while the controlled benchmark can still serve as a
regression test.

The S3E cross-agent cadence sweep also keeps relative and absolute capability separate. Controlled
cross-Wingman factors reduce relative translation RMSE from 14.261 m to 0.133 m at 12.84
factors/min, while absolute ATE remains 6.028 m. These factors can improve fleet consistency but
do not add absolute global information; an external observation or measured global landmark is
required to bridge Alpha's 0.2 s live horizon. Controlled association noise of 0.05 m and 0.20 m
raises relative RMSE to 0.149 m and 0.268 m respectively, while leaving the absolute failure
essentially unchanged. The benchmark artifact stores aggregate metrics and bounded sweeps instead
of redundant per-pose trajectories and scheduler traces, reducing its current payload from
58.1 KB to 18.9 KB even with the four-point scheduler-demand sweep.

`CorrectionCadenceScheduler` turns per-Wingman translation/orientation error, drift, covariance,
freshness, link, queue, and payload estimates into a bounded correction schedule. It uses the
earlier translation or orientation deadline. It may defer safe optional work under load, but
mandatory predicted target breaches override cycle capacity rather than silently violating the
pose-error bound. S3E now selects a bounded demand sweep on both fleet and maximum per-Wingman
error. The 0.75 m internal demand envelope yields 0.080 m fleet ATE and 0.044/0.085/0.099 m for
Alpha/Bob/Carol with 48 corrections, 30.4% fewer than fixed cadence. The 1.0 m point appears safe
at 0.082 m fleet ATE but fails Carol at 0.108 m, so fleet-only selection is explicitly rejected.
The selected schedule has two capacity-override cycles instead of 19 at the previous conservative
demand setting. The S3E controlled trace reports correction count, rate, and bytes separately for
Alpha, Bob, and Carol instead of hiding asymmetric demand behind a fleet average.

The production S3E VIO path streams compressed stereo/IMU windows per Wingman, bridges bounded
windows to temporary ROS1 bags for OpenVINS, interpolates sparse RTK position ground truth, and
measures the correction cadence needed to bound residual translation drift. An implausible fitted
metric scale outside 0.25×–4× marks geometric divergence and disables correction scheduling.
Before launch, a backend-independent sensor contract checks timestamp domains, stereo skew, IMU
cadence/units, and AHRS/gyro consistency. A sampled SIFT/RANSAC gate checks stereo observability and
positive disparity, rejecting reversed camera order before backend compute. The exporter can
perform an explicit payload swap, fit a robust affine row correction from sampled correspondences,
apply it to the bounded right stream, and verify the corrected geometry without ground truth or
retaining decoded images after the run.
Repeated VIO reports are configuration-fingerprinted before aggregation. The reproducibility gate
requires at least three identical-input runs and bounds ATE variation, Sim(3) spread, pose-count
variation, reset/lost-frame stability, tracking health, and target passes before a local trajectory
may support a global-pose claim. Correction capacity uses the worst observed rate, shortest
reaction interval, and largest burst across healthy replicates, so a favorable stochastic run
cannot under-provision the Intelligence node.
Identity quaternions in the outdoor RTK files are marked unavailable rather than treated as an
orientation reference. The 100 Hz IMU/AHRS quaternion is evaluated separately after RTK-derived
world-frame and constant body-frame alignment. It is an independent orientation proxy for
stereo-only runs, but a shared-input consistency proxy for stereo-inertial VIO; it never enables
ground-truth orientation or SE(3) correction-load claims. A bounded alternating rigid-alignment and
lever-arm fit measures how much error a fixed rotating RTK antenna offset could optimistically
explain. It reports both same-window and first-half-to-second-half holdout effects, remains labeled
as fitted sensitivity rather than calibration, and never replaces the scored trajectory.
Piecewise local SE(3)/Sim(3) fits then sweep 0.5–10 second windows to separate frame drift from
time-varying metric deformation. They require at least 90% pose coverage and report the implied
anchor rate, but are deliberately marked non-causal because each transform uses its complete
future window. This creates a lower bound for a future Intelligence-node fixed-lag optimizer
without weakening the causal scheduler or scored VIO gates.
The paired causal sweep fits only trailing ideal position anchors and assumes zero processing or
network latency. It records coverage, anchor/update rate, transform jumps, and scale changes.
Threshold-held transmission additionally separates Intelligence fitting from Wingman traffic:
the balanced replicated Alpha sensitivity uses 294.8 RTK-interpolated scoring anchors/min and
75.2–78.9 corrections/min at
0.0906–0.0937 m ATE. A radio-minimum point uses 469 anchors/min and 68.6–72.2 corrections/min.
Native-observation mode keeps measured RTK positions exact and interpolates VIO to those timestamps.
Alpha's 60.06 native anchors/min yield 0.200–0.244 m online Sim(3) ATE across three runs, so S3E's
1 Hz RTK does not satisfy the live 0.1 m gate. A current-pose variant resets to each measured position and
holds a ten-segment exponentially weighted transform learned only from completed intervals. It
improves Alpha to 0.152–0.175 m at 59.45 updates/min, but all three replicates still fail and
Bob/Carol regress with implausible scales. The held-transform path therefore remains a measured
live observability limit, not a schedulable correction profile. Alpha remains under target only
through a fixed 0.2 s tested horizon, which implies at least 300 global observations/min; Bob and
Carol have no passing horizon. A separate fixed-lag path waits
for the next native endpoint, then maps relative VIO motion across the completed interval. It
finalizes Alpha history at 0.077–0.085 m with 0.539 s mean and 0.997 s maximum delay, but Bob and
Carol remain above target and no full-pose or live-correction claim is opened. Alpha segment
scales remain within 0.5–2.0× but span 0.704× p05 to 1.311× p95 across replicates, explicitly
measuring the time-varying deformation that the online transform cannot absorb.
An adaptive variant coalesces no more than two RTK intervals unless the observed scale or rotation
changes materially. The same policy keeps all three Alpha finalized-map trajectories at
0.091–0.097 m while reducing finalizations to 37.0–38.8/min, with 1.890 s p95 and 1.998 s maximum
delay. It consumes the same roughly 60 native observations/min and remains past-trajectory
evidence, not a live-pose result.
Matched stereo-only, stereo-inertial, metric-scale, and long-window ablations distinguish camera
and IMU contributions from scale, initialization, and path-shape errors. The evaluator derives a
position-only ideal event envelope for S3E and full SE(3) envelopes only
when a dataset publishes valid orientation truth. A capacity assessment uses the observable
envelope's average rate, shortest reaction interval, and one-second burst. It keeps tracking health
separate from live-correction eligibility: tracking failures and healthy-but-inaccurate VIO paths
receive distinct relocalization reasons, and their suppressed traffic remains observable rather
than disappearing from the capacity model. Correction deltas enforce
independent translation and rotation step/reset bounds so a large rotational discontinuity cannot
be hidden behind a small translation.
Fast IMU initialization is exposed only as a controlled S3E mode: it can retain map continuity when
the normal acceleration-gated initializer resets, but its accuracy and safety must be measured
before promotion.
Target failure remains distinct from backend process success. If the fastest tested cadence cannot
recover 0.1 m, the Intelligence node must mark the Wingman for relocalization rather than treating
unbounded correction traffic as a valid schedule.

The exchange reference extends this boundary across the node seam:

1. `ariadne.perception` normalizes and quality-gates a frame, produces a saliency field, and forms
   bounded connected regions.
2. `ariadne.object_state` accepts only confirmed static tracks and retains the highest-priority
   keyframes under explicit bounds.
3. `ariadne.communications` quantizes, compresses, checksums, prioritizes, expires, and transports
   a versioned static-object packet.
4. `ariadne.intelligence` validates, deduplicates, time-orders, and retains observations for global
   association.
5. `ariadne.benchmarks.exchange` executes the full local-to-global seam and emits the data used by
   the status website visuals.

The NumPy model and optimizer implementations are deterministic references, not substitutes for
the production runtimes. External adapters must preserve these boundaries and report results using
the same benchmark metric names.

The Intelligence-to-planning seam is executable through `ariadne.skyla`:

1. `SkylaHandoff` serializes a deterministic `ariadne.skyla.handoff.v1` envelope containing the
   fresh global context revision, mission goals, vehicle health/link state, frontiers, and no-fly
   constraints.
2. `SkylaMissionPlanner` fails closed on degraded or expired context, rejects stale revisions,
   filters unsafe direct routes, and allocates eligible Wingmen by mission-weighted utility.
3. Every `RouteRequest` carries context and mission revisions, a deterministic idempotency key,
   expiry, and `requires_local_safety_validation=True`; global allocation never bypasses local
   collision avoidance or flight-safety authority.
4. The operations benchmark signs the serialized hand-off, replays it to prove idempotency, and
   reports blocked frontiers, excluded vehicles, route count, and payload size.

Global scene updates are applied atomically: malformed, duplicate, or object-conflicting primitive
sets cannot partially mutate the map. `SceneSnapshot` provides a deterministic JSON artifact for
restart recovery, while `GlobalGaussianMap.rollback` restores retained contents as a new monotonic
revision so downstream consumers never observe version numbers moving backward.

Operational evidence uses one bounded `TelemetryCollector` surface. Every exported event is scoped
to a mission and node, configured secret-like fields are redacted before entering the ring buffer,
health includes healthy/degraded/recovering/failed states, and the same snapshot can be persisted
as JSON or exposed as low-cardinality Prometheus text.
