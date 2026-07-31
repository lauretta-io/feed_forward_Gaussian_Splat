# VIO and global-pose experiment decision log

This is the concise decision record for the VIO, correction, and global-pose processes tested in
ARIADNE. It complements the detailed methodology in [Real VIO backends](real_vio.md) and the
dataset-specific evidence in [Dataset evaluation](datasets.md). Results below are measurements from
the checked-out artifacts as of 2026-07-31. The position target is 0.1 m ATE; the full-pose target
also requires no more than 0.05 rad orientation RMSE. A controlled or offline pass is never treated
as a production claim.

Status terms:

- **Production pass**: real backend, real inputs, reproducible, causal, and claim eligible.
- **Controlled pass**: validates an algorithmic boundary but uses controlled or truth-derived
  factors.
- **Diagnostic**: localizes a failure or establishes a lower/upper bound without opening a claim.
- **Rejected**: fails accuracy, tracking, causality, provenance, or reproducibility gates.

## Dataset and runtime foundation

| Process tried | Result | Decision | Planned next step |
|---|---|---|---|
| Active corpus selection | Retained MILUV, D2SLAM, and S3E; removed QDrone because it lacks the multi-agent vision needed by this path. | Keep the smallest corpus that exercises production VIO plus multi-agent fusion. | Add data only when it closes a named observability or regression gap. |
| Streamed S3E EuRoC export | Compressed passthrough reduced a 50-frame Alpha run from about 453 MiB to 63 MiB peak RSS; temporary decoded images are removed. | Retain; this is the canonical low-memory replay path. | Stream directly into a backend-native transport when it avoids the temporary EuRoC filesystem. |
| S3E timestamp/IMU preflight | Alpha, Bob, and Carol pass header time, stereo skew, IMU cadence/gravity, quaternion, and gyro/AHRS checks. | Basic timing, IMU units, and quaternion convention are not the main failure. | Add the same fail-fast contract to deployment capture tooling. |
| S3E stereo observability preflight | Positive disparity is 97.6% Alpha, 100% Bob, and 0.95% Carol; swapping Carol reaches 98.9%. | Camera order is a real Carol-specific defect. | Require preflight success before every production VIO launch. |

## Production VIO and calibration processes

| Process tried | Result | Decision | Planned next step |
|---|---|---|---|
| D2SLAM OpenVINS baseline | 0.066 m ATE and 0.0116 m RPE over 5,853 poses. | Production pass on its supported single-agent sequence; not a backend ranking because the ORB run uses another window. | Run matched windows if a backend comparison becomes decision-critical. |
| D2SLAM ORB-SLAM3 baseline | 0.011 m ATE and 0.0040 m RPE over 397 poses. | Production pass on the bounded D2SLAM window. | Preserve as the real-backend regression baseline. |
| S3E 500-frame ORB-SLAM3 | Alpha/Bob/Carol reach 2.18/14.91/4.20 m ATE; Bob loses 89 frames and Carol loses 312 with 44 resets. | Rejected for global pose; Alpha is the only near-term tracking candidate. | Improve Alpha reproducibility; relocalize Bob/Carol before scheduling corrections. |
| Carol automatic order and affine-row repair | Removes lost frames, but three runs span 29.00–69.21 m ATE, 1–4 resets, and 464–491 ideal corrections/min. | Diagnostic only: visual continuity improved without global accuracy or repeatability. | Validate physical stereo calibration independently; keep Carol on relocalization. |
| Carol 0.42x baseline coupling | ATE falls to 4.97 m but 294 frames are lost and Sim(3) remains 4.88 m. | Rejected; static scale changes estimator dynamics and is not transferable calibration. | Do not tune another scalar baseline without independent geometry evidence. |
| Carol reconstructed right extrinsic | Worsened ATE to 20.50 m. | Removed; published rectification fields are not treated as a raw-camera extrinsic. | Obtain or calibrate a physical raw stereo transform. |
| Alpha stereo-only | 8.73 m ATE, 7.20 m Sim(3), 1.385 m RPE, and one reset. | Rejected; IMU fusion is essential on this motion. | Retain only as the independent orientation comparison. |
| Alpha 1.15x baseline, 500 frames | 0.560 m ATE with healthy tracking. | Useful metric-scale sensitivity, still 5.6x above target. | Test across longer and disjoint windows before considering calibration. |
| Alpha 1.15x baseline, 1,000 frames | 2.83 m ATE, three resets, and 46 lost frames. | Rejected as a transferable calibration. | Treat long-window continuity separately from short-window scale. |
| Alpha fast IMU initialization | Removes long-window resets/lost frames; 1.15x reaches 1.76 m ATE. | Retain as a controlled S3E mode, not a dataset-wide default. | Replace the bypass with a measured initialization policy. |
| Alpha 1.20x baseline plus fast initialization | First run reaches 1.34 m ATE and 0.052 m RPE; all 998 poses survive. | Best healthy long-run ORB configuration, but still far above target. | Improve path-shape stability rather than tuning another static scale. |
| Alpha high-recall ORB features | Improves RPE to 0.043 m but regresses ATE from 1.34 m to 2.03 m. | Rejected; more features do not improve global consistency. | Measure feature geometry/association quality, not raw feature count. |
| Alpha disjoint frames 1000–1999 | Default/1.20x baselines reach 8.07/4.96 m ATE with a residual 1.221x scale. | Rejected; one static multiplier does not generalize across motion windows. | Estimate time-varying deformation or fix the underlying calibration/observability. |
| Matched Alpha OpenVINS | Initializes after raising the static threshold, then reaches 563.20 m ATE, 8.72 m Sim(3), and 539.1 ideal corrections/min. | Rejected; the S3E failure is not confined to ORB-SLAM3. | Validate shared calibration and motion observability before adding a third backend. |
| Alpha repeated ORB runs | Three identical runs keep 998 poses with no resets/losses, but span 1.338–1.635 m ATE and 1.262–1.609 m Sim(3). | Rejected by the 0.347 m Sim(3) spread; tracking is repeatable, trajectory shape is not. | Isolate asynchronous mapping decisions and require a three-run gate for every candidate. |
| Alpha single-CPU/thread-controlled ORB runs | Three runs span 1.005–1.559 m ATE and 0.980–1.502 m Sim(3); Sim(3) spread worsens to 0.521 m while pose/reset/loss counts remain identical. | Diagnostic; numeric-library and multi-core scheduling controls are insufficient and must not be promoted as production timing evidence. | Test explicit local-mapper synchronization or a deterministic offline mapping mode, then compare against normal real-time pacing. |
| Alpha offline local-mapper synchronization | Three runs span 1.393–1.635 m ATE and 1.347–1.522 m Sim(3), with 998 poses, zero losses, and zero resets every run. ATE/Sim(3) spreads narrow 18.5%/49.7% versus normal pacing, but median ATE worsens 19.9% to 1.608 m. | Diagnostic; local-mapper overlap contributes to trajectory variation but is not the root cause, and offline timing is not production evidence. | Instrument keyframe/map state and isolate remaining loop-closing or map-state divergence before another calibration sweep. |
| IMU/AHRS orientation proxy | Best Alpha stereo-inertial is 0.032 rad RMSE; stereo-only is an independent but worse 0.299 rad check. | Rotation appears more stable than translation, but shared-IMU evidence is not ground truth. | Obtain independent orientation truth before opening full-SE(3) claims. |
| RTK lever-arm sensitivity | Bounded fit saturates 1 m, leaves 1.27 m ATE, and worsens holdout by 1.1%; unconstrained fit requests 6.22 m. | A fixed antenna offset is not the dominant long-window error. | Require physical lever-arm calibration; stop fitting it on evaluation truth. |

## Vision correction and Intelligence-node timing processes

| Process tried | Result | Decision | Planned next step |
|---|---|---|---|
| Periodic ideal translation correction | Alpha needs 0.5 s/120 messages-min on the short run and 0.2 s/300 messages-min on the long run; Bob/Carol still miss at 0.1 s. | Lower bound only; periodic traffic is too high and truth-derived. | Prefer event thresholds and measured uncertainty. |
| Ideal event-triggered translation | Alpha reaches 0.1 m at 55.6 messages/min short-window and 69.2–72.8/min across long replicates, with up to five/s bursts. | Better radio lower bound, still zero-latency truth-derived. | Learn a causal error predictor from real telemetry. |
| Offline local SE(3)/Sim(3) fits | Alpha passes at 1 s SE(3) (0.073 m) and 2 s Sim(3) (0.051 m); 5 s misses at 0.117 m. | Non-causal upper bound showing time-varying scale/path deformation. | Convert only the supported short horizon into causal fixed-lag logic. |
| Causal trailing-anchor fit | Alpha passes only with 0.2 s ideal anchors: 0.070 m SE(3), 0.041 m Sim(3), about 295 anchor fits/min. | Removes the apparent low-rate advantage. | Separate high-rate Intelligence ingress from lower-rate Wingman transmission. |
| Threshold-held Sim(3) transmission | Balanced policy reaches 0.0906–0.0937 m at 75.2–78.9 corrections/min; radio-minimum reaches at most 0.0987 m at 68.6–72.2/min but needs 469 truth anchors/min. | Controlled lower bound; fit cadence and radio cadence can differ. | Replace ideal anchors with measured global observations and calibrated uncertainty. |
| Native 1 Hz RTK online Sim(3) | Alpha reaches 0.200–0.244 m, Bob 1.089 m, Carol 1.798 m; no live target pass. | Production live correction rejected; bandwidth is not the immediate limit. | Alpha needs a real global source near 5 Hz or better VIO; Bob/Carol need relocalization. |
| Past-segment current-pose hold | Alpha improves to 0.152–0.175 m; Bob/Carol worsen to 1.364/2.309 m. | Past-only live prediction cannot model the next second of deformation. | Use horizon-specific gating instead of transmitting another held transform. |
| Horizon-resolved live hold | Alpha is below target through 0.2 s (0.072 m) and misses by 0.5 s (0.123 m); Bob/Carol fail within 0.1 s. | Establishes Alpha's 5 Hz global-observation requirement. | Measure a real 5 Hz source; do not synthesize observations by interpolation. |
| Full-rate fixed-lag map finalization | Alpha delayed history reaches 0.077–0.085 m at 59.45 updates/min and at most 0.997 s delay; Bob/Carol reach 0.420/0.447 m. | Controlled delayed-map pass for Alpha, not a live-pose claim. | Use for map/object history only until current-pose causality passes. |
| Adaptive fixed-lag finalization | Alpha remains at 0.091–0.097 m while finalizations fall to 37.0–38.8/min (34.7–37.8% lower); p95 delay is 1.890 s. | Retain as the lower-load delayed-map policy. | Calibrate transform-change thresholds from independent observations. |
| Fail-closed live correction capacity | Alpha is tracking-healthy but live-pose-ineligible; Bob/Carol fail tracking. Relocalization suppresses 170.03 candidate corrections/min and a combined three/s peak. | Retain; avoided traffic is not mislabeled as recovered accuracy. | Measure recovery/re-entry conditions on real Wingman telemetry. |

## Multi-agent global-pose rationalization processes

| Process tried | Result | Decision | Planned next step |
|---|---|---|---|
| Controlled S3E SE(3) graph | Reduces 12.640 m/0.294 rad to 0.080 m/0.0046 rad; worst Wingman is 0.099 m. | Controlled pass; RTK-derived translation and controlled orientation keep the deployment claim closed. | Replace controlled odometry/orientation and truth-derived corrections. |
| S3E per-Wingman adaptive scheduling | 48 corrections versus 69 fixed (30.4% lower); the 46-correction point hides Carol at 0.108 m despite fleet ATE of 0.082 m. | Retain per-node gating; reject fleet-average false economies. | Calibrate demand/covariance from real telemetry. |
| S3E cross-Wingman relative factors | Relative RMSE improves 14.261 to 0.133 m at 12.84 factors/min, but absolute ATE remains 6.028 m. | Relative vision can synchronize the fleet but cannot create a global gauge. | Add measured global landmarks/observations; keep relative and absolute claims separate. |
| S3E correction-noise sweep | All nodes pass only through 0.025 m translation and 0.005 rad rotation noise. | Establishes the current association/correction quality requirement. | Measure real static-object association covariance. |
| Controlled MILUV full-SE(3) graph | Reduces 0.163 m/0.1069 rad to 0.0195 m/0.0069 rad with 15 corrections. | Controlled pass on real truth geometry. | Replace controlled local odometry and relative pose. |
| MILUV independent scalar-UWB corrections | Best graph result is 0.143 m at up to 58.2 messages/min. | Rejected; independently solved ranges do not meet 0.1 m. | Carry bias and trajectory state jointly. |
| MILUV causal fixed-lag UWB | Seed 7 reaches 0.0930 m fleet and 0.0986 m worst-UAV at 17.2–23.1 messages/min; seed 17 leaves one UAV at 0.1016 m. | Strong controlled position result, not robust per-node production closure. | Add production VIO/orientation and sparse marginalization. |
| MILUV full-batch UWB upper bound | Reaches 0.0783 m position ATE but orientation remains 0.1069 rad; non-causal. | Upper bound only. | Use it to validate sparse causal implementations, never as deployment evidence. |
| MILUV adaptive versus fixed correction cadence | Adaptive improves controlled ATE to 0.0109 m but uses 18 corrections versus 15 at fixed 16.08 s. | Keep fixed cadence as the lower-load choice. | Adapt only when telemetry shows a net accuracy/load benefit. |

## ATE and metric progression from original testing to the current evaluation

The web report graphs this progression as two ordered dot-and-range charts on a logarithmic ATE
axis. Lower is better, and the vertical reference is the 0.1 m target. The first chart describes
configuration experiments; the second shows what the current correction layers do to the selected
Alpha trajectory. They are deliberately separate: a 500-frame result is not directly comparable to
a 1,000-frame result, and a delayed controlled map pass is not a live-pose pass.

### Technical summary

- On the original 500-frame Alpha window, changing the stereo baseline from the published value to
  1.15x reduced rigid ATE from 2.180 m to 0.560 m, a 74.3% diagnostic improvement. It did not
  generalize to the longer or disjoint windows.
- On the comparable 1,000-frame window, fast IMU initialization reduced ATE from 2.830 m to
  1.759 m and removed 46 lost frames and three map resets. Raising the baseline scale to 1.20x then
  reduced ATE to 1.338 m, 52.7% below the original long-window result.
- Increasing ORB features improved local RPE by 16.0% but worsened global ATE by 52.0%. This is the
  clearest counterexample to treating local feature/motion quality as global-pose improvement.
- Three normal repeats produced 1.338–1.635 m ATE. The current single-CPU/thread-controlled runs
  improved the median by only 3.6%, to 1.293 m, while widening the ATE range by 86.5% and the
  Sim(3) range by 50.2%. Tracking counts stayed identical, so the current failure is trajectory-shape
  reproducibility rather than frame survival.
- The current native 1 Hz RTK online correction lowers median Sim(3) ATE to 0.209 m but has zero of
  three live target passes. Adaptive fixed-lag finalization reaches 0.0988 m at 38.83 updates/min,
  but it is delayed map history with 1.890 s p95 latency, not current-pose accuracy.
- Offline local-mapper synchronization narrows the normal ATE spread from 0.297 m to 0.242 m and
  Sim(3) spread from 0.347 m to 0.175 m. Its 1.608 m median ATE is 19.9% worse than normal pacing,
  so the ablation rejects mapper overlap as a sufficient explanation while retaining it as a
  measured contributor.

### Ordered VIO configuration progression

| Step | Evaluation scope | ATE / range | Other metrics | Change from the relevant prior baseline | Interpretation and next step |
|---|---|---:|---|---|---|
| 1. Original S3E Alpha | 500 frames, default baseline | 2.180 m | Sim(3) 0.488 m; RPE 0.0443 m; 476 poses; 0 lost; 0 resets | Initial baseline | Establishes that global scale/path error can be large even when local motion and tracking are healthy. |
| 2. Metric-scale sensitivity | Same 500 frames, 1.15x baseline | 0.560 m | Sim(3) 0.549 m; RPE 0.0443 m; 476 poses; 0 lost; 0 resets | ATE −74.3%; RPE effectively unchanged | Baseline scaling changes metric ATE but not the Sim(3) floor; test transfer before treating it as calibration. |
| 3. Original long window | 1,000 frames, 1.15x baseline | 2.830 m | Sim(3) 1.083 m; RPE 0.0549 m; 927 poses; 46 lost; 3 resets | New comparable long-window baseline | The longer run exposes continuity failure hidden by the bounded 500-frame test. |
| 4. Fast IMU initialization | Same 1,000 frames, 1.15x baseline | 1.759 m | Sim(3) 1.255 m; RPE 0.0525 m; 998 poses; 0 lost; 0 resets | ATE −37.8%; all lost frames/resets removed | Retain as a controlled mode, then replace the bypass with a measured initialization policy. |
| 5. Selected balanced profile | Same 1,000 frames, fast init, 1.20x baseline | 1.338 m | Sim(3) 1.262 m; RPE 0.0518 m; 998 poses; 0 lost; 0 resets | ATE −23.9% vs fast init and −52.7% vs long baseline | Best healthy single long-window setup; further scalar tuning is not supported by the disjoint-window failure. |
| 6. High-recall features | Same 1,000 frames, 2,400 features | 2.034 m | Sim(3) 1.873 m; RPE 0.0435 m; 998 poses; 0 lost; 0 resets | ATE +52.0% while RPE −16.0% | Reject; measure association geometry rather than increasing raw feature count. |
| 7. Normal real-time repeats | Same selected profile, 3 runs | 1.338–1.635 m; median 1.341 m | Sim(3) 1.262–1.609 m; median RPE 0.0481 m; 998 poses every run | First reproducibility gate | Tracking is repeatable but trajectory shape is not; isolate asynchronous mapper decisions. |
| 8. Single-CPU controlled runtime | Same profile, one CPU/thread, 3 runs | 1.005–1.559 m; median 1.293 m | Sim(3) 0.980–1.502 m; median RPE 0.0579 m; 998 poses every run | Median ATE −3.6%, but ATE spread +86.5%, Sim(3) spread +50.2%, and RPE +20.3% | CPU/numeric pinning is insufficient. Synchronize or disable asynchronous local mapping, then rerun this exact gate. |
| 9. Mapping-synchronized offline runtime | Same profile, mapper-idle barrier after every frame, no real-time sleep, 3 runs | 1.393–1.635 m; median 1.608 m | Sim(3) 1.347–1.522 m; median RPE 0.0486 m; 998 poses every run | Versus normal pacing: median ATE +19.9%, ATE spread −18.5%, Sim(3) spread −49.7%, and RPE +1.0% | Local mapping is a contributor but not the root cause. Instrument map-state divergence and keep normal pacing as production evidence. |

The 43.2% reduction from the original 1,000-frame ATE to the current controlled median is real for
this window, but the current median is still 16.1 times the target and the run-to-run gate fails.
The best mapping-synchronized run, 1.393 m, is reported as an observed minimum rather than selected
evidence. Normal real-time pacing remains the production reference.

### Current correction-layer evaluation

| Evaluation layer | Three-run ATE / range | Reduction from current raw median | Claim status | Key constraint and next step |
|---|---:|---:|---|---|
| Raw rigid VIO | 1.393–1.635 m; median 1.608 m | Baseline | Controlled offline backend output | Reproducibility fails despite 998 poses, zero lost frames, and zero resets in every run. |
| Whole-trajectory Sim(3) alignment | 1.347–1.522 m; median 1.436 m | 10.7% | Offline diagnostic | Global scale alignment helps but leaves large path-shape deformation. |
| Native 1 Hz RTK online Sim(3) | 0.207–0.217 m; median 0.209 m | 87.0% | Live algorithm on offline VIO, rejected | Zero target passes; obtain a measured source near 5 Hz or improve VIO. |
| Adaptive fixed-lag map finalization | 0.0939–0.0993 m; median 0.0988 m | 93.9% | Controlled delayed-map pass | 36.40–38.83 finalizations/min and 1.890 s p95 delay; use only for finalized history until live causality passes. |

ATE is rigidly aligned position RMSE unless a row explicitly says Sim(3). RPE measures consecutive
local displacement consistency. Ranges are minima and maxima over three identical-run evaluations;
they are not confidence intervals. Percentage changes are descriptive comparisons of the checked-out
artifacts, not causal estimates, and no result using fitted truth, future samples, or delayed history
opens a production global-pose claim.

## Current ordered next steps

### Dense global Gaussian stage

| Process tried | Result | Decision | Planned next step |
|---|---|---|---|
| Existing-artifact static atlas | Combined two real 245,760-Gaussian ReSplat outputs into one 491,519-Gaussian, 62-property PLY; one non-finite source primitive was filtered. Capture times were unavailable. | Format, transform, validation, and provenance smoke passes. The unrelated manual-atlas inputs cannot establish multi-Wingman registration. | Move to actual S3E Wingman camera windows. |
| S3E asynchronous three-Wingman ReSplat | Independently selected 80-frame Alpha/Bob/Carol windows produced three 61,440-Gaussian splats and one 184,320-Gaussian, 44 MiB PLY. Pivot times differ by 8–18 seconds and remain out of order. Offline VIO-to-truth alignment is 1.28/13.31/9.12 m RMSE with 1.005x/4.492x/0.133x scale. | End-to-end multi-Wingman reconstruction and no-time-sync fusion passes as a diagnostic. Truth-fitted transforms, Bob/Carol tracking failure, and unrotated directional SH keep metric and appearance claims closed. | Recover Bob/Carol, use claim-eligible finalized ARIADNE poses at each source capture time, rotate SH into the global basis, then measure overlap and deduplication. |

1. Instrument ORB-SLAM3 keyframe/map state and isolate remaining loop-closing or map-state
   divergence with the same three-replicate Alpha gate; retain normal real-time pacing as the
   production reference.
2. Independently validate S3E raw stereo/IMU calibration and metric observability before tuning
   another backend, baseline scalar, or feature count.
3. Acquire or derive a measured global observation near 5 Hz for Alpha; otherwise improve VIO so
   its verified live horizon exceeds the native one-second RTK interval.
4. Recover Bob and Carol tracking/relocalization before spending correction bandwidth on them.
5. Replace controlled MILUV/S3E odometry and orientation with production VIO factors, then move the
   dense fixed-lag solver to sparse marginalization on target Intelligence hardware.
6. Measure real cross-Wingman association covariance, correction queue pressure, and relocalization
   recovery to calibrate per-node scheduling.
