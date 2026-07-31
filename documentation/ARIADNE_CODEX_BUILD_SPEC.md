# ARIADNE Codex Build Specification

**Repository:** `lauretta-io/feed_forward_Gaussian_Splat`  
**Target location:** `applications/ariadne/`  
**Status:** Initial engineering specification  
**Primary objective:** Build a modular, testable, vision-first and passive-sensor-based distributed autonomy system for small UAVs using Wingman Nodes and an Intelligence Node.

---

## 1. Codex Operating Instructions

Codex should treat this document as the implementation contract for ARIADNE.

### 1.1 Required behavior

1. Implement one module at a time in the order defined in this document.
2. Do not silently change message schemas, coordinate conventions or module responsibilities.
3. Every module must include:
   - typed interfaces;
   - configuration schema;
   - deterministic unit tests;
   - integration tests where applicable;
   - structured logging;
   - metrics;
   - failure handling;
   - a README with run commands;
   - a minimal executable example.
4. Prefer small, composable modules over monolithic pipelines.
5. Separate research model wrappers from production runtime code.
6. Keep all ARIADNE-specific code beneath `applications/ariadne/` unless a component is demonstrably reusable by the parent Feed Forward Gaussian Splat repository.
7. Reusable code should be placed under `src/ariadne_common/` only after its interface has stabilized.
8. Do not vendor model weights or datasets into Git.
9. All model downloads must be explicit, checksummed and license-documented.
10. CPU-only smoke paths must exist even when production execution requires CUDA.

### 1.2 Definition of Done for every module

A module is complete only when:

- public interfaces are typed and documented;
- unit tests pass;
- integration tests pass;
- configuration validation exists;
- expected latency and memory are benchmarked;
- failure modes are tested;
- logs and metrics are emitted;
- the module can be invoked from CLI;
- an example configuration is provided;
- generated artifacts are written outside source directories;
- no hardcoded absolute paths or credentials exist.

---

# 2. Proposed Repository Layout

```text
feed_forward_Gaussian_Splat/
├── applications/
│   └── ariadne/
│       ├── README.md
│       ├── pyproject.toml
│       ├── configs/
│       │   ├── common/
│       │   ├── wingman/
│       │   ├── intelligence/
│       │   ├── simulation/
│       │   └── missions/
│       ├── docs/
│       │   ├── architecture.md
│       │   ├── coordinate_frames.md
│       │   ├── interfaces.md
│       │   ├── model_registry.md
│       │   ├── test_plan.md
│       │   └── deployment.md
│       ├── schemas/
│       │   ├── observation.proto
│       │   ├── embedding.proto
│       │   ├── pose.proto
│       │   ├── correction.proto
│       │   ├── telemetry.proto
│       │   └── mission.proto
│       ├── src/ariadne/
│       │   ├── common/
│       │   ├── sensors/
│       │   ├── perception/
│       │   ├── embeddings/
│       │   ├── vio/
│       │   ├── saliency/
│       │   ├── tracking/
│       │   ├── object_state/
│       │   ├── communications/
│       │   ├── wingman/
│       │   ├── intelligence/
│       │   ├── splatting/
│       │   ├── pose_correction/
│       │   ├── planning/
│       │   ├── telemetry/
│       │   └── cli/
│       ├── tests/
│       │   ├── unit/
│       │   ├── integration/
│       │   ├── simulation/
│       │   ├── hardware/
│       │   └── regression/
│       ├── benchmarks/
│       ├── scripts/
│       ├── docker/
│       └── examples/
├── src/
├── scripts/
├── config/
└── third_party/
```

---

# 3. Canonical System Flow

## 3.1 Wingman Node

```text
Vision + passive sensor input
    → timestamp alignment
    → image preprocessing
    → visual feature and object embedding generation
    → VIO using visual features + IMU
    → local drift correction and local pose estimate
    → saliency detection
    → saliency clustering
    → object tracking and motion classification
    → static/dynamic separation
    → package static object embeddings + local pose + uncertainty
    → transmit to Intelligence Node
    → receive global correction delta
    → apply correction to local pose and local map frame
```

## 3.2 Intelligence Node

```text
Receive static object embeddings and local pose estimates
    → validate, time-order and deduplicate observations
    → associate objects across Wingman Nodes
    → run feed-forward Gaussian-splat reconstruction/correction
    → maintain global object and scene representation
    → optimize global pose consistency
    → compute per-Wingman correction deltas
    → return deltas and confidence to Wingman Nodes
    → expose unified context to swarm planning
```

---

# 4. Module 00 — Project Bootstrap and Build System

## Purpose

Create the ARIADNE application skeleton without disturbing existing ReSplat, MVSplat, AnySplat or OpenSplat paths.

## Required contents

- `applications/ariadne/pyproject.toml`
- package namespace `ariadne`
- CLI entry point `ariadne`
- base configuration loader
- structured logging
- test runner
- linting and formatting
- Docker development image
- optional ROS 2 adapter package, kept separate from the core runtime

## Initial implementation

- Python 3.12 to match the parent repository.
- Pydantic v2 for configuration and message validation.
- Hydra/OmegaConf for hierarchical runtime configuration, consistent with the parent repository where practical.
- `pytest`, `ruff`, `mypy` and `pre-commit`.
- Logging through Python `logging` with JSON output option.

## Commands

```bash
ariadne validate-config --config configs/wingman/default.yaml
ariadne wingman run --config configs/wingman/default.yaml
ariadne intelligence run --config configs/intelligence/default.yaml
ariadne simulate --scenario configs/simulation/two_node.yaml
ariadne benchmark --suite smoke
```

## Tests

- import package in clean virtual environment;
- validate all example configs;
- run CPU-only CLI smoke command;
- verify no import side effects download models or initialize CUDA.

## Exit criteria

- CI passes on CPU-only GitHub runner;
- CLI help works;
- all configuration files validate;
- parent repository scripts continue to function.

---

# 5. Module 01 — Common Types, Time and Coordinate Frames

## Purpose

Define non-negotiable conventions shared by every component.

## Required contents

### Coordinate frames

- `camera_<id>`: camera optical frame;
- `imu`: IMU body frame;
- `body`: vehicle body frame;
- `local_<wingman_id>`: locally initialized navigation frame;
- `global`: Intelligence Node global frame;
- `object_<uuid>`: optional persistent object frame.

### Transform convention

- homogeneous 4×4 transforms;
- explicit source and destination frame names;
- quaternions stored as `x, y, z, w`;
- positions in meters;
- timestamps in integer nanoseconds from a monotonic clock plus optional UTC metadata;
- camera convention adapters must be explicit because the parent ReSplat code uses OpenCV camera-to-world conventions.

### Core types

- `Timestamp`
- `FrameId`
- `TransformSE3`
- `PoseEstimate`
- `PoseCovariance`
- `CameraCalibration`
- `ImuCalibration`
- `SensorHealth`
- `ModelVersion`

## Implementation requirements

- immutable dataclasses or Pydantic models;
- transform composition and inversion helpers;
- serialization tests;
- no implicit Euler-angle conversions in core code;
- all uncertainty matrices validated for shape and finite values.

## Tests

- transform round-trip;
- quaternion normalization;
- frame mismatch rejection;
- timestamp ordering;
- covariance validation;
- OpenCV/ReSplat convention conversion tests.

## Evaluation

- transform numerical error below `1e-6` in double precision test path;
- serialization round-trip lossless for integer identifiers and timestamps.

---

# 6. Module 02 — Sensor Ingestion and Synchronization

## Purpose

Acquire vision and passive sensor data and produce synchronized observations.

## Inputs

- monocular or stereo RGB cameras;
- IMU: accelerometer and gyroscope;
- optional passive sensors:
  - magnetometer;
  - barometer;
  - GNSS when available, never required;
  - RF link metadata;
  - optional passive acoustic features.

## Outputs

`SynchronizedObservation` containing:

- one or more image frames;
- camera calibration IDs;
- IMU sample window;
- timestamps and synchronization quality;
- sensor health flags;
- optional sensor measurements.

## Initial development

1. Build file/replay adapters first.
2. Add ROS bag and EuRoC/TartanAir adapters.
3. Add live GStreamer/V4L2 camera adapter.
4. Add hardware-specific adapters only after the replay path is stable.

## Requirements

- bounded queues;
- configurable drop policy;
- support approximate and exact synchronization;
- preserve original sensor timestamps;
- expose synchronization error;
- never block the flight-control thread;
- configurable image resolution and frame-rate throttling.

## Tests

- out-of-order IMU samples;
- missing frames;
- duplicate timestamps;
- camera stalls;
- IMU burst loss;
- clock offset and drift simulation;
- replay determinism.

## Evaluation metrics

- synchronization error, milliseconds;
- dropped frame rate;
- queue age;
- ingest latency;
- CPU use;
- memory growth over 30 minutes.

## Initial target

- median synchronization error below 2 ms for hardware-synchronized stereo;
- bounded memory under all drop conditions;
- no deadlock after sensor disconnect/reconnect.

---

# 7. Module 03 — Image Preprocessing and Quality Control

## Purpose

Prepare images for embedding, VIO, saliency and Gaussian-splat pipelines.

## Functions

- decode and color conversion;
- resize and crop;
- lens undistortion;
- rectification for stereo;
- normalization per model;
- blur, exposure and occlusion scoring;
- optional rolling-shutter correction metadata;
- keyframe quality scoring.

## Outputs

`PreprocessedFrame` containing:

- image tensor;
- valid image mask;
- original and processed resolution;
- transform between pixel spaces;
- quality scores;
- calibration reference.

## Requirements

- zero-copy path where supported;
- CPU reference implementation;
- CUDA implementation where beneficial;
- no hidden model-specific normalization outside model adapters;
- deterministic preprocessing for evaluation.

## Tests

- calibration fixtures;
- known rectification pairs;
- image-size edge cases;
- corrupted images;
- under/overexposure;
- CPU/GPU numerical comparison.

## Evaluation

- preprocessing latency per camera;
- CPU/GPU memory copies;
- image quality rejection accuracy;
- downstream VIO and embedding impact.

---

# 8. Module 04 — Visual Feature and Embedding Generation

## Purpose

Generate compact visual representations for local motion estimation, saliency analysis and cross-node object association.

## Initial models

### Primary baseline

- **DINOv2 ViT-S/14** for general visual embeddings.

### Secondary candidates

- DINOv2 ViT-B/14;
- SigLIP 2 image encoder;
- SAM 2 image features;
- lightweight MobileViT or EfficientViT variant for low-SWaP tests;
- learned local features such as SuperPoint or ALIKED for VIO-specific matching.

## Important separation

The system should expose two feature products:

1. **Local geometric features** for VIO and frame-to-frame matching.
2. **Semantic/object embeddings** for saliency, clustering and cross-node association.

These may use different models. Do not force one embedding to serve both roles until evaluation supports it.

## Inputs

- preprocessed image;
- optional region proposals;
- optional masks;
- model configuration.

## Outputs

- dense feature map;
- global image embedding;
- region/object embeddings;
- feature scale and coordinate mapping;
- model version;
- confidence/quality metadata.

## Development stages

1. PyTorch reference adapter.
2. Batched inference.
3. ONNX export where supported.
4. TensorRT implementation for Jetson.
5. Hailo and Sakura compilation feasibility study for supported submodels.
6. quantization calibration path.

## Tests

- deterministic output for pinned model and input;
- embedding shape and normalization;
- empty region handling;
- batching consistency;
- export parity;
- precision comparison across FP32, FP16, BF16 and INT8 where supported.

## Evaluation datasets

- Mapillary;
- MegaDepth;
- Google Landmarks v2 subset;
- VisDrone;
- custom low-altitude UAV imagery;
- repeated-route dataset with day, angle and altitude changes.

## Evaluation metrics

- Recall@1, Recall@5 and Recall@10;
- cosine separation between same and different objects;
- viewpoint robustness;
- illumination robustness;
- embedding drift over time;
- latency;
- memory;
- energy per frame;
- transmitted bytes per accepted static object.

## Initial acceptance target

- reproducible baseline report for at least DINOv2-S and one lightweight alternative;
- explicit recommendation for Wingman deployment;
- no model selected solely on benchmark accuracy without power and latency data.

---

# 9. Module 05 — Visual-Inertial Odometry

## Purpose

Estimate each Wingman Node's local motion using visual features and IMU measurements, and correct short-term local drift.

## Initial implementations to evaluate

- **ORB-SLAM3 stereo-inertial** as a classical baseline;
- **VINS-Fusion** as a second classical baseline;
- **DPVO** or a comparable learned visual odometry baseline;
- optional OpenVINS for estimator robustness and covariance access.

## Architecture

The ARIADNE VIO interface must allow implementations to be swapped without changing downstream modules.

## Inputs

- synchronized image pair or monocular frame;
- IMU sample window;
- camera and IMU calibration;
- optional visual feature map;
- latest global correction state.

## Outputs

- local pose;
- velocity;
- pose covariance or quality proxy;
- tracked feature count;
- estimator status;
- local keyframe reference.

## Requirements

- support stereo-inertial first;
- monocular-inertial optional;
- recover from brief visual loss;
- detect estimator divergence;
- preserve a local continuous frame while global corrections are applied through a separate transform;
- never directly overwrite internal estimator state with abrupt global deltas unless the selected estimator supports it.

## Tests

- EuRoC sequences;
- TartanAir sequences;
- synthetic IMU bias;
- dropped frames;
- motion blur;
- low texture;
- repeated patterns;
- static camera;
- aggressive rotation;
- correction-delta injection.

## Evaluation metrics

- Absolute Trajectory Error;
- Relative Pose Error;
- translational drift percentage;
- rotational drift per meter;
- tracking uptime;
- recovery time;
- compute latency;
- memory and power.

## Initial acceptance target

- select one production baseline based on accuracy and Jetson suitability;
- retain at least one alternate implementation behind the same interface;
- generate trajectory and drift reports automatically.

---

# 10. Module 06 — Saliency Detection

## Purpose

Identify visually significant regions worth tracking, classifying and potentially communicating.

## Initial methods

1. Feature-map saliency derived from DINOv2 attention or feature variance.
2. Spectral residual or classical saliency as a CPU baseline.
3. SAM/SAM2 mask proposals for object-aware saliency.
4. Motion saliency from optical flow residuals.

## Inputs

- preprocessed frame;
- dense feature map;
- optional optical flow;
- optional semantic detections;
- quality mask.

## Outputs

- saliency map;
- ranked candidate regions;
- region masks or boxes;
- saliency confidence;
- reason codes such as semantic, motion, novelty or persistence.

## Requirements

- configurable top-k and threshold;
- temporal smoothing;
- mask invalid image regions;
- support sparse operation when compute is constrained;
- avoid treating camera motion as object saliency.

## Tests

- empty scene;
- uniform texture;
- camera-only motion;
- moving object;
- sudden lighting change;
- repeated background pattern;
- occlusion.

## Evaluation

- precision/recall against annotated salient regions;
- stability across adjacent frames;
- downstream static-object retention;
- false saliency caused by ego motion;
- latency and memory.

---

# 11. Module 07 — Saliency Clustering and Region Formation

## Purpose

Convert per-frame saliency into coherent candidate objects or scene regions.

## Initial methods

- connected components over thresholded saliency;
- DBSCAN/HDBSCAN in joint image-feature space;
- mask merging from SAM proposals;
- temporal region association.

## Inputs

- saliency map;
- feature map;
- candidate masks;
- optical flow;
- camera calibration.

## Outputs

`SalientRegion` objects with:

- region ID;
- mask and bounding box;
- centroid;
- embedding;
- saliency statistics;
- motion statistics;
- temporal age;
- provenance.

## Requirements

- reject tiny unstable regions;
- merge overlapping proposals;
- preserve separate nearby objects when embeddings differ;
- stable IDs across short sequences where possible.

## Tests

- adjacent objects;
- fragmented masks;
- large background region;
- object entering and leaving frame;
- rapid camera motion;
- cluster count limits.

## Evaluation

- intersection-over-union with object masks;
- over-segmentation and under-segmentation rate;
- temporal ID stability;
- compute cost.

---

# 12. Module 08 — Tracking and Static/Dynamic Classification

## Purpose

Track candidate regions and determine whether each represents a static environmental feature or a dynamic object.

## Initial methods

- ByteTrack or OC-SORT for box-level tracking;
- mask tracking where masks are available;
- ego-motion-compensated optical flow;
- triangulated position consistency for stereo;
- temporal embedding similarity.

## Classification states

- `UNKNOWN`
- `STATIC_CANDIDATE`
- `STATIC_CONFIRMED`
- `DYNAMIC`
- `LOST`
- `REJECTED`

## Inputs

- salient regions;
- local VIO pose;
- optical flow;
- depth/stereo estimates;
- previous tracks.

## Outputs

`ObjectTrack` containing:

- stable local ID;
- object embedding history;
- masks/boxes;
- estimated 3D position;
- motion residual;
- static probability;
- uncertainty;
- observation count.

## Requirements

- compensate for ego motion before motion classification;
- require temporal persistence before static confirmation;
- support reclassification;
- dynamic objects must not be added to the persistent static map by default;
- maintain bounded track history.

## Tests

- stationary object with moving camera;
- moving object with stationary camera;
- same-direction object and camera motion;
- temporary occlusion;
- object becoming stationary;
- false detection track.

## Evaluation datasets

- KITTI tracking;
- BDD100K;
- TAO;
- VisDrone video;
- custom drone flyby and orbit sequences.

## Metrics

- static/dynamic F1;
- ID switches;
- track recall;
- false static insertion rate;
- time to confirm static;
- memory per active track.

---

# 13. Module 09 — Local Object State and Keyframe Store

## Purpose

Maintain the Wingman Node's bounded local memory of static objects, embeddings, keyframes and uncertainty.

## Data products

- local object table;
- keyframe metadata;
- local observation graph;
- embedding cache;
- transmission state;
- correction history.

## Requirements

- bounded by configurable memory budget;
- evict low-value data first;
- preserve objects needed for active correction;
- store model and calibration versions;
- support deterministic replay;
- never assume network connectivity.

## Prioritization score

Consider:

- static confidence;
- novelty;
- observation count;
- geometric coverage;
- embedding quality;
- time since last transmission;
- Intelligence Node request priority.

## Tests

- cache pressure;
- repeated observations;
- model-version change;
- restart and recovery;
- stale correction;
- corruption handling.

## Evaluation

- memory bound compliance;
- object retention utility;
- replay consistency;
- bandwidth reduction.

---

# 14. Module 10 — Wingman Uplink Packaging

## Purpose

Convert selected static object observations into compact, versioned messages for transmission.

## Message contents

- Wingman ID;
- observation ID;
- timestamp;
- local pose and covariance;
- camera calibration version;
- object local ID;
- semantic/object embedding;
- optional compressed geometry or keypoints;
- object mask summary;
- static probability;
- quality score;
- model version;
- message priority.

## Requirements

- protobuf schema;
- optional quantized embedding transport;
- compression independent of transport;
- message checksums;
- backward-compatible schema versioning;
- idempotent observation IDs;
- configurable privacy mode that prevents raw image inclusion.

## Compression experiments

- FP16 embeddings;
- INT8 affine quantization;
- product quantization;
- PCA projection;
- top-k token/feature selection.

## Tests

- schema compatibility;
- packet truncation;
- duplicate packet handling;
- quantization parity;
- maximum message size;
- malformed input rejection.

## Evaluation

- bytes per object;
- association accuracy after compression;
- encode/decode latency;
- packet-loss resilience.

---

# 15. Module 11 — Mesh Communications and Transport

## Purpose

Provide resilient device-to-device and device-to-Intelligence-Node communication.

## Initial transports

- QUIC for primary IP transport;
- gRPC over QUIC/TCP for development where practical;
- UDP datagrams for telemetry and low-latency updates;
- ROS 2 DDS adapter for lab testing, not the canonical over-air protocol.

## Message classes

1. heartbeats;
2. health and capability;
3. object observations;
4. correction deltas;
5. map/model updates;
6. task assignments;
7. acknowledgements and retransmission requests.

## Requirements

- priority queues;
- bandwidth budgets;
- encryption and authentication;
- peer identity;
- store-and-forward option;
- reconnect and resume;
- duplicate suppression;
- clock offset estimation;
- configurable reliability per message type;
- no unbounded retransmission.

## Network impairment tests

- latency from 10 ms to 2 s;
- packet loss from 0% to 50%;
- bandwidth from 50 kbps to 100 Mbps;
- reordering;
- disconnection and reconnection;
- asymmetric link;
- Intelligence Node handover.

## Evaluation

- useful observation throughput;
- correction round-trip latency;
- queue age;
- packet overhead;
- recovery time;
- behavior under partition.

---

# 16. Module 12 — Intelligence Node Ingest and Observation Registry

## Purpose

Receive, validate, order and persist observations from all Wingman Nodes.

## Functions

- schema validation;
- authentication and authorization;
- deduplication;
- timestamp normalization;
- model-version compatibility check;
- quality filtering;
- per-node sequence tracking;
- short-term observation buffering;
- audit logging.

## Outputs

- normalized static-object observations;
- node status;
- ingest metrics;
- rejected-message reason codes.

## Requirements

- idempotent ingest;
- tolerate out-of-order delivery;
- preserve original message;
- separate raw observation log from derived state;
- bounded hot storage with configurable persistence backend.

## Initial storage

- SQLite or DuckDB for single-machine development;
- append-only Parquet logs for experiment replay;
- pluggable production database interface.

## Tests

- duplicate observations;
- incompatible schema;
- unknown model version;
- stale messages;
- clock skew;
- node restart;
- database interruption.

## Evaluation

- ingest messages per second;
- end-to-end ingest latency;
- storage growth;
- replay fidelity;
- rejection accuracy.

---

# 17. Module 13 — Cross-Wingman Object Association

## Purpose

Determine whether observations from different Wingman Nodes refer to the same persistent static object or scene region.

## Inputs

- object embeddings;
- local poses and uncertainty;
- timestamps;
- object geometry;
- current global transform estimates;
- semantic labels where available.

## Initial association approach

1. embedding nearest-neighbor candidate generation;
2. geometric gating using transformed position uncertainty;
3. temporal plausibility check;
4. multi-view consistency score;
5. graph-based assignment;
6. persistent global object ID update.

## Initial tools

- FAISS for embedding search;
- Hungarian assignment for small batches;
- min-cost flow or graph clustering for larger windows;
- robust geometric verification when keypoints are available.

## Requirements

- retain multiple hypotheses when uncertainty is high;
- never merge solely on semantic class;
- allow split and merge correction;
- record association evidence;
- expose confidence calibration.

## Tests

- same object, different viewpoint;
- similar repeated objects;
- wrong local pose;
- delayed observation;
- embedding model upgrade;
- object moved between missions.

## Evaluation datasets

- custom multi-drone repeated-object dataset;
- Mapillary sequences;
- MegaDepth pairs;
- synthetic repeated-structure scenes.

## Metrics

- association precision and recall;
- false merge rate;
- false split rate;
- global-ID stability;
- association latency;
- sensitivity to pose error.

---

# 18. Module 14 — Feed-Forward Gaussian Splat Adapter

## Purpose

Integrate ARIADNE with the parent repository's feed-forward Gaussian-splat models without copying or tightly coupling application logic.

## Initial backends

- ReSplat as primary baseline;
- MVSplat side-by-side runtime;
- AnySplat inference integration;
- optional CPU OpenSplat rendering for smoke/debug only.

## Adapter responsibilities

- translate ARIADNE observations to model input batches;
- resolve camera intrinsics/extrinsics;
- select keyframes/views;
- call the configured backend;
- capture predicted Gaussians, depth and confidence where available;
- expose standardized reconstruction output;
- preserve backend-specific diagnostics;
- isolate model-specific dependencies.

## Standard output

`GaussianSceneUpdate`:

- Gaussian means;
- scales;
- rotations;
- opacities;
- colors/features;
- source observations;
- global/local frame reference;
- confidence;
- backend name and version;
- runtime metrics.

## Requirements

- backend plugin interface;
- batch-size and view-count limits configurable;
- explicit GPU memory accounting;
- fail gracefully on OOM;
- support offline replay before real-time execution;
- no assumption that every backend supports recurrent updates.

## Tests

- tiny synthetic scene;
- known COLMAP demo;
- backend input/output shape tests;
- camera convention tests;
- OOM and invalid calibration;
- deterministic pinned-checkpoint regression.

## Evaluation datasets

- RealEstate10K;
- DL3DV;
- ACID;
- Tanks and Temples;
- Mip-NeRF 360;
- custom aerial orbit and flythrough sequences.

## Metrics

- PSNR;
- SSIM;
- LPIPS;
- depth error;
- pose-correction utility;
- Gaussian count;
- runtime;
- peak memory;
- update bandwidth;
- temporal consistency.

## Initial acceptance target

- ReSplat adapter runs the parent repository's smoke/demo path from ARIADNE;
- produces standardized outputs;
- records reproducible evaluation results;
- does not modify ReSplat internals unless required by a separately reviewed core change.

---

# 19. Module 15 — Global Gaussian Scene and Object Map

## Purpose

Maintain the Intelligence Node's persistent, globally consistent 3D representation.

## Representation layers

1. Gaussian scene representation;
2. persistent global object registry;
3. observation provenance graph;
4. pose graph;
5. map version history.

## Requirements

- incremental updates;
- region-based access;
- object-to-Gaussian linkage;
- confidence and uncertainty;
- pruning and compaction;
- dynamic-object exclusion by default;
- map snapshots;
- rollback to last valid version;
- support multiple concurrent local frames before convergence.

## Update strategy

- ingest new static observations;
- associate with global objects;
- select reconstruction window;
- produce Gaussian update;
- validate update quality;
- merge or reject;
- increment map version;
- update pose graph constraints.

## Tests

- duplicate scene update;
- conflicting observations;
- corrupted update;
- map rollback;
- object deletion/movement;
- bounded-memory pruning;
- simultaneous Wingman inputs.

## Evaluation

- reconstruction quality;
- global consistency;
- map growth;
- update latency;
- rollback correctness;
- object-map accuracy.

---

# 20. Module 16 — Global Pose Optimization

## Purpose

Estimate globally consistent Wingman poses from object associations, Gaussian reconstruction evidence and local VIO estimates.

## Initial methods

- factor graph using GTSAM or Ceres;
- robust loss functions for false associations;
- local VIO odometry factors;
- cross-node object correspondence factors;
- optional image alignment/reprojection factors;
- optional Gaussian rendering-error factors.

## Inputs

- local pose trajectories;
- pose covariance;
- object association constraints;
- reconstruction alignment evidence;
- current global map.

## Outputs

- optimized global transforms for each Wingman local frame;
- per-node correction delta;
- confidence and covariance;
- rejected constraint list;
- optimization diagnostics.

## Requirements

- incremental optimization;
- robust to bad constraints;
- maintain local continuity;
- cap correction magnitude per update or flag reset-required state;
- preserve correction history;
- support disconnected subgraphs.

## Tests

- synthetic known trajectory;
- loop closure;
- false association;
- single disconnected node;
- high VIO drift;
- delayed constraints;
- optimizer non-convergence.

## Evaluation

- global ATE and RPE;
- correction accuracy;
- convergence iterations;
- false-constraint tolerance;
- compute latency;
- stability of repeated updates.

---

# 21. Module 17 — Correction Delta Generation and Wingman Application

## Purpose

Return safe, bounded global corrections to each Wingman Node and apply them without destabilizing local navigation.

## Correction message

- Wingman ID;
- source map version;
- applicable local pose timestamp/range;
- translation delta;
- rotation delta;
- optional scale correction;
- covariance/confidence;
- expiry;
- application mode.

## Application modes

- frame-transform update;
- gradual blend;
- hard reset request;
- advisory only.

## Requirements

- default to updating `T_global_local`, not rewriting VIO history;
- reject stale or incompatible corrections;
- smooth large corrections;
- expose corrected and uncorrected pose;
- persist correction lineage;
- detect oscillating corrections.

## Tests

- small correction;
- large correction;
- stale map version;
- out-of-order correction;
- repeated correction;
- oscillation;
- temporary disconnect.

## Evaluation

- reduction in global drift;
- local trajectory continuity;
- time to converge;
- correction round-trip latency;
- rejection correctness.

---

# 22. Module 18 — Unified Context and Scene Graph

## Purpose

Expose a machine-usable unified context for downstream planning rather than only a rendered Gaussian map.

## Scene graph contents

- global objects and regions;
- semantic labels;
- static/dynamic state;
- geometry and occupancy;
- traversability hints;
- observation confidence;
- source Wingman provenance;
- freshness;
- mission relevance.

## Requirements

- query by region, type, time and confidence;
- maintain stable IDs;
- separate observed facts from inferred attributes;
- support uncertain and conflicting hypotheses;
- emit change events;
- allow planning to operate on degraded context.

## Initial APIs

- Python query API;
- protobuf/gRPC service;
- optional ROS 2 message adapter;
- map snapshot export.

## Tests

- object update;
- conflicting semantic labels;
- stale object;
- moved object;
- partial map;
- query performance.

## Evaluation

- query latency;
- scene-graph correctness;
- object persistence;
- memory use;
- planner utility.

---

# 23. Module 19 — Swarm Planning and Task Allocation

## Purpose

Use unified context to coordinate small UAVs while retaining safe local autonomy.

## Initial scope

The first implementation should not attempt end-to-end learned swarm control. Begin with deterministic planning and allocation.

## Initial methods

- frontier or information-gain exploration;
- auction-based task allocation;
- cost-based assignment using distance, battery, sensor capability and communication quality;
- coverage planning;
- static obstacle-aware route requests;
- replanning after node failure.

## Inputs

- mission goals;
- unified context;
- node state and capabilities;
- link quality;
- battery and health;
- no-fly zones and constraints.

## Outputs

- per-node task assignments;
- priorities;
- task dependencies;
- expected completion criteria;
- reassignment events.

## Requirements

- Intelligence Node plans globally;
- Wingman retains collision avoidance and flight safety authority;
- planning must work with incomplete context;
- task messages idempotent and versioned;
- safe fallback on Intelligence Node loss.

## Tests

- two-node coverage;
- node failure;
- poor communications;
- duplicate task;
- conflicting goals;
- stale map;
- battery-constrained assignment.

## Evaluation

- mission completion time;
- area coverage;
- redundant observation rate;
- communication cost;
- reassignment latency;
- robustness to node loss.

---

# 24. Module 20 — Telemetry, Health and Diagnostics

## Purpose

Make the system observable and diagnosable during simulation, bench testing and flight.

## Metrics

- sensor rates and drop counts;
- pipeline latency per stage;
- queue depth and age;
- GPU/CPU utilization;
- memory;
- temperature and power;
- model inference FPS;
- VIO quality;
- active tracks;
- static object throughput;
- uplink/downlink bandwidth;
- map version;
- correction quality;
- planner state.

## Requirements

- Prometheus-compatible metrics where possible;
- local ring-buffer logs;
- structured event logs;
- mission ID and node ID on every record;
- configurable redaction;
- time-aligned trace export;
- health state machine: `OK`, `DEGRADED`, `FAILED`, `RECOVERING`.

## Tests

- disk full;
- telemetry link loss;
- metric overload;
- subsystem failure;
- timestamp mismatch.

## Evaluation

- logging overhead;
- data completeness;
- fault-detection latency;
- ability to reconstruct mission timeline.

---

# 25. Module 21 — Simulation and Deterministic Replay

## Purpose

Allow development and regression testing without flight hardware.

## Initial simulation support

- dataset replay mode;
- recorded ROS bag replay;
- synthetic camera/IMU trajectory generator;
- network impairment emulator;
- multi-Wingman process orchestration;
- optional integration with AirSim, Isaac Sim or Gazebo after core replay works.

## Scenario format

A YAML scenario must define:

- node count;
- sensor sources;
- trajectories;
- calibration;
- network conditions;
- failures;
- model selection;
- expected outputs;
- random seed.

## Required scenarios

1. one Wingman, no Intelligence Node;
2. two Wingmen with perfect link;
3. two Wingmen with packet loss;
4. three Wingmen with overlapping observations;
5. Intelligence Node outage and recovery;
6. VIO drift corrected by global mapping;
7. repeated similar static objects;
8. dynamic objects crossing the scene.

## Tests

- exact deterministic replay from seed;
- process crash recovery;
- scenario validation;
- expected metric assertions;
- golden map comparison.

## Evaluation

- simulation reproducibility;
- real-time factor;
- resource use;
- regression sensitivity.

---

# 26. Module 22 — Dataset, Experiment and Model Registry

## Purpose

Make model and system evaluations reproducible.

## Registry contents

### Model entry

- model name;
- upstream source;
- license;
- version/commit;
- checksum;
- input/output contract;
- preprocessing;
- supported precision;
- supported hardware;
- calibration dataset;
- benchmark results.

### Dataset entry

- dataset name;
- source;
- license;
- version;
- local preparation script;
- split definitions;
- calibration assumptions;
- known limitations.

### Experiment entry

- Git commit;
- configuration snapshot;
- model checksums;
- dataset split;
- hardware;
- environment;
- metrics;
- artifacts;
- random seed.

## Initial tooling

- Weights & Biases integration optional but supported;
- local JSON/Parquet experiment manifest mandatory;
- model files stored outside Git;
- DVC optional for datasets and artifacts.

## Tests

- missing checksum;
- model mismatch;
- dataset version mismatch;
- reproducibility check;
- offline mode.

---

# 27. Module 23 — Benchmark and Model Evaluation Harness

## Purpose

Evaluate every candidate model and full pipeline using one consistent harness.

## Evaluation levels

### Level A — Model-only

- accuracy;
- latency;
- throughput;
- memory;
- power;
- precision mode.

### Level B — Module integration

- effect on VIO;
- effect on object association;
- static/dynamic performance;
- compression impact.

### Level C — System-level

- global pose accuracy;
- map quality;
- correction latency;
- bandwidth;
- mission completion.

## Hardware profiles

- x86 CUDA development workstation;
- Jetson Orin NX;
- Jetson AGX Orin;
- Hailo-8 attached to host;
- Hailo-10H attached to host;
- Hailo-15H vision processor profile;
- EdgeCortix SAKURA-II attached to host.

## Important reporting rule

Do not compare vendor peak TOPS as if they represent equivalent sustained application performance. Report measured end-to-end throughput, latency, memory, power and task accuracy under a defined workload.

## Output format

Each run must produce:

- machine-readable JSON;
- CSV summary;
- plots;
- system configuration;
- model version;
- raw timing samples;
- pass/fail against target profile.

## Tests

- warmup handling;
- outlier removal policy;
- repeated-run variance;
- thermal throttling detection;
- power-mode recording;
- missing hardware counters.

---

# 28. Module 24 — Security and Trust

## Purpose

Prevent unauthorized nodes, malicious messages and corrupted model/map updates.

## Requirements

- node identity and certificates;
- mutual authentication;
- encrypted transport;
- signed model and configuration manifests;
- message replay protection;
- authorization by node role;
- input size and rate limits;
- audit log;
- secure credential loading;
- no secrets in configs committed to Git.

## Threat cases

- spoofed Wingman;
- replayed correction;
- malformed embedding message;
- model-file substitution;
- denial of service;
- stale mission command;
- compromised node quarantine.

## Tests

- invalid certificate;
- expired certificate;
- replayed nonce;
- tampered message;
- oversized packet;
- rate limit;
- certificate rotation.

---

# 29. Module 25 — Deployment and Hardware Abstraction

## Purpose

Package ARIADNE for laboratory systems and drone hardware.

## Deployment profiles

### Wingman Jetson profile

- Jetson Orin NX baseline;
- optional Hailo accelerator for detector/embedding offload;
- camera and IMU adapters;
- NVENC/NVDEC usage where video recording or transmission is enabled.

### Intelligence Node profile

- Jetson AGX Orin or x86 GPU baseline;
- larger memory budget;
- Gaussian-splat backend;
- global database and planner.

### Alternate accelerator profiles

- Hailo-15H for integrated vision preprocessing and supported inference workloads;
- SAKURA-II for supported embedding or transformer inference;
- accelerator modules must not be treated as complete replacements for general CUDA rasterization unless a native backend is implemented and measured.

## Requirements

- Docker images per platform;
- pinned dependencies;
- hardware capability detection;
- model selection by capability;
- graceful fallback;
- service supervision;
- persistent configuration and logs;
- update and rollback procedure.

## Tests

- clean install;
- cold boot auto-start;
- service crash restart;
- model download failure;
- insufficient GPU memory;
- thermal throttling;
- disconnected sensors.

---

# 30. Module 26 — End-to-End Integration

## Purpose

Provide executable Wingman and Intelligence Node runtimes.

## Wingman runtime components

- sensor ingest;
- preprocessing;
- embeddings;
- VIO;
- saliency;
- clustering;
- tracking/static classification;
- local object state;
- uplink packaging;
- communications;
- correction application;
- telemetry.

## Intelligence runtime components

- communications;
- ingest registry;
- object association;
- FFGS adapter;
- global scene/map;
- pose optimization;
- correction generation;
- unified context;
- planning;
- telemetry.

## Orchestration requirements

- explicit lifecycle states;
- readiness checks;
- dependency startup order;
- clean shutdown;
- backpressure;
- fault isolation;
- degraded modes.

## Degraded modes

- Wingman without network: local VIO and local mission execution;
- Intelligence Node without splatting backend: association and pose-graph-only mode;
- embedding model unavailable: geometric/VIO-only fallback where possible;
- one camera failed: monocular fallback if configured;
- IMU failed: vision-only degraded mode if supported.

## End-to-end tests

- one recorded mission through full pipeline;
- two Wingmen with shared static objects;
- global correction reduces drift;
- packet-loss scenario;
- model backend failure;
- node restart;
- map snapshot and restore.

---

# 31. Initial Model Evaluation Matrix

| Function | Baseline | Alternative | Primary metrics |
|---|---|---|---|
| Semantic embedding | DINOv2-S/14 | SigLIP 2, DINOv2-B | Recall@k, latency, power |
| Local features | SuperPoint | ALIKED, learned DINO features | match recall, VIO impact |
| VIO | ORB-SLAM3 | VINS-Fusion, DPVO, OpenVINS | ATE, RPE, uptime |
| Saliency | DINO feature saliency | SAM2, spectral residual | region F1, stability |
| Tracking | ByteTrack | OC-SORT, mask tracking | IDF1, static F1 |
| Static classification | ego-motion residual rules | learned classifier | false static rate |
| Gaussian splatting | ReSplat | MVSplat, AnySplat | PSNR, SSIM, LPIPS, runtime |
| Association | FAISS + geometric gating | graph matching | false merge/split |
| Pose optimization | GTSAM | Ceres | ATE reduction, stability |
| Planning | frontier + auction | information gain | mission time, coverage |

---

# 32. Test Plan by Development Phase

## Phase 0 — Bootstrap

- package installs;
- CLI and configs validate;
- CPU smoke test;
- no regression to parent repository.

## Phase 1 — Single-Wingman replay

- camera/IMU replay;
- preprocessing;
- embedding;
- VIO;
- saliency and tracking;
- static object output.

**Exit:** deterministic static-object message stream from a recorded dataset.

## Phase 2 — Intelligence Node ingest and association

- receive observations;
- deduplicate;
- associate cross-view objects;
- maintain global object IDs.

**Exit:** two replayed Wingmen produce a shared global object registry.

## Phase 3 — Gaussian reconstruction integration

- ReSplat adapter;
- keyframe selection;
- standardized Gaussian scene output;
- map versioning.

**Exit:** shared observations produce a reproducible global reconstruction.

## Phase 4 — Global pose correction

- factor graph;
- correction deltas;
- Wingman application;
- drift reduction report.

**Exit:** global corrections measurably reduce trajectory error without local discontinuity.

## Phase 5 — Network degradation

- packet loss;
- latency;
- partition;
- reconnect;
- stale-message handling.

**Exit:** no unbounded queues or system deadlock; corrections resume after reconnect.

## Phase 6 — Hardware-in-the-loop

- Jetson Wingman;
- camera and IMU;
- Intelligence Node GPU;
- measured power, latency and temperature.

**Exit:** sustained run for at least 30 minutes with bounded memory and no thermal failure.

## Phase 7 — Flight test

- single drone baseline;
- two-drone overlap mission;
- GPS-assisted truth initially;
- GPS-denied processing evaluated offline first;
- progressive onboard autonomy.

**Exit:** repeatable mission showing unified map and corrected global poses.

---

# 33. System-Level Acceptance Metrics

These are initial targets and must be refined after baseline measurements.

| Category | Initial target |
|---|---:|
| Wingman perception rate | 10–20 Hz minimum |
| VIO update rate | 20 Hz or higher |
| Static-object uplink rate | 1–10 Hz, adaptive |
| Correction update rate | 0.5–5 Hz |
| Median correction RTT | under 500 ms on healthy link |
| Wingman memory | bounded, profile-specific |
| Network partition tolerance | at least 60 seconds |
| Static false-insertion rate | under 5% baseline target |
| Global drift improvement | at least 30% over local VIO baseline |
| Long-run stability | 30-minute HIL without leak or deadlock |
| Reproducibility | same seed/config within defined tolerance |

---

# 34. Codex Work Order

Codex should implement in this order:

1. Module 00 — Project Bootstrap.
2. Module 01 — Common Types and Frames.
3. Module 02 — Sensor Ingestion.
4. Module 03 — Preprocessing.
5. Module 04 — Embedding Generation.
6. Module 05 — VIO Adapter.
7. Module 06 — Saliency Detection.
8. Module 07 — Saliency Clustering.
9. Module 08 — Tracking and Static/Dynamic Classification.
10. Module 09 — Local Object Store.
11. Module 10 — Uplink Packaging.
12. Module 11 — Communications.
13. Module 12 — Intelligence Ingest.
14. Module 13 — Object Association.
15. Module 14 — Gaussian Splat Adapter.
16. Module 15 — Global Map.
17. Module 16 — Global Pose Optimization.
18. Module 17 — Correction Delta Application.
19. Module 18 — Unified Context.
20. Module 20 — Telemetry.
21. Module 21 — Simulation and Replay.
22. Module 23 — Benchmark Harness.
23. Module 19 — Planning.
24. Module 24 — Security.
25. Module 25 — Deployment.
26. Module 26 — End-to-End Integration.

For each work item, Codex should first generate:

- implementation plan;
- files to create or modify;
- interfaces and schemas;
- tests;
- benchmark plan;
- risks and assumptions.

Only then should it write code.

---

# 35. First Codex Prompt

```text
You are working in the repository lauretta-io/feed_forward_Gaussian_Splat.

Read documentation/ARIADNE_CODEX_BUILD_SPEC.md and inspect the existing repository before making changes. ARIADNE must be implemented under applications/ariadne and must not break the existing ReSplat, MVSplat, AnySplat or OpenSplat paths.

Start with Module 00 and Module 01 only.

Before writing code:
1. Report the existing repository structure relevant to the work.
2. Propose the exact files to create or modify.
3. Identify dependency and compatibility risks.
4. Define the public interfaces and test cases.

Then implement:
- the ARIADNE Python package skeleton;
- pyproject and CLI;
- typed configuration loading and validation;
- structured logging;
- common timestamp, frame, transform, pose and covariance types;
- transform utilities;
- unit tests;
- CPU-only smoke test;
- README instructions.

Constraints:
- Python 3.12.
- No model downloads during import or tests.
- No hardcoded paths.
- No changes to parent model behavior.
- All tests must run without a GPU.
- Use precise source/destination frame names for every transform.
- Add comments only where they clarify non-obvious behavior.

At completion, provide:
- summary of changes;
- files changed;
- test commands and results;
- unresolved risks;
- recommended next module.
```
