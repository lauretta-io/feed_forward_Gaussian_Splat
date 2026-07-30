# ARIADNE implementation plans

This is the execution ledger for Modules 00-26 in the build specification. “Reference” means an
executable CPU implementation that validates the contract; it does not claim production model or
flight readiness. Every module retains the same evidence shape: typed public API, strict config,
failure tests, structured metrics/logs, a CLI or integrated benchmark path, and artifacts outside
package source.

| Module | State | Implementation and files | Interfaces / schema | Tests and benchmark | Primary risk / assumption |
|---|---|---|---|---|---|
| 00 Bootstrap | Validated | `pyproject.toml`, `config.py`, `cli.py`, Docker and logging | `AriadneConfig`, CLI contract | config/CLI/import smoke, transform microbenchmark | CPU import must remain free of CUDA/model side effects |
| 01 Common types | Validated | `common/types.py`, `common/conventions.py` | time, frames, calibration, health, versioned SE(3) types | serialization, frame mismatch, covariance and round-trip tests | all adapters must convert into the canonical frame convention |
| 02 Sensor ingestion | Validated | `replay/sources.py`, `replay/synchronizer.py`, dataset adapters | `ImageFrame`, `ImuSample`, `ReplayBatch`, `SynchronizedPacket` | replay/timing tests plus real corpus evaluation | live camera and hardware adapters remain environment-specific |
| 03 Preprocessing | Reference | `perception/preprocessing.py`; config under `wingman.preprocessing` | `FrameQuality`, `PreprocessedFrame`, `ImagePreprocessor` | rejection/determinism tests and exchange latency/quality metrics | CPU nearest-neighbor resize is the parity baseline, not the production fast path |
| 04 Features / embeddings | Reference | `models/features.py` | geometric `FeatureSet` and semantic `SemanticEmbedding` remain separate | feature recall, cosine separation and latency in Phase 1 | production model weights and accelerator parity require external artifacts |
| 05 VIO | Validated | `models/vio.py`, `backends/external_vio.py`, backend scripts | swappable VIO backend and versioned pose stream | synthetic regression plus OpenVINS/ORB-SLAM3 D2SLAM reports | current real runs use different replay windows and are not rankable |
| 06 Saliency | Reference | `perception/saliency.py`; config under `wingman.saliency` | `SaliencyMap`, `SaliencyDetector` | rejected-frame and deterministic heatmap tests; exchange latency | gradient/contrast baseline must later be compared with production saliency models |
| 07 Region formation | Reference | `perception/saliency.py` | `SaliencyRegion`, `SaliencyClusterer` | connected-region size/ranking tests; region-count benchmark | 4-connectivity can split diagonally connected evidence |
| 08 Tracking/classification | Reference | `tracking/static_filter.py` | `TrackObservation`, `TrackState`, four-state hysteresis | transition, reset, dynamic-rejection and Phase 1 acceptance tests | thresholds require calibration on labeled flight data |
| 09 Local object store | Reference | `object_state/store.py`; bounded config and atomic JSON snapshots | `KeyframeRecord`, `LocalObjectRecord`, `LocalObjectStore`; `ariadne.local-object-store.v1` | confirmed-only admission, stale-update rejection, priority/eviction, snapshot recovery and exchange tests | JSON provides restart-safe reference persistence; flight profiles still need a transactional storage backend |
| 10 Uplink packaging | Reference | `communications/uplink.py` | versioned `UplinkPacket` JSON payload with SHA-256 and zlib | size bound, mismatch and round-trip tests; bytes/object metric | JSON is inspectable but Protobuf may be smaller in production |
| 11 Mesh transport | Reference | `communications/transport.py`; bounded queue, reliable outbox and dedupe window | message priority, TTL, authenticated-direction receipt, finite retry budget and packet envelope | drop, duplicate, expiry, queue/outbox saturation, spoofed acknowledgement, retry exhaustion and delivery tests | in-memory transport models reliability policy, not QUIC/radio behavior or reconnect persistence |
| 12 Intelligence ingest | Reference | `intelligence/registry.py` plus `intelligence/journal.py`; bounded hot state, clock-skew policy, atomic snapshots and segmented raw-envelope journal | `RegisteredObservation`, `ObservationRegistry`, `ObservationJournal`; versioned registry and journal schemas | validation, journal-before-mutation, checksum/chain corruption, torn-tail recovery, dedupe, retention, replay fidelity and restart tests | the JSONL reference journal is single-writer; production still needs transactional database/Parquet adapters and interruption testing on target storage |
| 13 Object association | Reference | `tracking/association.py`; bounded objects/evidence plus atomic state recovery | geometry/cosine gates, persistent local-to-global mappings, stable IDs and scored `AssociationEvidence` | merge/non-merge, capacity, bounded evidence, restart stability and Phase 1 global-ID gates | pairwise greedy assignment still needs Hungarian/min-cost batch assignment and multi-hypothesis correction for dense ambiguous scenes |
| 14 Gaussian splat adapter | Reference | `splatting/adapter.py` with protocol-checked backend registry, bounded executor and CPU object-Gaussian backend | standardized primitive/result/diagnostics, explicit observation/object/memory budgets and normalized backend/resource/OOM failures | fusion, registry, request-bound, invalid-output and simulated OOM tests plus executor-backed global-scene metrics | ReSplat/MVSplat/AnySplat still need concrete camera-batch adapters and measured accelerator memory rather than the reference working-set estimate |
| 15 Global scene map | Reference | `splatting/scene_map.py` with atomic bounded updates, crash-safe snapshot replacement, on-disk retention, recovery fallback and rollback | Gaussian/object state, provenance, immutable snapshots and `ariadne.global-scene-snapshot.v1` | insert/merge/time-order/bound/conflict/rollback/interrupted-write/restart tests plus persistence benchmark | the JSON reference store still needs replacement with a transactional backend for concurrent production writers |
| 16 Global pose optimization | Reference | `optimization/se3_pose_graph.py`; bounded deduplicated incremental constraints, bounded result history and atomic restart state | revisioned full SE(3) constraints, covariance propagation, disconnected components, rejected loops and rerunnable recovery | rotation/covariance/disconnection/outlier/capacity/dedupe/restart tests plus restored-revision benchmark | maximum-information forest validates state and robustness contracts but production still needs nonlinear convergence diagnostics and delayed-factor relinearization |
| 17 Correction deltas | Reference | `pose_correction/deltas.py`; persistent bounded generator/application histories and reset-required safety gate | stable per-agent sequences, monotonic issue time, expiry, confidence, bounded smoothing and restart-safe replay suppression | idempotence across restart, expiry, frame mismatch, time travel, history bounds, unsafe jump and sequence recovery tests | target flight integration must define the relocalization procedure used after `CorrectionResetRequiredError` |
| 18 Unified context | Reference | `context/scene_graph.py` plus executable `skyla/handoff.py` boundary | typed nodes/edges, mission goals, constraints, vehicle state, freshness and versioned global-frame envelope | graph update/query, serialization, unknown-frontier, stale-data and degraded-context tests | planners must never infer unavailable certainty |
| 19 Planning | Reference | `planning/frontier.py` allocator plus `skyla/planner.py` mission-to-route reference | battery/link/health eligibility, no-fly filtering, idempotent route requests and local safety gate | unique assignment, low-battery exclusion, blocked route, expiry, replay, stale revision and node-loss replan tests | route requests remain advisory until a flight controller validates them locally |
| 20 Telemetry | Reference | `telemetry/collector.py` bounded health aggregation, redacted event ring, JSON trace and Prometheus text export | mission/node-scoped counters, gauges, distributions, four-state health and events | cardinality/sample/event bounds, redaction, recovering/failure health, persistence, Prometheus and p50/p95 tests | power metrics depend on hardware-specific readers |
| 21 Simulation/replay | Reference | deterministic multi-Wingman network, clock, drift and dynamics evaluator | seed, packet loss, partition, recovery and invariant report | replay-hash regression and 60-second partition recovery | synthetic success is not field evidence |
| 22 Dataset/model registry | Reference | typed `registry/catalog.py` plus dataset/model YAML catalogs | licenses, provenance, backends, versions and experiment records | schema/duplicate/write/read tests and operations inventory | upstream licenses and artifacts can change independently |
| 23 Benchmark harness | Reference | Phase 1, exchange, global-scene, operations and W&B evaluation | `DatasetEvaluation` JSON plus optional W&B artifact | deterministic suite tests and CLI integration | hardware comparisons require fixed replay windows and profiles |
| 24 Security/trust | Reference | `security/envelope.py` with injected key map and replay window | signed identity, nonce, timestamp, destination and HMAC | tamper, replay, expiry, unknown-node and destination tests | key custody is deployment-specific and secrets stay clone-local |
| 25 Deployment/HAL | Reference | Docker profiles plus `deployment/capabilities.py` and profile YAML | compute/camera/accelerator capabilities and explicit requirements | compatible and missing-capability tests plus operations gate | accelerator toolchains are not available on every CI runner |
| 26 End-to-end integration | Reference | `runtime/reference.py` composes four executable gates | lifecycle result, stage evidence and degraded-mode outcomes | full suite, 60-second partition recovery and low-battery planning exclusion | CPU composition is not a hardware-in-loop or flight claim |

## Current executable gates

```bash
ariadne benchmark --suite phase1 --output outputs/ariadne/phase1/benchmark.json
ariadne benchmark --suite exchange --output outputs/ariadne/exchange/benchmark.json
ariadne benchmark --suite global-scene --output outputs/ariadne/global-scene/benchmark.json
ariadne benchmark --suite operations --output outputs/ariadne/operations/benchmark.json
ariadne benchmark --suite end-to-end --output outputs/ariadne/end-to-end/benchmark.json
pytest
```

The status website consumes all benchmark artifacts plus a contiguous 20-frame real replay evidence
segment. `benchmarks/video_evidence.py` selects the segment at production ORB-SLAM3 initialization,
runs the actual reference perception implementations per frame, and attaches the nearest production
pose within 60 ms. Camera-facing modules show that slowed 20-frame playback; transport, global-state,
planning, trust, deployment, and operations modules show their deterministic benchmark outputs so a
video is never used as decorative evidence. A planned module is promoted to Reference or Validated
only after its implementation, focused failure tests, integrated benchmark evidence, and an output
appropriate to the component are present.
