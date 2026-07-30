# Bootstrap Architecture

The core package is independent of ROS, CUDA, model weights, and the parent model runtimes.
Configuration and common geometry are safe to import in CPU-only processes. Runtime commands
provide deterministic component, exchange, global-scene, operations, and end-to-end gates behind
one command boundary.

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
