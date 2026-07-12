# Bootstrap Architecture

The core package is independent of ROS, CUDA, model weights, and the parent model runtimes.
Configuration and common geometry are safe to import in CPU-only processes. Runtime commands
currently provide deterministic bootstrap probes; later modules replace the probes behind the
same command boundary.

Configuration flows from YAML through Pydantic validation before logging or runtime setup. The
common package owns frame and transform conventions so downstream sensor, mapping, and correction
modules cannot silently reinterpret matrices.

The Phase 1 reference pipeline adds explicit boundaries beneath that bootstrap layer:

1. `ariadne.replay` aligns camera frames and per-agent IMU windows and drops frames outside the
   configured synchronization tolerance.
2. `ariadne.models` defines swappable VIO, geometric-feature, and semantic-embedding contracts.
3. `ariadne.tracking` applies temporal hysteresis before static observations can enter the global
   registry, then associates confirmed observations using geometry and embeddings.
4. `ariadne.optimization` provides a robust incremental translation-graph reference backend.
5. `ariadne.benchmarks.phase1` executes the chain and emits one versioned JSON/W&B report.

The NumPy model and optimizer implementations are deterministic references, not substitutes for
the production runtimes. External adapters must preserve these boundaries and report results using
the same benchmark metric names.
