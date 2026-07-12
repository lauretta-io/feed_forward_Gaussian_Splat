# Bootstrap Architecture

The core package is independent of ROS, CUDA, model weights, and the parent model runtimes.
Configuration and common geometry are safe to import in CPU-only processes. Runtime commands
currently provide deterministic bootstrap probes; later modules replace the probes behind the
same command boundary.

Configuration flows from YAML through Pydantic validation before logging or runtime setup. The
common package owns frame and transform conventions so downstream sensor, mapping, and correction
modules cannot silently reinterpret matrices.
