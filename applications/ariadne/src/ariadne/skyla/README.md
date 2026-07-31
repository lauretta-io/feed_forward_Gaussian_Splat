# ARIADNE-to-SKYLA reference boundary

`SkylaHandoff` packages the current global context revision, mission goals, vehicle state,
frontiers, and no-fly constraints into the deterministic `ariadne.skyla.handoff.v1` JSON contract.
`SkylaMissionPlanner` consumes that contract and emits versioned, expiring, idempotent
`RouteRequest` values.

The reference planner fails closed on degraded context by default, rejects expired or stale
revisions, filters routes that intersect spherical no-fly zones, and excludes unavailable,
failed, low-battery, or low-link Wingmen. Route requests are advisory and always require local
collision and flight-safety validation.

Run the integrated reference and focused tests from `applications/ariadne`:

```bash
PYTHONPATH=src ../../.venv/bin/python -m ariadne benchmark \
  --suite operations \
  --output outputs/ariadne/operations/benchmark.json
PYTHONPATH=src ../../.venv/bin/python -m pytest -q tests/unit/test_skyla.py
```
