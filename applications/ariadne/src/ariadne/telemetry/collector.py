"""Low-cardinality counters, gauges, distributions, and component health."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np

from ariadne.common import Timestamp

LOGGER = logging.getLogger(__name__)
_METRIC_NAME = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


class ComponentHealth(StrEnum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    RECOVERING = "recovering"
    FAILED = "failed"


@dataclass(frozen=True)
class TelemetryEvent:
    timestamp: Timestamp
    mission_id: str
    node_id: str
    component: str
    event: str
    severity: str
    fields: dict[str, str | int | float | bool]


@dataclass(frozen=True)
class TelemetrySnapshot:
    timestamp: Timestamp
    mission_id: str
    node_id: str
    counters: dict[str, int]
    gauges: dict[str, float]
    distributions: dict[str, dict[str, float | int]]
    health: dict[str, ComponentHealth]
    events: tuple[TelemetryEvent, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "timestamp": self.timestamp.to_dict(),
            "mission_id": self.mission_id,
            "node_id": self.node_id,
            "counters": self.counters,
            "gauges": self.gauges,
            "distributions": self.distributions,
            "health": {key: value.value for key, value in self.health.items()},
            "events": [
                {
                    "timestamp": event.timestamp.to_dict(),
                    "mission_id": event.mission_id,
                    "node_id": event.node_id,
                    "component": event.component,
                    "event": event.event,
                    "severity": event.severity,
                    "fields": event.fields,
                }
                for event in self.events
            ],
        }

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")


class TelemetryCollector:
    def __init__(
        self,
        *,
        max_distribution_samples: int = 1024,
        max_metrics: int = 256,
        max_events: int = 1024,
        mission_id: str = "reference",
        node_id: str = "reference",
        redacted_keys: tuple[str, ...] = ("token", "secret", "password", "key"),
    ) -> None:
        if max_distribution_samples <= 0 or max_metrics <= 0 or max_events <= 0:
            raise ValueError("telemetry bounds must be positive")
        if not mission_id or not node_id:
            raise ValueError("telemetry mission and node identifiers are required")
        self.max_distribution_samples = max_distribution_samples
        self.max_metrics = max_metrics
        self.max_events = max_events
        self.mission_id = mission_id
        self.node_id = node_id
        self.redacted_keys = {key.lower() for key in redacted_keys}
        self._counters: dict[str, int] = {}
        self._gauges: dict[str, float] = {}
        self._samples: dict[str, list[float]] = {}
        self._health: dict[str, ComponentHealth] = {}
        self._events: list[TelemetryEvent] = []

    def _validate_name(self, name: str) -> None:
        if not _METRIC_NAME.fullmatch(name):
            raise ValueError(f"invalid telemetry name: {name!r}")
        names = set(self._counters) | set(self._gauges) | set(self._samples)
        if name not in names and len(names) >= self.max_metrics:
            raise ValueError("telemetry metric cardinality limit exceeded")

    def increment(self, name: str, value: int = 1) -> None:
        self._validate_name(name)
        if value < 0:
            raise ValueError("counter increments must be non-negative")
        self._counters[name] = self._counters.get(name, 0) + value

    def gauge(self, name: str, value: float) -> None:
        self._validate_name(name)
        if not np.isfinite(value):
            raise ValueError("gauge values must be finite")
        self._gauges[name] = float(value)

    def observe(self, name: str, value: float) -> None:
        self._validate_name(name)
        if not np.isfinite(value):
            raise ValueError("distribution values must be finite")
        samples = self._samples.setdefault(name, [])
        samples.append(float(value))
        del samples[: max(0, len(samples) - self.max_distribution_samples)]

    def set_health(self, component: str, health: ComponentHealth) -> None:
        if not component:
            raise ValueError("component name must not be empty")
        self._health[component] = health
        if health is not ComponentHealth.HEALTHY:
            LOGGER.warning("component_health component=%s state=%s", component, health.value)

    def record_event(
        self,
        timestamp: Timestamp,
        component: str,
        event: str,
        *,
        severity: str = "INFO",
        fields: dict[str, Any] | None = None,
    ) -> None:
        if not component or not event or severity not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
            raise ValueError("telemetry event fields are invalid")
        sanitized: dict[str, str | int | float | bool] = {}
        for key, value in (fields or {}).items():
            if not key:
                raise ValueError("telemetry event field names must not be empty")
            if key.lower() in self.redacted_keys:
                sanitized[key] = "[REDACTED]"
            elif isinstance(value, bool | int | str):
                sanitized[key] = value
            elif isinstance(value, float) and np.isfinite(value):
                sanitized[key] = value
            else:
                raise ValueError("telemetry event values must be finite scalar values")
        self._events.append(
            TelemetryEvent(
                timestamp,
                self.mission_id,
                self.node_id,
                component,
                event,
                severity,
                sanitized,
            )
        )
        del self._events[: max(0, len(self._events) - self.max_events)]

    def snapshot(self, timestamp: Timestamp) -> TelemetrySnapshot:
        distributions = {
            name: {
                "count": len(samples),
                "p50": float(np.percentile(samples, 50)),
                "p95": float(np.percentile(samples, 95)),
                "max": max(samples),
            }
            for name, samples in sorted(self._samples.items())
            if samples
        }
        return TelemetrySnapshot(
            timestamp,
            self.mission_id,
            self.node_id,
            dict(sorted(self._counters.items())),
            dict(sorted(self._gauges.items())),
            distributions,
            dict(sorted(self._health.items())),
            tuple(self._events),
        )

    def prometheus_text(self, timestamp: Timestamp) -> str:
        snapshot = self.snapshot(timestamp)
        labels = (
            f'mission_id="{self._escape_label(self.mission_id)}",'
            f'node_id="{self._escape_label(self.node_id)}"'
        )
        lines: list[str] = []
        for name, counter_value in snapshot.counters.items():
            lines.append(f"ariadne_{name}_total{{{labels}}} {counter_value}")
        for name, gauge_value in snapshot.gauges.items():
            lines.append(f"ariadne_{name}{{{labels}}} {gauge_value}")
        for name, summary in snapshot.distributions.items():
            for statistic in ("count", "p50", "p95", "max"):
                lines.append(
                    f"ariadne_{name}_{statistic}{{{labels}}} {summary[statistic]}"
                )
        health_values = {
            ComponentHealth.HEALTHY: 0,
            ComponentHealth.DEGRADED: 1,
            ComponentHealth.RECOVERING: 2,
            ComponentHealth.FAILED: 3,
        }
        for component, health in snapshot.health.items():
            component_labels = (
                f'{labels},component="{self._escape_label(component)}"'
            )
            lines.append(
                f"ariadne_component_health{{{component_labels}}} {health_values[health]}"
            )
        return "\n".join(lines) + ("\n" if lines else "")

    @staticmethod
    def _escape_label(value: str) -> str:
        return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')
