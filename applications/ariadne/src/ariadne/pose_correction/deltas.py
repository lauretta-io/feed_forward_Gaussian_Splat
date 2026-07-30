"""Versioned correction deltas that preserve local VIO continuity."""

from __future__ import annotations

import json
import logging
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, cast

import numpy as np

from ariadne.common import Timestamp, TransformSE3

LOGGER = logging.getLogger(__name__)
GENERATOR_SCHEMA = "ariadne.correction-generator.v1"
APPLIER_SCHEMA = "ariadne.correction-applier.v1"


class CorrectionResetRequiredError(RuntimeError):
    """The global correction is too large for continuity-preserving application."""


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


@dataclass(frozen=True)
class CorrectionDelta:
    correction_id: str
    agent_id: str
    issued_at: Timestamp
    expires_at_ns: int
    local_to_global: TransformSE3
    confidence: float

    def __post_init__(self) -> None:
        if not self.correction_id or not self.agent_id:
            raise ValueError("correction identifiers must not be empty")
        if self.expires_at_ns <= self.issued_at.monotonic_ns:
            raise ValueError("correction expiry must follow issue time")
        if not 0 <= self.confidence <= 1:
            raise ValueError("correction confidence must be between zero and one")

    def to_dict(self) -> dict[str, object]:
        return {
            "correction_id": self.correction_id,
            "agent_id": self.agent_id,
            "issued_at": self.issued_at.to_dict(),
            "expires_at_ns": self.expires_at_ns,
            "local_to_global": self.local_to_global.to_dict(),
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CorrectionDelta:
        return cls(
            str(payload["correction_id"]),
            str(payload["agent_id"]),
            Timestamp.from_dict(cast(dict[str, Any], payload["issued_at"])),
            int(payload["expires_at_ns"]),
            TransformSE3.from_dict(cast(dict[str, Any], payload["local_to_global"])),
            float(payload["confidence"]),
        )


@dataclass(frozen=True)
class AppliedCorrection:
    correction_id: str
    corrected_pose: TransformSE3
    applied_fraction: float


class CorrectionDeltaGenerator:
    def __init__(self, *, max_history: int = 1024) -> None:
        if max_history <= 0:
            raise ValueError("max_history must be positive")
        self.max_history = max_history
        self._sequence_by_agent: dict[str, int] = {}
        self._last_issued_by_agent: dict[str, int] = {}
        self._history: deque[CorrectionDelta] = deque()
        self.metrics = {"generated": 0, "rejected": 0, "restores": 0}

    @property
    def history(self) -> tuple[CorrectionDelta, ...]:
        return tuple(self._history)

    def generate(
        self,
        agent_id: str,
        local_pose: TransformSE3,
        optimized_global_pose: TransformSE3,
        *,
        issued_at: Timestamp,
        ttl_ns: int = 60_000_000_000,
        confidence: float = 1.0,
    ) -> CorrectionDelta:
        if local_pose.source != optimized_global_pose.source:
            raise ValueError("local and optimized poses must share a source frame")
        if not local_pose.destination.value.startswith("local_"):
            raise ValueError("local pose must terminate in a local_<agent> frame")
        if optimized_global_pose.destination.value != "global":
            raise ValueError("optimized pose must terminate in the global frame")
        if ttl_ns <= 0:
            raise ValueError("correction ttl_ns must be positive")
        previous_issue = self._last_issued_by_agent.get(agent_id)
        if previous_issue is not None and issued_at.monotonic_ns < previous_issue:
            self.metrics["rejected"] += 1
            raise ValueError("correction issue timestamps must be monotonic per agent")
        local_to_global = local_pose.inverse().then(optimized_global_pose)
        sequence = self._sequence_by_agent.get(agent_id, 0)
        self._sequence_by_agent[agent_id] = sequence + 1
        correction = CorrectionDelta(
            f"correction_{agent_id}_{sequence:06d}",
            agent_id,
            issued_at,
            issued_at.monotonic_ns + ttl_ns,
            local_to_global,
            confidence,
        )
        self._last_issued_by_agent[agent_id] = issued_at.monotonic_ns
        self._history.append(correction)
        while len(self._history) > self.max_history:
            self._history.popleft()
        self.metrics["generated"] += 1
        return correction

    def write_json(self, path: str | Path) -> Path:
        return _write_json(
            Path(path),
            {
                "schema": GENERATOR_SCHEMA,
                "max_history": self.max_history,
                "sequence_by_agent": self._sequence_by_agent,
                "last_issued_by_agent": self._last_issued_by_agent,
                "history": [item.to_dict() for item in self.history],
            },
        )

    @classmethod
    def read_json(cls, path: str | Path) -> CorrectionDeltaGenerator:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("schema") != GENERATOR_SCHEMA:
            raise ValueError("unsupported correction generator snapshot schema")
        generator = cls(max_history=int(payload["max_history"]))
        generator._sequence_by_agent = {
            str(key): int(value)
            for key, value in cast(dict[str, Any], payload["sequence_by_agent"]).items()
        }
        generator._last_issued_by_agent = {
            str(key): int(value)
            for key, value in cast(dict[str, Any], payload["last_issued_by_agent"]).items()
        }
        history = [
            CorrectionDelta.from_dict(item)
            for item in cast(list[dict[str, Any]], payload["history"])
        ]
        if len(history) > generator.max_history:
            raise ValueError("correction generator history exceeds configured bounds")
        generator._history.extend(history)
        generator.metrics["restores"] = 1
        return generator


class CorrectionApplier:
    def __init__(
        self,
        *,
        max_translation_step_m: float = 0.5,
        max_total_translation_m: float = 10.0,
        max_applied_history: int = 4096,
    ) -> None:
        if (
            max_translation_step_m <= 0
            or max_total_translation_m < max_translation_step_m
            or max_applied_history <= 0
        ):
            raise ValueError("correction application bounds are invalid")
        self.max_translation_step_m = max_translation_step_m
        self.max_total_translation_m = max_total_translation_m
        self.max_applied_history = max_applied_history
        self._applied: set[str] = set()
        self._applied_order: deque[str] = deque()
        self.metrics = {
            "applied": 0,
            "duplicates": 0,
            "expired": 0,
            "smoothed": 0,
            "reset_required": 0,
            "restores": 0,
        }

    def apply(
        self,
        local_pose: TransformSE3,
        correction: CorrectionDelta,
        *,
        now: Timestamp,
    ) -> AppliedCorrection:
        if correction.correction_id in self._applied:
            self.metrics["duplicates"] += 1
            raise ValueError("correction has already been applied")
        if now.monotonic_ns >= correction.expires_at_ns:
            self.metrics["expired"] += 1
            raise ValueError("correction has expired")
        if local_pose.destination != correction.local_to_global.source:
            raise ValueError("local pose and correction frames do not compose")
        translation = correction.local_to_global.translation_m
        distance = float(np.linalg.norm(translation))
        if distance > self.max_total_translation_m:
            self.metrics["reset_required"] += 1
            raise CorrectionResetRequiredError(
                "correction exceeds the continuity-preserving translation bound"
            )
        epsilon = float(np.finfo(np.float64).eps)
        fraction = min(1.0, self.max_translation_step_m / max(distance, epsilon))
        quaternion = correction.local_to_global.quaternion_xyzw()
        partial_quaternion = (
            np.array([0.0, 0.0, 0.0, 1.0]) * (1.0 - fraction) + quaternion * fraction
        )
        partial_quaternion /= np.linalg.norm(partial_quaternion)
        partial = TransformSE3.from_translation_quaternion(
            correction.local_to_global.source,
            correction.local_to_global.destination,
            translation * fraction,
            partial_quaternion,
        )
        corrected = local_pose.then(partial)
        self._applied.add(correction.correction_id)
        self._applied_order.append(correction.correction_id)
        while len(self._applied_order) > self.max_applied_history:
            self._applied.remove(self._applied_order.popleft())
        self.metrics["applied"] += 1
        self.metrics["smoothed"] += int(fraction < 1.0)
        LOGGER.info("correction_applied id=%s fraction=%.3f", correction.correction_id, fraction)
        return AppliedCorrection(correction.correction_id, corrected, fraction)

    def write_json(self, path: str | Path) -> Path:
        return _write_json(
            Path(path),
            {
                "schema": APPLIER_SCHEMA,
                "max_translation_step_m": self.max_translation_step_m,
                "max_total_translation_m": self.max_total_translation_m,
                "max_applied_history": self.max_applied_history,
                "applied_ids": list(self._applied_order),
            },
        )

    @classmethod
    def read_json(cls, path: str | Path) -> CorrectionApplier:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("schema") != APPLIER_SCHEMA:
            raise ValueError("unsupported correction applier snapshot schema")
        applier = cls(
            max_translation_step_m=float(payload["max_translation_step_m"]),
            max_total_translation_m=float(payload["max_total_translation_m"]),
            max_applied_history=int(payload["max_applied_history"]),
        )
        applied_ids = [str(value) for value in cast(list[object], payload["applied_ids"])]
        if len(applied_ids) > applier.max_applied_history or len(applied_ids) != len(
            set(applied_ids)
        ):
            raise ValueError("correction applier history is invalid")
        applier._applied_order.extend(applied_ids)
        applier._applied.update(applied_ids)
        applier.metrics["restores"] = 1
        return applier
