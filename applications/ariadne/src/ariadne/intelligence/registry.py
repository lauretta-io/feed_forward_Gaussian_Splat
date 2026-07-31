"""Validated, ordered, deduplicated Intelligence-node observation registry."""

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
import numpy.typing as npt

from ariadne.common import ModelVersion, Timestamp
from ariadne.communications import TransportMessage, UplinkPackager
from ariadne.intelligence.journal import ObservationJournal

LOGGER = logging.getLogger(__name__)
SNAPSHOT_SCHEMA = "ariadne.observation-registry.v1"


@dataclass(frozen=True)
class RegisteredObservation:
    observation_id: str
    agent_id: str
    local_id: str
    timestamp: Timestamp
    position_m: npt.NDArray[np.float64]
    embedding: npt.NDArray[np.float64]
    covariance_diagonal: npt.NDArray[np.float64]
    confidence: float
    model_version: ModelVersion

    def __post_init__(self) -> None:
        if not self.observation_id or not self.agent_id or not self.local_id:
            raise ValueError("registered observation identifiers must not be empty")
        if not 0 <= self.confidence <= 1:
            raise ValueError("registered observation confidence must be between zero and one")
        position = np.asarray(self.position_m, dtype=np.float64)
        embedding = np.asarray(self.embedding, dtype=np.float64)
        covariance = np.asarray(self.covariance_diagonal, dtype=np.float64)
        if (
            position.shape != (3,)
            or covariance.shape != (3,)
            or embedding.ndim != 1
            or not embedding.size
        ):
            raise ValueError("registered observation vector shapes are invalid")
        if not all(np.all(np.isfinite(item)) for item in (position, embedding, covariance)):
            raise ValueError("registered observation vectors must be finite")
        if np.any(covariance < 0):
            raise ValueError("registered observation covariance must be non-negative")
        for name, item in (
            ("position_m", position),
            ("embedding", embedding),
            ("covariance_diagonal", covariance),
        ):
            copy = item.copy()
            copy.setflags(write=False)
            object.__setattr__(self, name, copy)


class ObservationRegistry:
    def __init__(
        self,
        *,
        retention_ns: int = 60_000_000_000,
        max_observations: int = 8192,
        max_sequence_history: int = 4096,
        max_future_skew_ns: int = 1_000_000_000,
        journal: ObservationJournal | None = None,
    ) -> None:
        if (
            retention_ns <= 0
            or max_observations <= 0
            or max_sequence_history <= 0
            or max_future_skew_ns < 0
        ):
            raise ValueError("registry bounds must be valid")
        self.retention_ns = retention_ns
        self.max_observations = max_observations
        self.max_sequence_history = max_sequence_history
        self.max_future_skew_ns = max_future_skew_ns
        self.journal = journal
        self._observations: dict[str, RegisteredObservation] = {}
        self._sequences: set[tuple[str, int]] = set()
        self._sequence_order: deque[tuple[str, int]] = deque()
        self.metrics = {
            "packets": 0,
            "observations": 0,
            "duplicates": 0,
            "expired": 0,
            "evicted": 0,
            "future": 0,
            "snapshots": 0,
            "restores": 0,
            "journaled": 0,
            "journal_replays": 0,
        }

    @property
    def observations(self) -> tuple[RegisteredObservation, ...]:
        return tuple(
            sorted(
                self._observations.values(),
                key=lambda item: (item.timestamp.monotonic_ns, item.observation_id),
            )
        )

    def ingest(
        self,
        message: TransportMessage,
        now: Timestamp,
        *,
        persist_raw: bool = True,
    ) -> tuple[RegisteredObservation, ...]:
        packet = message.packet
        key = (packet.agent_id, packet.sequence)
        if key in self._sequences:
            self.metrics["duplicates"] += 1
            return ()
        if packet.timestamp.monotonic_ns + self.retention_ns < now.monotonic_ns:
            self.metrics["expired"] += 1
            return ()
        if packet.timestamp.monotonic_ns > now.monotonic_ns + self.max_future_skew_ns:
            self.metrics["future"] += 1
            return ()
        body = UplinkPackager.decode(packet)
        sequence = cast(int | str, body["sequence"])
        if body["agent_id"] != packet.agent_id or int(sequence) != packet.sequence:
            raise ValueError("uplink envelope does not match its payload")
        accepted: list[RegisteredObservation] = []
        for index, raw in enumerate(cast(list[dict[str, Any]], body["objects"])):
            model = cast(dict[str, Any], raw["model"])
            observation_id = f"{packet.agent_id}:{packet.sequence}:{index}"
            observation = RegisteredObservation(
                observation_id,
                packet.agent_id,
                str(raw["local_id"]),
                Timestamp(int(raw["timestamp_ns"])),
                np.asarray(raw["position_m"], dtype=np.float64),
                np.asarray(raw["embedding"], dtype=np.float64),
                np.asarray(raw["covariance_diagonal"], dtype=np.float64),
                float(raw["confidence"]),
                ModelVersion(
                    str(model["name"]), str(model["version"]), model.get("checksum_sha256")
                ),
            )
            accepted.append(observation)
        if persist_raw and self.journal is not None and self.journal.append(message):
            self.metrics["journaled"] += 1
        self._remember_sequence(key)
        for observation in accepted:
            self._observations[observation.observation_id] = observation
        self._enforce_observation_bound()
        self.metrics["packets"] += 1
        self.metrics["observations"] += len(accepted)
        self._prune(now)
        LOGGER.info("observations_ingested agent=%s count=%d", packet.agent_id, len(accepted))
        return tuple(accepted)

    def replay_journal(
        self,
        journal: ObservationJournal,
        *,
        now: Timestamp,
    ) -> tuple[RegisteredObservation, ...]:
        accepted: list[RegisteredObservation] = []
        messages = journal.messages()
        for message in messages:
            accepted.extend(self.ingest(message, now, persist_raw=False))
        self.metrics["journal_replays"] += len(messages)
        return tuple(accepted)

    def _prune(self, now: Timestamp) -> None:
        expired = [
            observation_id
            for observation_id, observation in self._observations.items()
            if observation.timestamp.monotonic_ns + self.retention_ns < now.monotonic_ns
        ]
        for observation_id in expired:
            del self._observations[observation_id]
            self.metrics["expired"] += 1

    def _remember_sequence(self, key: tuple[str, int]) -> None:
        self._sequences.add(key)
        self._sequence_order.append(key)
        while len(self._sequence_order) > self.max_sequence_history:
            removed = self._sequence_order.popleft()
            self._sequences.remove(removed)

    def _enforce_observation_bound(self) -> None:
        overflow = len(self._observations) - self.max_observations
        if overflow <= 0:
            return
        oldest = sorted(
            self._observations.values(),
            key=lambda item: (item.timestamp.monotonic_ns, item.observation_id),
        )[:overflow]
        for observation in oldest:
            del self._observations[observation.observation_id]
            self.metrics["evicted"] += 1

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": SNAPSHOT_SCHEMA,
            "retention_ns": self.retention_ns,
            "max_observations": self.max_observations,
            "max_sequence_history": self.max_sequence_history,
            "max_future_skew_ns": self.max_future_skew_ns,
            "sequences": [
                {"agent_id": agent_id, "sequence": sequence}
                for agent_id, sequence in self._sequence_order
            ],
            "observations": [
                {
                    "observation_id": observation.observation_id,
                    "agent_id": observation.agent_id,
                    "local_id": observation.local_id,
                    "timestamp_ns": observation.timestamp.monotonic_ns,
                    "position_m": observation.position_m.tolist(),
                    "embedding": observation.embedding.tolist(),
                    "covariance_diagonal": observation.covariance_diagonal.tolist(),
                    "confidence": observation.confidence,
                    "model_version": observation.model_version.to_dict(),
                }
                for observation in self.observations
            ],
        }

    def write_json(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as stream:
                temporary = Path(stream.name)
                json.dump(self.to_dict(), stream, indent=2, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        self.metrics["snapshots"] += 1
        return destination

    @classmethod
    def read_json(
        cls,
        path: str | Path,
        *,
        journal: ObservationJournal | None = None,
    ) -> ObservationRegistry:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("schema") != SNAPSHOT_SCHEMA:
            raise ValueError("unsupported observation registry snapshot schema")
        registry = cls(
            retention_ns=int(payload["retention_ns"]),
            max_observations=int(payload["max_observations"]),
            max_sequence_history=int(payload["max_sequence_history"]),
            max_future_skew_ns=int(payload["max_future_skew_ns"]),
            journal=journal,
        )
        raw_sequences = payload.get("sequences")
        if (
            not isinstance(raw_sequences, list)
            or len(raw_sequences) > registry.max_sequence_history
        ):
            raise ValueError("observation registry sequence history exceeds bounds")
        for raw_value in raw_sequences:
            raw = cast(dict[str, Any], raw_value)
            key = (str(raw["agent_id"]), int(raw["sequence"]))
            if key in registry._sequences:
                raise ValueError("observation registry snapshot contains duplicate sequences")
            registry._remember_sequence(key)
        raw_observations = payload.get("observations")
        if (
            not isinstance(raw_observations, list)
            or len(raw_observations) > registry.max_observations
        ):
            raise ValueError("observation registry snapshot observations exceed bounds")
        for raw_value in raw_observations:
            raw = cast(dict[str, Any], raw_value)
            observation = RegisteredObservation(
                observation_id=str(raw["observation_id"]),
                agent_id=str(raw["agent_id"]),
                local_id=str(raw["local_id"]),
                timestamp=Timestamp(int(raw["timestamp_ns"])),
                position_m=np.asarray(raw["position_m"], dtype=np.float64),
                embedding=np.asarray(raw["embedding"], dtype=np.float64),
                covariance_diagonal=np.asarray(
                    raw["covariance_diagonal"], dtype=np.float64
                ),
                confidence=float(raw["confidence"]),
                model_version=ModelVersion.from_dict(
                    cast(dict[str, Any], raw["model_version"])
                ),
            )
            if observation.observation_id in registry._observations:
                raise ValueError("observation registry snapshot contains duplicate identifiers")
            registry._observations[observation.observation_id] = observation
        registry.metrics["restores"] += 1
        return registry
