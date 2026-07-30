"""Versioned, checksummed, compressed Wingman observation packaging."""

from __future__ import annotations

import hashlib
import json
import logging
import zlib
from dataclasses import dataclass
from typing import cast

from ariadne.common import Timestamp
from ariadne.object_state import LocalObjectRecord

LOGGER = logging.getLogger(__name__)
SCHEMA_VERSION = "ariadne.uplink.v1"


@dataclass(frozen=True)
class UplinkPacket:
    agent_id: str
    timestamp: Timestamp
    sequence: int
    payload: bytes
    checksum_sha256: str
    object_count: int

    def __post_init__(self) -> None:
        if not self.agent_id or self.sequence < 0 or self.object_count < 0:
            raise ValueError("uplink packet metadata is invalid")
        if hashlib.sha256(self.payload).hexdigest() != self.checksum_sha256:
            raise ValueError("uplink packet checksum mismatch")


class UplinkPackager:
    def __init__(self, *, max_payload_bytes: int = 64_000, embedding_decimals: int = 4) -> None:
        if max_payload_bytes <= 128 or not 0 <= embedding_decimals <= 8:
            raise ValueError("uplink packaging controls are invalid")
        self.max_payload_bytes = max_payload_bytes
        self.embedding_decimals = embedding_decimals
        self._sequence_by_agent: dict[str, int] = {}
        self.metrics = {"packets": 0, "objects": 0, "bytes": 0}

    def package(
        self,
        agent_id: str,
        timestamp: Timestamp,
        records: tuple[LocalObjectRecord, ...],
    ) -> UplinkPacket:
        if not agent_id or any(record.agent_id != agent_id for record in records):
            raise ValueError("all uplink records must belong to the packet agent")
        sequence = self._sequence_by_agent.get(agent_id, 0)
        body = {
            "schema": SCHEMA_VERSION,
            "agent_id": agent_id,
            "timestamp_ns": timestamp.monotonic_ns,
            "sequence": sequence,
            "objects": [
                {
                    "local_id": record.local_id,
                    "timestamp_ns": record.timestamp.monotonic_ns,
                    "position_m": record.position_m.round(6).tolist(),
                    "embedding": record.embedding.round(self.embedding_decimals).tolist(),
                    "covariance_diagonal": record.covariance_diagonal.round(6).tolist(),
                    "confidence": round(record.confidence, 6),
                    "model": record.model_version.to_dict(),
                    "observation_count": record.observation_count,
                    "keyframes": [keyframe.frame_index for keyframe in record.keyframes],
                }
                for record in records
            ],
        }
        payload = zlib.compress(json.dumps(body, sort_keys=True, separators=(",", ":")).encode())
        if len(payload) > self.max_payload_bytes:
            raise ValueError(
                f"compressed uplink payload exceeds {self.max_payload_bytes} bytes: {len(payload)}"
            )
        self._sequence_by_agent[agent_id] = sequence + 1
        self.metrics["packets"] += 1
        self.metrics["objects"] += len(records)
        self.metrics["bytes"] += len(payload)
        LOGGER.debug(
            "uplink_packaged agent=%s sequence=%d bytes=%d", agent_id, sequence, len(payload)
        )
        return UplinkPacket(
            agent_id,
            timestamp,
            sequence,
            payload,
            hashlib.sha256(payload).hexdigest(),
            len(records),
        )

    @staticmethod
    def decode(packet: UplinkPacket) -> dict[str, object]:
        if hashlib.sha256(packet.payload).hexdigest() != packet.checksum_sha256:
            raise ValueError("uplink packet checksum mismatch")
        decoded = cast(dict[str, object], json.loads(zlib.decompress(packet.payload)))
        if decoded.get("schema") != SCHEMA_VERSION:
            raise ValueError("unsupported uplink schema")
        return decoded
