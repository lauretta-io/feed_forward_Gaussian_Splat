"""HMAC reference for identity, integrity, expiry, and replay protection."""

from __future__ import annotations

import hashlib
import hmac
import logging
from collections import deque
from dataclasses import dataclass

from ariadne.common import Timestamp

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SecureEnvelope:
    source: str
    destination: str
    nonce: str
    issued_at: Timestamp
    expires_at_ns: int
    payload: bytes
    signature_sha256: str

    def __post_init__(self) -> None:
        if not self.source or not self.destination or not self.nonce or not self.payload:
            raise ValueError("secure envelope fields must not be empty")
        if self.expires_at_ns <= self.issued_at.monotonic_ns:
            raise ValueError("secure envelope expiry must follow issue time")


class HmacEnvelopeSecurity:
    def __init__(
        self,
        keys: dict[str, bytes],
        *,
        replay_window_entries: int = 4096,
        max_ttl_ns: int = 60_000_000_000,
    ) -> None:
        if not keys or any(not identity or len(key) < 16 for identity, key in keys.items()):
            raise ValueError("each identity needs a key of at least 16 bytes")
        if replay_window_entries <= 0 or max_ttl_ns <= 0:
            raise ValueError("security replay and TTL bounds must be positive")
        self._keys = dict(keys)
        self.replay_window_entries = replay_window_entries
        self.max_ttl_ns = max_ttl_ns
        self._seen: set[tuple[str, str]] = set()
        self._seen_order: deque[tuple[str, str]] = deque()
        self.metrics = {"signed": 0, "verified": 0, "rejected": 0, "replays": 0}

    @staticmethod
    def _message(
        source: str,
        destination: str,
        nonce: str,
        issued_at_ns: int,
        expires_at_ns: int,
        payload: bytes,
    ) -> bytes:
        header = f"{source}\0{destination}\0{nonce}\0{issued_at_ns}\0{expires_at_ns}\0".encode()
        return header + payload

    def sign(
        self,
        source: str,
        destination: str,
        nonce: str,
        issued_at: Timestamp,
        expires_at_ns: int,
        payload: bytes,
    ) -> SecureEnvelope:
        key = self._keys.get(source)
        if key is None:
            raise ValueError("unknown signing identity")
        if not 0 < expires_at_ns - issued_at.monotonic_ns <= self.max_ttl_ns:
            raise ValueError("secure envelope TTL exceeds the configured bound")
        message = self._message(
            source, destination, nonce, issued_at.monotonic_ns, expires_at_ns, payload
        )
        signature = hmac.new(key, message, hashlib.sha256).hexdigest()
        self.metrics["signed"] += 1
        return SecureEnvelope(
            source, destination, nonce, issued_at, expires_at_ns, payload, signature
        )

    def verify(self, envelope: SecureEnvelope, *, destination: str, now: Timestamp) -> bytes:
        replay_key = (envelope.source, envelope.nonce)
        if replay_key in self._seen:
            self.metrics["replays"] += 1
            raise ValueError("secure envelope replay detected")
        if destination != envelope.destination or now.monotonic_ns >= envelope.expires_at_ns:
            self.metrics["rejected"] += 1
            raise ValueError("secure envelope destination or expiry is invalid")
        key = self._keys.get(envelope.source)
        if key is None:
            self.metrics["rejected"] += 1
            raise ValueError("unknown signing identity")
        message = self._message(
            envelope.source,
            envelope.destination,
            envelope.nonce,
            envelope.issued_at.monotonic_ns,
            envelope.expires_at_ns,
            envelope.payload,
        )
        expected = hmac.new(key, message, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, envelope.signature_sha256):
            self.metrics["rejected"] += 1
            raise ValueError("secure envelope signature is invalid")
        self._seen.add(replay_key)
        self._seen_order.append(replay_key)
        while len(self._seen_order) > self.replay_window_entries:
            self._seen.discard(self._seen_order.popleft())
        self.metrics["verified"] += 1
        LOGGER.debug("secure_envelope_verified source=%s", envelope.source)
        return envelope.payload
