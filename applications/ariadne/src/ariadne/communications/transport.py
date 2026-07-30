"""Priority-aware bounded in-memory mesh transport for deterministic tests."""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from ariadne.common import Timestamp
from ariadne.communications.uplink import UplinkPacket

LOGGER = logging.getLogger(__name__)


class MessageClass(IntEnum):
    TELEMETRY = 1
    OBSERVATION = 2
    CORRECTION = 3
    CONTROL = 4


@dataclass(frozen=True)
class TransportMessage:
    message_id: str
    source: str
    destination: str
    message_class: MessageClass
    created_at: Timestamp
    expires_at_ns: int
    packet: UplinkPacket

    def __post_init__(self) -> None:
        if not self.message_id or not self.source or not self.destination:
            raise ValueError("transport identifiers must not be empty")
        if self.expires_at_ns <= self.created_at.monotonic_ns:
            raise ValueError("transport expiry must follow creation")


@dataclass(frozen=True)
class DeliveryReceipt:
    message_id: str
    source: str
    destination: str
    acknowledged_at: Timestamp

    def __post_init__(self) -> None:
        if not self.message_id or not self.source or not self.destination:
            raise ValueError("delivery receipt identifiers must not be empty")

    @classmethod
    def from_message(
        cls, message: TransportMessage, acknowledged_at: Timestamp
    ) -> DeliveryReceipt:
        return cls(
            message.message_id,
            message.destination,
            message.source,
            acknowledged_at,
        )


@dataclass
class _PendingDelivery:
    message: TransportMessage
    attempts: int
    next_retry_ns: int
    queued: bool


class InMemoryMeshTransport:
    def __init__(
        self,
        *,
        capacity: int = 128,
        drop_probability: float = 0.0,
        seed: int = 0,
        max_retries: int = 0,
        retry_interval_ns: int = 100_000_000,
        dedupe_capacity: int | None = None,
    ) -> None:
        resolved_dedupe_capacity = capacity * 8 if dedupe_capacity is None else dedupe_capacity
        if (
            capacity <= 0
            or not 0 <= drop_probability <= 1
            or max_retries < 0
            or retry_interval_ns <= 0
            or resolved_dedupe_capacity < capacity
        ):
            raise ValueError("transport controls are invalid")
        self.capacity = capacity
        self.drop_probability = drop_probability
        self.max_retries = max_retries
        self.retry_interval_ns = retry_interval_ns
        self.dedupe_capacity = resolved_dedupe_capacity
        self._rng = np.random.default_rng(seed)
        self._queue: list[TransportMessage] = []
        self._seen: set[str] = set()
        self._seen_order: deque[str] = deque()
        self._pending: dict[str, _PendingDelivery] = {}
        self.metrics = {
            "sent": 0,
            "delivered": 0,
            "dropped": 0,
            "expired": 0,
            "duplicates": 0,
            "retries": 0,
            "acknowledged": 0,
            "retry_exhausted": 0,
            "invalid_receipts": 0,
            "outbox_full": 0,
        }

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    def send(self, message: TransportMessage) -> bool:
        self.metrics["sent"] += 1
        if message.message_id in self._seen:
            self.metrics["duplicates"] += 1
            return False
        reliable = self.max_retries > 0
        if reliable and len(self._pending) >= self.capacity:
            self.metrics["dropped"] += 1
            self.metrics["outbox_full"] += 1
            return False
        self._remember_message_id(message.message_id)
        if reliable:
            self._pending[message.message_id] = _PendingDelivery(
                message,
                attempts=0,
                next_retry_ns=message.created_at.monotonic_ns,
                queued=False,
            )
        queued = self._attempt(message, is_retry=False)
        if reliable:
            pending = self._pending[message.message_id]
            pending.queued = queued
            pending.next_retry_ns = (
                message.created_at.monotonic_ns + self.retry_interval_ns
            )
            return True
        return queued

    def _remember_message_id(self, message_id: str) -> None:
        self._seen.add(message_id)
        self._seen_order.append(message_id)
        while len(self._seen_order) > self.dedupe_capacity:
            oldest = self._seen_order.popleft()
            self._seen.remove(oldest)

    def _attempt(self, message: TransportMessage, *, is_retry: bool) -> bool:
        if is_retry:
            self.metrics["retries"] += 1
        if self._rng.random() < self.drop_probability:
            self.metrics["dropped"] += 1
            return False
        self._queue.append(message)
        self._queue.sort(key=lambda item: (-int(item.message_class), item.created_at.monotonic_ns))
        if len(self._queue) > self.capacity:
            removed = self._queue.pop()
            pending = self._pending.get(removed.message_id)
            if pending is not None:
                pending.queued = False
            self.metrics["dropped"] += 1
            return removed.message_id != message.message_id
        return True

    def receive(self, destination: str, now: Timestamp) -> tuple[TransportMessage, ...]:
        delivered: list[TransportMessage] = []
        retained: list[TransportMessage] = []
        for message in self._queue:
            if message.expires_at_ns <= now.monotonic_ns:
                self.metrics["expired"] += 1
                self._pending.pop(message.message_id, None)
            elif message.destination == destination:
                delivered.append(message)
                pending = self._pending.get(message.message_id)
                if pending is not None:
                    pending.queued = False
            else:
                retained.append(message)
        self._queue = retained
        self.metrics["delivered"] += len(delivered)
        LOGGER.debug("mesh_delivered destination=%s count=%d", destination, len(delivered))
        return tuple(delivered)

    def acknowledge(self, receipt: DeliveryReceipt) -> bool:
        pending = self._pending.get(receipt.message_id)
        if pending is None:
            self.metrics["invalid_receipts"] += 1
            return False
        message = pending.message
        if (
            receipt.source != message.destination
            or receipt.destination != message.source
            or receipt.acknowledged_at.monotonic_ns < message.created_at.monotonic_ns
            or receipt.acknowledged_at.monotonic_ns >= message.expires_at_ns
        ):
            self.metrics["invalid_receipts"] += 1
            return False
        del self._pending[receipt.message_id]
        self._queue = [
            queued for queued in self._queue if queued.message_id != receipt.message_id
        ]
        self.metrics["acknowledged"] += 1
        return True

    def retry(self, now: Timestamp) -> int:
        queued = 0
        for message_id, pending in tuple(self._pending.items()):
            message = pending.message
            if message.expires_at_ns <= now.monotonic_ns:
                self._pending.pop(message_id)
                self._queue = [
                    queued_message
                    for queued_message in self._queue
                    if queued_message.message_id != message_id
                ]
                self.metrics["expired"] += 1
                continue
            if pending.queued or now.monotonic_ns < pending.next_retry_ns:
                continue
            if pending.attempts >= self.max_retries:
                self._pending.pop(message_id)
                self.metrics["retry_exhausted"] += 1
                continue
            pending.attempts += 1
            pending.next_retry_ns = now.monotonic_ns + self.retry_interval_ns
            pending.queued = self._attempt(message, is_retry=True)
            queued += int(pending.queued)
        return queued
