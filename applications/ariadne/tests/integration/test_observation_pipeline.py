from __future__ import annotations

from pathlib import Path

import numpy as np

from ariadne.common import ModelVersion, Timestamp
from ariadne.communications import (
    DeliveryReceipt,
    InMemoryMeshTransport,
    MessageClass,
    TransportMessage,
    UplinkPackager,
)
from ariadne.intelligence import ObservationRegistry
from ariadne.object_state import KeyframeRecord, LocalObjectStore
from ariadne.tracking import TemporalStaticFilter, TrackObservation


def test_two_wingmen_arrive_in_registry_in_timestamp_order() -> None:
    transport = InMemoryMeshTransport(seed=11)
    for agent_index, agent in enumerate(("wingman_02", "wingman_01"), start=1):
        filter_ = TemporalStaticFilter()
        track = None
        for step in range(3):
            track = filter_.update(
                TrackObservation(
                    Timestamp(step + agent_index),
                    agent,
                    "tower",
                    np.array([4.0 + agent_index * 0.05, 2.0, 0.0]),
                    np.array([1.0, 0.0, 0.0]),
                    0.01,
                    0.99,
                    0.98,
                )
            )
        assert track is not None
        store = LocalObjectStore()
        store.upsert(
            track,
            model_version=ModelVersion("reference", "1"),
            keyframe=KeyframeRecord(agent_index, Timestamp(agent_index), 0.9, "tower"),
        )
        packet = UplinkPackager().package(agent, Timestamp(agent_index), store.objects)
        transport.send(
            TransportMessage(
                f"message-{agent}",
                agent,
                "intelligence_01",
                MessageClass.OBSERVATION,
                Timestamp(agent_index),
                100,
                packet,
            )
        )
    registry = ObservationRegistry()
    for message in transport.receive("intelligence_01", Timestamp(10)):
        registry.ingest(message, Timestamp(10))
    assert [item.agent_id for item in registry.observations] == ["wingman_02", "wingman_01"]


def test_unacknowledged_delivery_is_deduplicated_after_node_restart(
    tmp_path: Path,
) -> None:
    filter_ = TemporalStaticFilter()
    track = None
    for step in range(3):
        track = filter_.update(
            TrackObservation(
                Timestamp(100 + step),
                "wingman_01",
                "tower",
                np.array([4.0, 2.0, 0.0]),
                np.array([1.0, 0.0, 0.0]),
                0.01,
                0.99,
                0.98,
            )
        )
    assert track is not None
    store = LocalObjectStore()
    store.upsert(
        track,
        model_version=ModelVersion("reference", "1"),
        keyframe=KeyframeRecord(2, Timestamp(102), 0.9, "tower"),
    )
    restored_store = LocalObjectStore.read_json(
        store.write_json(tmp_path / "local-objects.json")
    )
    packet = UplinkPackager().package(
        "wingman_01",
        Timestamp(103),
        restored_store.objects,
    )
    message = TransportMessage(
        "restart-message",
        "wingman_01",
        "intelligence_01",
        MessageClass.OBSERVATION,
        Timestamp(103),
        1_000,
        packet,
    )
    transport = InMemoryMeshTransport(max_retries=2, retry_interval_ns=10)
    assert transport.send(message)
    delivered = transport.receive("intelligence_01", Timestamp(104))
    registry = ObservationRegistry()
    assert len(registry.ingest(delivered[0], Timestamp(104))) == 1

    restored_registry = ObservationRegistry.read_json(
        registry.write_json(tmp_path / "observations.json")
    )
    assert transport.retry(Timestamp(113)) == 1
    redelivered = transport.receive("intelligence_01", Timestamp(113))
    assert restored_registry.ingest(redelivered[0], Timestamp(113)) == ()
    assert transport.acknowledge(
        DeliveryReceipt.from_message(redelivered[0], Timestamp(114))
    )
    assert transport.pending_count == 0
    assert len(restored_registry.observations) == 1
