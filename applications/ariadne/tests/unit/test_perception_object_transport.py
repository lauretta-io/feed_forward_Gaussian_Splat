from __future__ import annotations

import numpy as np
import pytest

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
from ariadne.perception import (
    ImagePreprocessor,
    PretrainedU2NetSaliencyDetector,
    SaliencyClusterer,
    SaliencyDetector,
)
from ariadne.replay import ImageFrame
from ariadne.tracking import StaticTrackState, TemporalStaticFilter, TrackObservation


def patterned_frame() -> ImageFrame:
    image = np.full((32, 40), 0.35)
    image[5:14, 6:16] = 0.9
    image[19:28, 25:36] = 0.05
    return ImageFrame(Timestamp(300_000_000), "wingman_01", image, 3)


def confirmed_track():
    filter_ = TemporalStaticFilter(min_confirmations=3)
    state = None
    for step in range(3):
        state = filter_.update(
            TrackObservation(
                Timestamp(step * 100_000_000),
                "wingman_01",
                "landmark",
                np.array([4.0, 2.0, 0.0]),
                np.array([1.0, 0.0, 0.0]),
                0.01,
                0.98,
                0.97,
            )
        )
    assert state is not None and state.state is StaticTrackState.STATIC_CONFIRMED
    return state


def test_preprocessing_saliency_and_regions_are_deterministic() -> None:
    processor = ImagePreprocessor(width=40, height=32, max_occlusion_fraction=0.8)
    first = processor.process(patterned_frame())
    second = processor.process(patterned_frame())
    assert first.quality.accepted
    np.testing.assert_array_equal(first.image, second.image)
    saliency = SaliencyDetector(quantile=0.75).detect(first)
    regions = SaliencyClusterer(min_area_px=3).cluster(saliency)
    assert regions
    assert all(region.area_px >= 3 for region in regions)
    assert regions == SaliencyClusterer(min_area_px=3).cluster(saliency)


def test_quality_rejection_stops_saliency() -> None:
    dark = ImageFrame(Timestamp(0), "wingman_01", np.zeros((8, 8)), 0)
    rejected = ImagePreprocessor().process(dark)
    assert not rejected.quality.accepted
    with pytest.raises(ValueError, match="quality-rejected"):
        SaliencyDetector().detect(rejected)


def test_u2net_adapter_preserves_saliency_contract(tmp_path) -> None:
    torch = pytest.importorskip("torch")

    class TestModel(torch.nn.Module):
        def forward(self, inputs):
            output = inputs[:, :1]
            return (output,)

    processed = ImagePreprocessor(
        width=40,
        height=32,
        min_blur_score=0.0,
        min_exposure_score=0.0,
        max_occlusion_fraction=1.0,
    ).process(patterned_frame())
    saliency = PretrainedU2NetSaliencyDetector(
        tmp_path / "not-needed.pth",
        input_size=32,
        device="cpu",
        model=TestModel(),
    ).detect(processed)

    assert saliency.backend == "u2net"
    assert saliency.model_version == "u2net-injected-test-model"
    assert saliency.scores.shape == processed.image.shape
    assert saliency.mask.dtype == np.bool_
    assert float(saliency.scores.min()) == 0.0
    assert float(saliency.scores.max()) == 1.0


def test_static_object_packaging_transport_and_ingest() -> None:
    store = LocalObjectStore(max_objects=4, max_keyframes_per_object=2)
    track = confirmed_track()
    record = store.upsert(
        track,
        model_version=ModelVersion("spatial-pyramid-reference", "1.0.0"),
        keyframe=KeyframeRecord(2, Timestamp(200_000_000), 0.9, "region_0001"),
    )
    assert record is not None
    packager = UplinkPackager()
    packet = packager.package("wingman_01", Timestamp(300_000_000), store.objects)
    assert UplinkPackager.decode(packet)["schema"] == "ariadne.uplink.v1"
    message = TransportMessage(
        "message-1",
        "wingman_01",
        "intelligence_01",
        MessageClass.OBSERVATION,
        Timestamp(300_000_000),
        1_000_000_000,
        packet,
    )
    transport = InMemoryMeshTransport(seed=7)
    assert transport.send(message)
    assert not transport.send(message)
    delivered = transport.receive("intelligence_01", Timestamp(400_000_000))
    assert delivered == (message,)
    registry = ObservationRegistry()
    accepted = registry.ingest(delivered[0], Timestamp(400_000_000))
    assert len(accepted) == 1
    assert accepted[0].local_id == "wingman_01:landmark"
    assert registry.ingest(delivered[0], Timestamp(400_000_000)) == ()
    assert registry.metrics["duplicates"] == 1


def test_transport_expires_messages_and_packet_bounds_fail_closed() -> None:
    packager = UplinkPackager(max_payload_bytes=129)
    record_store = LocalObjectStore()
    record_store.upsert(
        confirmed_track(),
        model_version=ModelVersion("reference", "1"),
        keyframe=KeyframeRecord(1, Timestamp(1), 1.0, "region"),
    )
    with pytest.raises(ValueError, match="exceeds"):
        packager.package("wingman_01", Timestamp(2), record_store.objects)


def test_local_object_store_snapshot_round_trip_and_rejects_stale_updates(
    tmp_path,
) -> None:
    store = LocalObjectStore(max_objects=4, max_keyframes_per_object=2)
    track = confirmed_track()
    stored = store.upsert(
        track,
        model_version=ModelVersion("reference", "1"),
        keyframe=KeyframeRecord(2, Timestamp(200_000_000), 0.9, "region"),
    )
    assert stored is not None
    snapshot = store.write_json(tmp_path / "nested" / "local-objects.json")
    restored = LocalObjectStore.read_json(snapshot)

    assert restored.max_objects == 4
    assert restored.max_keyframes_per_object == 2
    assert restored.objects[0].local_id == stored.local_id
    np.testing.assert_allclose(restored.objects[0].position_m, stored.position_m)
    np.testing.assert_allclose(restored.objects[0].embedding, stored.embedding)
    assert restored.metrics["restores"] == 1

    older_observation = TrackObservation(
        Timestamp(1),
        track.observation.agent_id,
        track.observation.track_id,
        track.observation.position_m,
        track.observation.embedding,
        track.observation.motion_residual_mps,
        track.observation.depth_consistency,
        track.observation.embedding_similarity,
    )
    stale_track = type(track)(
        older_observation,
        track.state,
        track.static_probability,
        track.observation_count,
    )
    assert (
        restored.upsert(
            stale_track,
            model_version=ModelVersion("reference", "1"),
            keyframe=KeyframeRecord(1, Timestamp(1), 0.8, "older"),
        )
        is None
    )
    assert restored.metrics["rejected"] == 1


def test_observation_registry_snapshot_bounds_replay_history_and_future_skew(
    tmp_path,
) -> None:
    store = LocalObjectStore()
    store.upsert(
        confirmed_track(),
        model_version=ModelVersion("reference", "1"),
        keyframe=KeyframeRecord(1, Timestamp(1), 1.0, "region"),
    )
    packager = UplinkPackager()
    registry = ObservationRegistry(
        retention_ns=1_000,
        max_observations=2,
        max_sequence_history=2,
        max_future_skew_ns=5,
    )
    messages = []
    for sequence, timestamp_ns in enumerate((10, 11, 12)):
        packet = packager.package("wingman_01", Timestamp(timestamp_ns), store.objects)
        message = TransportMessage(
            f"message-{sequence}",
            "wingman_01",
            "intelligence_01",
            MessageClass.OBSERVATION,
            Timestamp(timestamp_ns),
            10_000,
            packet,
        )
        messages.append(message)
        assert len(registry.ingest(message, Timestamp(timestamp_ns))) == 1

    snapshot = registry.write_json(tmp_path / "registry.json")
    restored = ObservationRegistry.read_json(snapshot)
    assert len(restored.observations) == 2
    assert registry.metrics["evicted"] == 1
    assert restored.metrics["restores"] == 1
    assert restored.ingest(messages[-1], Timestamp(12)) == ()
    assert restored.metrics["duplicates"] == 1
    assert len(restored.ingest(messages[0], Timestamp(10))) == 1

    future_packet = packager.package("wingman_01", Timestamp(100), store.objects)
    future = TransportMessage(
        "future",
        "wingman_01",
        "intelligence_01",
        MessageClass.OBSERVATION,
        Timestamp(100),
        10_000,
        future_packet,
    )
    assert restored.ingest(future, Timestamp(90)) == ()
    assert restored.metrics["future"] == 1


def test_reliable_transport_retries_until_acknowledged() -> None:
    packet = UplinkPackager().package("wingman_01", Timestamp(100), ())
    message = TransportMessage(
        "reliable-message",
        "wingman_01",
        "intelligence_01",
        MessageClass.OBSERVATION,
        Timestamp(100),
        1_000,
        packet,
    )
    transport = InMemoryMeshTransport(
        drop_probability=0.5,
        seed=2,
        max_retries=3,
        retry_interval_ns=10,
    )

    assert transport.send(message)
    assert transport.pending_count == 1
    assert transport.receive("intelligence_01", Timestamp(100)) == ()
    assert transport.retry(Timestamp(110)) == 0
    assert transport.retry(Timestamp(120)) == 1
    assert transport.receive("intelligence_01", Timestamp(120)) == (message,)
    receipt = DeliveryReceipt.from_message(message, Timestamp(121))
    assert transport.acknowledge(receipt)
    assert transport.pending_count == 0
    assert transport.retry(Timestamp(140)) == 0
    assert transport.metrics["retries"] == 2
    assert transport.metrics["acknowledged"] == 1


def test_reliable_transport_rejects_spoofed_ack_and_bounds_retries() -> None:
    packet = UplinkPackager().package("wingman_01", Timestamp(100), ())
    message = TransportMessage(
        "bounded-message",
        "wingman_01",
        "intelligence_01",
        MessageClass.OBSERVATION,
        Timestamp(100),
        1_000,
        packet,
    )
    transport = InMemoryMeshTransport(
        drop_probability=1.0,
        max_retries=2,
        retry_interval_ns=10,
    )
    assert transport.send(message)
    spoofed = DeliveryReceipt(
        message.message_id,
        "untrusted-node",
        message.source,
        Timestamp(101),
    )
    assert not transport.acknowledge(spoofed)
    assert transport.retry(Timestamp(110)) == 0
    assert transport.retry(Timestamp(120)) == 0
    assert transport.retry(Timestamp(130)) == 0
    assert transport.pending_count == 0
    assert transport.metrics["retry_exhausted"] == 1
    assert transport.metrics["invalid_receipts"] == 1


def test_reliable_transport_rejects_messages_when_outbox_is_full() -> None:
    transport = InMemoryMeshTransport(
        capacity=1,
        max_retries=1,
        retry_interval_ns=10,
    )
    packet = UplinkPackager().package("wingman_01", Timestamp(100), ())
    first = TransportMessage(
        "first",
        "wingman_01",
        "intelligence_01",
        MessageClass.OBSERVATION,
        Timestamp(100),
        1_000,
        packet,
    )
    second = TransportMessage(
        "second",
        "wingman_01",
        "intelligence_01",
        MessageClass.OBSERVATION,
        Timestamp(101),
        1_000,
        packet,
    )
    assert transport.send(first)
    assert not transport.send(second)
    assert transport.pending_count == 1
    assert transport.metrics["outbox_full"] == 1
