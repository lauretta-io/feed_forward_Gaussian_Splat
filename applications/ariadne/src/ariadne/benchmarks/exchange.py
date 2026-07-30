"""Wingman perception-to-Intelligence exchange reference benchmark."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter_ns

import numpy as np

from ariadne.common import ModelVersion, Timestamp
from ariadne.communications import (
    DeliveryReceipt,
    InMemoryMeshTransport,
    MessageClass,
    TransportMessage,
    UplinkPackager,
)
from ariadne.datasets import DatasetEvaluation
from ariadne.intelligence import ObservationJournal, ObservationRegistry
from ariadne.object_state import KeyframeRecord, LocalObjectStore
from ariadne.perception import ImagePreprocessor, SaliencyClusterer, SaliencyDetector
from ariadne.replay import ImageFrame
from ariadne.tracking import TemporalStaticFilter, TrackObservation


def run_exchange_benchmark(seed: int = 7) -> DatasetEvaluation:
    start_ns = perf_counter_ns()
    rng = np.random.default_rng(seed)
    image = np.full((32, 40), 0.35)
    image[5:14, 6:16] = 0.9
    image[19:28, 25:36] = 0.05
    image += rng.normal(0.0, 0.002, image.shape)
    frame = ImageFrame(Timestamp(500_000_000), "wingman_01", image, 5)
    processor = ImagePreprocessor(width=40, height=32, max_occlusion_fraction=0.8)
    processed = processor.process(frame)
    detector = SaliencyDetector(quantile=0.75)
    saliency = detector.detect(processed)
    regions = SaliencyClusterer(min_area_px=3).cluster(saliency)

    static_filter = TemporalStaticFilter()
    store = LocalObjectStore(max_objects=8, max_keyframes_per_object=3)
    states: list[str] = []
    for step in range(5):
        track = static_filter.update(
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
        states.append(track.state.value)
        if regions:
            store.upsert(
                track,
                model_version=ModelVersion("spatial-pyramid-reference", "1.0.0"),
                keyframe=KeyframeRecord(
                    step,
                    track.observation.timestamp,
                    min(1.0, 0.8 + step * 0.03),
                    regions[0].region_id,
                ),
            )

    with TemporaryDirectory(prefix="ariadne-exchange-") as temporary_directory:
        object_snapshot = store.write_json(Path(temporary_directory) / "local-objects.json")
        restored_store = LocalObjectStore.read_json(object_snapshot)
        packager = UplinkPackager()
        packet = packager.package(
            "wingman_01",
            Timestamp(500_000_000),
            restored_store.objects,
        )
    transport = InMemoryMeshTransport(
        seed=seed,
        max_retries=2,
        retry_interval_ns=50_000_000,
    )
    message = TransportMessage(
        "exchange-0001",
        "wingman_01",
        "intelligence_01",
        MessageClass.OBSERVATION,
        Timestamp(500_000_000),
        2_000_000_000,
        packet,
    )
    transport.send(message)
    delivered = transport.receive("intelligence_01", Timestamp(600_000_000))
    with TemporaryDirectory(prefix="ariadne-registry-") as temporary_directory:
        registry_root = Path(temporary_directory)
        journal = ObservationJournal(
            registry_root / "journal",
            max_entries_per_segment=2,
            max_segments=2,
        )
        registry = ObservationRegistry(journal=journal)
        accepted = (
            registry.ingest(delivered[0], Timestamp(600_000_000)) if delivered else ()
        )
        rebuilt_registry = ObservationRegistry()
        rebuilt = rebuilt_registry.replay_journal(journal, now=Timestamp(600_000_000))
        journal_entries = journal.entry_count
        registry_snapshot = registry.write_json(registry_root / "registry.json")
        registry = ObservationRegistry.read_json(registry_snapshot, journal=journal)
        transport.retry(Timestamp(650_000_000))
        redelivered = transport.receive("intelligence_01", Timestamp(650_000_000))
        duplicate_accepted = (
            registry.ingest(redelivered[0], Timestamp(650_000_000))
            if redelivered
            else ()
        )
        acknowledged = bool(
            delivered
            and transport.acknowledge(
                DeliveryReceipt.from_message(delivered[0], Timestamp(651_000_000))
            )
        )
    metrics: dict[str, int | float | str] = {
        "seed": seed,
        "preprocessing_latency_ms": processed.latency_ms,
        "blur_score": processed.quality.blur_score,
        "exposure_score": processed.quality.exposure_score,
        "saliency_latency_ms": saliency.latency_ms,
        "salient_fraction": float(np.mean(saliency.mask)),
        "saliency_region_count": len(regions),
        "local_object_count": len(store.objects),
        "local_object_snapshot_restored": restored_store.metrics["restores"],
        "keyframe_count": sum(len(record.keyframes) for record in store.objects),
        "uplink_bytes": len(packet.payload),
        "transport_delivered": transport.metrics["delivered"],
        "transport_retries": transport.metrics["retries"],
        "transport_acknowledged": transport.metrics["acknowledged"],
        "transport_pending": transport.pending_count,
        "registry_observation_count": len(accepted),
        "registry_duplicate_packets": registry.metrics["duplicates"],
        "registry_snapshot_restored": registry.metrics["restores"],
        "registry_journal_entries": journal_entries,
        "registry_journal_replayed_observations": len(rebuilt),
        "benchmark_latency_ms": (perf_counter_ns() - start_ns) / 1e6,
    }
    passed = (
        processed.quality.accepted
        and bool(regions)
        and len(store.objects) == 1
        and len(restored_store.objects) == 1
        and len(delivered) == 1
        and len(redelivered) == 1
        and len(accepted) == 1
        and len(rebuilt) == 1
        and journal_entries == 1
        and not duplicate_accepted
        and acknowledged
        and transport.pending_count == 0
    )
    return DatasetEvaluation(
        dataset="exchange-reference",
        status="passed" if passed else "failed",
        agents=("wingman_01", "intelligence_01"),
        modalities=("image", "saliency", "static_object", "mesh_uplink"),
        metrics=metrics,
        details={
            "processed_image": processed.image[::2, ::2].round(3).tolist(),
            "saliency_scores": saliency.scores[::2, ::2].round(3).tolist(),
            "regions": [
                {
                    "id": region.region_id,
                    "bbox_xyxy": region.bbox_xyxy,
                    "centroid_xy": region.centroid_xy,
                    "area_px": region.area_px,
                    "mean_score": region.mean_score,
                }
                for region in regions
            ],
            "tracking_states": states,
            "local_objects": [
                {
                    "local_id": record.local_id,
                    "confidence": record.confidence,
                    "keyframes": [keyframe.frame_index for keyframe in record.keyframes],
                }
                for record in store.objects
            ],
            "transport_metrics": transport.metrics,
            "registry_ids": [observation.observation_id for observation in accepted],
        },
    )
