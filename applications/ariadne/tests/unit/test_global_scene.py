from __future__ import annotations

import numpy as np
import pytest

from ariadne.common import FrameId, ModelVersion, Timestamp, TransformSE3
from ariadne.context import SceneEdge, SceneNode, UnifiedContext
from ariadne.context.scene_graph import SceneNodeKind
from ariadne.intelligence import RegisteredObservation
from ariadne.pose_correction import CorrectionApplier, CorrectionDeltaGenerator
from ariadne.splatting import (
    GaussianPrimitive,
    GlobalGaussianMap,
    ReconstructionResult,
    ReferenceGaussianSplatAdapter,
    SceneSnapshot,
    SceneSnapshotStore,
)


def observation(agent: str, index: int, offset: float = 0.0) -> RegisteredObservation:
    return RegisteredObservation(
        f"{agent}:0:{index}",
        agent,
        f"{agent}:tower",
        Timestamp(100 + index),
        np.array([4.0 + offset, 2.0, 0.0]),
        np.array([1.0, 0.2, 0.1]),
        np.array([0.04, 0.09, 0.01]),
        0.9,
        ModelVersion("reference", "1"),
    )


def transform(source: str, destination: str, translation: tuple[float, float, float]):
    return TransformSE3.from_translation_quaternion(
        FrameId(source), FrameId(destination), translation, (0.0, 0.0, 0.0, 1.0)
    )


def test_reconstruction_fuses_observations_with_provenance() -> None:
    adapter = ReferenceGaussianSplatAdapter()
    result = adapter.reconstruct(
        (observation("wingman_01", 0), observation("wingman_02", 1, 0.1)),
        timestamp=Timestamp(200),
    )
    assert result.input_observations == 2
    assert len(result.primitives) == 1
    primitive = result.primitives[0]
    assert primitive.object_id == "tower"
    assert primitive.source_observation_ids == ("wingman_01:0:0", "wingman_02:0:1")
    np.testing.assert_allclose(primitive.mean_m, [4.05, 2.0, 0.0])


def test_scene_map_versions_merges_and_rejects_time_travel() -> None:
    adapter = ReferenceGaussianSplatAdapter()
    scene = GlobalGaussianMap(max_primitives=2)
    first = adapter.reconstruct((observation("wingman_01", 0),), timestamp=Timestamp(200))
    assert scene.apply(first).revision == 1
    second = adapter.reconstruct((observation("wingman_02", 1, 0.2),), timestamp=Timestamp(300))
    snapshot = scene.apply(second)
    assert snapshot.revision == 2
    assert len(snapshot.primitives) == 1
    assert scene.metrics["merged"] == 1
    duplicate = scene.apply(first)
    assert duplicate.revision == 2
    assert scene.metrics["duplicate_updates"] == 1
    older = adapter.reconstruct((observation("wingman_03", 2),), timestamp=Timestamp(150))
    with pytest.raises(ValueError, match="time ordered"):
        scene.apply(older)


def test_scene_map_rolls_back_with_monotonic_revision_history() -> None:
    adapter = ReferenceGaussianSplatAdapter()
    scene = GlobalGaussianMap(max_history=4)
    first = scene.apply(
        adapter.reconstruct((observation("wingman_01", 0),), timestamp=Timestamp(200))
    )
    scene.apply(
        adapter.reconstruct(
            (observation("wingman_02", 1, offset=1.0),),
            timestamp=Timestamp(300),
        )
    )
    rolled_back = scene.rollback(first.revision, timestamp=Timestamp(400))
    assert rolled_back.revision == 3
    assert rolled_back.operation == "rollback"
    assert rolled_back.source_revision == 1
    np.testing.assert_allclose(rolled_back.primitives[0].mean_m, first.primitives[0].mean_m)
    assert [snapshot.revision for snapshot in scene.history] == [0, 1, 2, 3]
    assert scene.metrics["rollbacks"] == 1


def test_scene_snapshot_persists_and_restores(tmp_path) -> None:
    adapter = ReferenceGaussianSplatAdapter()
    scene = GlobalGaussianMap()
    snapshot = scene.apply(
        adapter.reconstruct((observation("wingman_01", 0),), timestamp=Timestamp(200))
    )
    path = tmp_path / "scene.json"
    snapshot.write_json(path)
    loaded = SceneSnapshot.read_json(path)
    restored = GlobalGaussianMap.restore(loaded)
    np.testing.assert_allclose(
        restored.snapshot().primitives[0].mean_m,
        snapshot.primitives[0].mean_m,
    )
    assert restored.snapshot().revision == snapshot.revision
    assert restored.history[0].operation == "restore"
    assert restored.metrics["restores"] == 1


def test_scene_snapshot_store_recovers_latest_and_bounds_retention(tmp_path) -> None:
    adapter = ReferenceGaussianSplatAdapter()
    scene = GlobalGaussianMap()
    store = SceneSnapshotStore(tmp_path / "scene-store", max_snapshots=2)
    snapshots = []
    for index, timestamp_ns in enumerate((200, 300, 400)):
        snapshot = scene.apply(
            adapter.reconstruct(
                (observation(f"wingman_{index + 1:02d}", index, offset=index * 0.1),),
                timestamp=Timestamp(timestamp_ns),
            )
        )
        snapshots.append(snapshot)
        store.persist(snapshot)

    assert [path.name for path in store.snapshot_paths] == [
        "scene-00000000000000000002.json",
        "scene-00000000000000000003.json",
    ]
    assert store.load_latest().revision == 3

    store.latest_path.write_text("{interrupted", encoding="utf-8")
    recovered = store.load_latest()
    assert recovered.revision == 3
    np.testing.assert_allclose(
        recovered.primitives[0].mean_m,
        snapshots[-1].primitives[0].mean_m,
    )
    with pytest.raises(ValueError, match="monotonic"):
        store.persist(snapshots[0])


def test_conflicting_scene_update_is_rejected_atomically() -> None:
    adapter = ReferenceGaussianSplatAdapter()
    scene = GlobalGaussianMap()
    snapshot = scene.apply(
        adapter.reconstruct((observation("wingman_01", 0),), timestamp=Timestamp(200))
    )
    primitive = snapshot.primitives[0]
    conflict = GaussianPrimitive(
        primitive.primitive_id,
        "different-object",
        primitive.mean_m,
        primitive.scale_m,
        primitive.color_rgb,
        primitive.opacity,
        primitive.confidence,
        ("conflicting-observation",),
    )
    result = ReconstructionResult(
        Timestamp(300),
        ModelVersion("conflict", "1"),
        (conflict,),
        0.1,
        1,
    )
    with pytest.raises(ValueError, match="conflicts"):
        scene.apply(result)
    assert scene.snapshot().revision == 1
    np.testing.assert_allclose(
        scene.snapshot().primitives[0].mean_m,
        primitive.mean_m,
    )


def test_correction_is_bounded_idempotent_and_expires() -> None:
    local_pose = transform("body", "local_wingman_01", (1.0, 0.0, 0.0))
    global_pose = transform("body", "global", (2.2, 0.0, 0.0))
    correction = CorrectionDeltaGenerator().generate(
        "wingman_01", local_pose, global_pose, issued_at=Timestamp(100), ttl_ns=100
    )
    applier = CorrectionApplier(max_translation_step_m=0.5)
    applied = applier.apply(local_pose, correction, now=Timestamp(150))
    assert applied.applied_fraction == pytest.approx(0.5 / 1.2)
    assert applied.corrected_pose.destination == FrameId("global")
    assert applied.corrected_pose.translation_m[0] == pytest.approx(1.5)
    with pytest.raises(ValueError, match="already"):
        applier.apply(local_pose, correction, now=Timestamp(150))
    expiring = CorrectionDeltaGenerator().generate(
        "wingman_01", local_pose, global_pose, issued_at=Timestamp(100), ttl_ns=10
    )
    with pytest.raises(ValueError, match="expired"):
        CorrectionApplier().apply(local_pose, expiring, now=Timestamp(110))


def test_context_reports_stale_degraded_state() -> None:
    context = UnifiedContext()
    context.upsert_node(
        SceneNode("wingman_01", SceneNodeKind.WINGMAN, np.zeros(3), 0.9, Timestamp(100))
    )
    context.upsert_node(
        SceneNode(
            "object_tower", SceneNodeKind.OBJECT, np.array([4.0, 2.0, 0.0]), 0.8, Timestamp(100)
        )
    )
    context.upsert_edge(SceneEdge("wingman_01", "object_tower", "observes", 0.8))
    fresh = context.snapshot(Timestamp(150), max_age_ns=100)
    assert not fresh.degraded
    assert len(fresh.edges) == 1
    stale = context.snapshot(Timestamp(201), max_age_ns=100)
    assert stale.degraded
    assert stale.stale_node_ids == ("object_tower", "wingman_01")
