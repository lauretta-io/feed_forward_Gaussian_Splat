"""Global reconstruction, correction, and unified-context reference benchmark."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter_ns

import numpy as np

from ariadne.common import FrameId, ModelVersion, Timestamp, TransformSE3
from ariadne.context import SceneEdge, SceneNode, UnifiedContext
from ariadne.context.scene_graph import SceneNodeKind
from ariadne.datasets import DatasetEvaluation
from ariadne.intelligence import RegisteredObservation
from ariadne.optimization import RobustSE3PoseGraph, SE3PoseConstraint
from ariadne.pose_correction import CorrectionApplier, CorrectionDeltaGenerator
from ariadne.splatting import (
    GaussianBackendRegistry,
    GaussianReconstructionExecutor,
    GlobalGaussianMap,
    ReconstructionLimits,
    ReferenceGaussianSplatAdapter,
    SceneSnapshotStore,
)


def _pose(destination: str, x: float) -> TransformSE3:
    return TransformSE3.from_translation_quaternion(
        FrameId("body"),
        FrameId(destination),
        (x, 0.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )


def run_global_scene_benchmark(seed: int = 7) -> DatasetEvaluation:
    start_ns = perf_counter_ns()
    rng = np.random.default_rng(seed)
    observations = tuple(
        RegisteredObservation(
            f"{agent}:0:0",
            agent,
            f"{agent}:tower",
            Timestamp(100 + index),
            np.array([4.0 + index * 0.08, 2.0, 0.0]) + rng.normal(0.0, 0.002, 3),
            np.array([1.0, 0.2, 0.1]),
            np.array([0.04, 0.09, 0.01]),
            0.92,
            ModelVersion("spatial-pyramid-reference", "1.0.0"),
        )
        for index, agent in enumerate(("wingman_01", "wingman_02"))
    )
    backend_registry = GaussianBackendRegistry(max_backends=2)
    backend_registry.register("reference", ReferenceGaussianSplatAdapter)
    executor = GaussianReconstructionExecutor(
        backend_registry.create("reference"),
        limits=ReconstructionLimits(
            max_observations=4,
            max_objects=2,
            max_estimated_memory_bytes=16_384,
        ),
        device="cpu",
    )
    reconstruction = executor.reconstruct(observations, timestamp=Timestamp(200))
    diagnostics = reconstruction.diagnostics
    if diagnostics is None:
        raise RuntimeError("Gaussian executor did not attach diagnostics")
    scene = GlobalGaussianMap()
    scene_snapshot = scene.apply(reconstruction)
    with TemporaryDirectory(prefix="ariadne-global-scene-") as temporary_directory:
        snapshot_store = SceneSnapshotStore(
            Path(temporary_directory),
            max_snapshots=2,
        )
        snapshot_store.persist(scene_snapshot)
        persisted_snapshot = snapshot_store.load_latest()
        restored_scene = GlobalGaussianMap.restore(persisted_snapshot)
        persisted_snapshot_count = len(snapshot_store.snapshot_paths)

    yaw_90 = np.array([0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)])
    graph = RobustSE3PoseGraph("wingman_01_t0")
    graph_constraints = (
        ("wingman_01_t0", "wingman_01_t1", (1.0, 0.0, 0.0), yaw_90, 10.0),
        ("wingman_01_t1", "object_tower", (1.0, 0.0, 0.0), np.array([0, 0, 0, 1]), 10.0),
        ("wingman_01_t0", "object_tower", (1.0, 1.0, 0.0), yaw_90, 9.0),
        ("wingman_01_t0", "object_tower", (8.0, -4.0, 0.0), np.array([0, 0, 0, 1]), 0.1),
        ("wingman_02_t0", "wingman_02_t1", (0.0, 0.0, 2.0), np.array([0, 0, 0, 1]), 10.0),
    )
    for source, destination, translation, quaternion, information in graph_constraints:
        graph.add_constraint(
            SE3PoseConstraint(
                source,
                destination,
                np.asarray(translation),
                quaternion,
                np.eye(6) * 0.01,
                information,
            )
        )
    graph_result = graph.optimize()

    local_pose = _pose("local_wingman_01", 1.0)
    optimized_pose = _pose("global", 1.8)
    correction_generator = CorrectionDeltaGenerator(max_history=4)
    correction = correction_generator.generate(
        "wingman_01", local_pose, optimized_pose, issued_at=Timestamp(200)
    )
    correction_applier = CorrectionApplier(
        max_translation_step_m=0.5,
        max_applied_history=4,
    )
    applied = correction_applier.apply(
        local_pose, correction, now=Timestamp(210)
    )
    with TemporaryDirectory(prefix="ariadne-optimization-state-") as temporary_directory:
        state_root = Path(temporary_directory)
        restored_graph = RobustSE3PoseGraph.read_json(
            graph.write_json(state_root / "pose-graph.json")
        )
        restored_graph_result = restored_graph.optimize()
        restored_generator = CorrectionDeltaGenerator.read_json(
            correction_generator.write_json(state_root / "correction-generator.json")
        )
        restored_applier = CorrectionApplier.read_json(
            correction_applier.write_json(state_root / "correction-applier.json")
        )
        duplicate_correction_rejected = False
        try:
            restored_applier.apply(local_pose, correction, now=Timestamp(211))
        except ValueError:
            duplicate_correction_rejected = True
        next_correction = restored_generator.generate(
            "wingman_01",
            local_pose,
            optimized_pose,
            issued_at=Timestamp(212),
        )

    context = UnifiedContext()
    context.upsert_node(
        SceneNode(
            "wingman_01",
            SceneNodeKind.WINGMAN,
            applied.corrected_pose.translation_m,
            correction.confidence,
            Timestamp(210),
        )
    )
    for primitive in scene_snapshot.primitives:
        context.upsert_node(
            SceneNode(
                f"object_{primitive.object_id}",
                SceneNodeKind.OBJECT,
                primitive.mean_m,
                primitive.confidence,
                Timestamp(210),
            )
        )
        context.upsert_edge(
            SceneEdge("wingman_01", f"object_{primitive.object_id}", "observes", 0.9)
        )
    context_snapshot = context.snapshot(Timestamp(220))
    metrics: dict[str, int | float | str] = {
        "seed": seed,
        "input_observations": reconstruction.input_observations,
        "gaussian_primitive_count": len(reconstruction.primitives),
        "reconstruction_latency_ms": reconstruction.latency_ms,
        "reconstruction_backend": diagnostics.backend,
        "reconstruction_estimated_memory_bytes": diagnostics.estimated_working_set_bytes,
        "reconstruction_executor_completed": executor.metrics["completed"],
        "scene_revision": scene_snapshot.revision,
        "scene_history_depth": len(scene.history),
        "scene_snapshot_bytes": len(
            json.dumps(scene_snapshot.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        ),
        "scene_persisted_snapshot_count": persisted_snapshot_count,
        "scene_restored_revision": restored_scene.snapshot().revision,
        "se3_graph_component_count": len(set(graph_result.components.values())),
        "se3_graph_rejected_constraints": len(graph_result.rejected_constraints),
        "se3_graph_revision": graph_result.revision,
        "se3_graph_restored_revision": restored_graph_result.revision,
        "se3_graph_state_restored": restored_graph.metrics["restores"],
        "se3_translation_rmse_m": graph_result.translation_rmse_m,
        "se3_rotation_rmse_rad": graph_result.rotation_rmse_rad,
        "correction_magnitude_m": float(np.linalg.norm(correction.local_to_global.translation_m)),
        "correction_applied_fraction": applied.applied_fraction,
        "correction_history_depth": len(correction_generator.history),
        "correction_state_restored": restored_applier.metrics["restores"],
        "correction_duplicate_after_restart_rejected": int(
            duplicate_correction_rejected
        ),
        "correction_next_sequence_preserved": int(
            next_correction.correction_id == "correction_wingman_01_000001"
        ),
        "context_node_count": len(context_snapshot.nodes),
        "context_edge_count": len(context_snapshot.edges),
        "context_degraded": int(context_snapshot.degraded),
        "benchmark_latency_ms": (perf_counter_ns() - start_ns) / 1e6,
    }
    passed = (
        len(reconstruction.primitives) == 1
        and executor.metrics["completed"] == 1
        and scene_snapshot.revision == 1
        and restored_scene.snapshot().revision == scene_snapshot.revision
        and persisted_snapshot_count == 1
        and len(scene.history) == 2
        and len(graph_result.rejected_constraints) == 1
        and len(set(graph_result.components.values())) == 2
        and restored_graph_result.revision == graph_result.revision
        and 0 < applied.applied_fraction < 1
        and duplicate_correction_rejected
        and next_correction.correction_id == "correction_wingman_01_000001"
        and len(context_snapshot.nodes) == 2
        and not context_snapshot.degraded
    )
    return DatasetEvaluation(
        dataset="global-scene-reference",
        status="passed" if passed else "failed",
        agents=("wingman_01", "wingman_02", "intelligence_01"),
        modalities=("static_observations", "gaussians", "correction", "scene_graph"),
        metrics=metrics,
        details={
            "gaussians": [
                {
                    "id": primitive.primitive_id,
                    "mean_m": primitive.mean_m.tolist(),
                    "scale_m": primitive.scale_m.tolist(),
                    "color_rgb": primitive.color_rgb.tolist(),
                    "opacity": primitive.opacity,
                    "sources": primitive.source_observation_ids,
                }
                for primitive in scene_snapshot.primitives
            ],
            "scene_map": {
                "revision": scene_snapshot.revision,
                "history_revisions": [snapshot.revision for snapshot in scene.history],
                "operation": scene_snapshot.operation,
                "snapshot_bytes": metrics["scene_snapshot_bytes"],
                "rollback_supported": True,
                "restore_supported": True,
                "persistent_store": "ariadne.global-scene-snapshot.v1",
                "persisted_snapshots": persisted_snapshot_count,
            },
            "correction": {
                "id": correction.correction_id,
                "local_pose_m": local_pose.translation_m.tolist(),
                "requested_pose_m": optimized_pose.translation_m.tolist(),
                "applied_pose_m": applied.corrected_pose.translation_m.tolist(),
                "fraction": applied.applied_fraction,
            },
            "pose_graph": {
                "nodes": [
                    {
                        "id": node,
                        "position_m": graph_result.poses[node][:3, 3].tolist(),
                        "component": graph_result.components[node],
                        "covariance_trace": float(np.trace(graph_result.covariances[node])),
                    }
                    for node in sorted(graph_result.poses)
                ],
                "rejected": list(graph_result.rejected_constraints),
                "translation_rmse_m": graph_result.translation_rmse_m,
                "rotation_rmse_rad": graph_result.rotation_rmse_rad,
            },
            "context": {
                "nodes": [
                    {
                        "id": node.node_id,
                        "kind": node.kind.value,
                        "position_m": node.position_m.tolist(),
                    }
                    for node in context_snapshot.nodes
                ],
                "edges": [
                    {
                        "source": edge.source,
                        "destination": edge.destination,
                        "relation": edge.relation,
                    }
                    for edge in context_snapshot.edges
                ],
            },
        },
    )
