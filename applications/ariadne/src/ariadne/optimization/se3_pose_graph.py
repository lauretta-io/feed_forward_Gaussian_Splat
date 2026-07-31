"""Robust covariance-aware SE(3) pose-graph forest reference."""

from __future__ import annotations

import json
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, cast

import numpy as np
import numpy.typing as npt

SNAPSHOT_SCHEMA = "ariadne.se3-pose-graph.v1"


def _vector(value: npt.ArrayLike, size: int, name: str) -> npt.NDArray[np.float64]:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite vector of length {size}")
    return result.copy()


def _rotation(quaternion_xyzw: npt.ArrayLike) -> npt.NDArray[np.float64]:
    x, y, z, w = _vector(quaternion_xyzw, 4, "quaternion_xyzw")
    norm = float(np.linalg.norm((x, y, z, w)))
    if norm <= np.finfo(float).eps:
        raise ValueError("quaternion norm must be non-zero")
    x, y, z, w = np.asarray((x, y, z, w)) / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _rotation_angle(rotation: npt.NDArray[np.float64]) -> float:
    return float(np.arccos(np.clip((float(np.trace(rotation)) - 1.0) / 2.0, -1.0, 1.0)))


def _project_rotation(matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    left, _, right_transpose = np.linalg.svd(matrix)
    rotation = left @ right_transpose
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right_transpose
    return np.asarray(rotation, dtype=np.float64)


def _transform(translation: npt.ArrayLike, quaternion: npt.ArrayLike) -> npt.NDArray[np.float64]:
    matrix = np.eye(4)
    matrix[:3, :3] = _rotation(quaternion)
    matrix[:3, 3] = _vector(translation, 3, "translation_m")
    return matrix


@dataclass(frozen=True)
class SE3PoseConstraint:
    source: str
    destination: str
    translation_m: npt.NDArray[np.float64]
    quaternion_xyzw: npt.NDArray[np.float64]
    covariance: npt.NDArray[np.float64]
    information: float = 1.0
    kind: str = "odometry"

    def __post_init__(self) -> None:
        if not self.source or not self.destination or self.source == self.destination:
            raise ValueError("constraint endpoints must be distinct and non-empty")
        if not np.isfinite(self.information) or self.information <= 0:
            raise ValueError("constraint information must be finite and positive")
        translation = _vector(self.translation_m, 3, "translation_m")
        quaternion = _vector(self.quaternion_xyzw, 4, "quaternion_xyzw")
        _rotation(quaternion)
        covariance = np.asarray(self.covariance, dtype=np.float64)
        if covariance.shape != (6, 6) or not np.all(np.isfinite(covariance)):
            raise ValueError("constraint covariance must be a finite 6x6 matrix")
        if (
            not np.allclose(covariance, covariance.T)
            or np.min(np.linalg.eigvalsh(covariance)) < -1e-12
        ):
            raise ValueError("constraint covariance must be symmetric positive semidefinite")
        object.__setattr__(self, "translation_m", translation)
        object.__setattr__(self, "quaternion_xyzw", quaternion)
        object.__setattr__(self, "covariance", covariance.copy())

    @property
    def matrix(self) -> npt.NDArray[np.float64]:
        return _transform(self.translation_m, self.quaternion_xyzw)

    @staticmethod
    def _covariance_block_information(
        covariance: npt.NDArray[np.float64],
        block: slice,
        fallback: float,
    ) -> float:
        variance = float(np.trace(covariance[block, block]) / 3.0)
        if variance <= np.finfo(np.float64).eps:
            return fallback
        return 1.0 / variance

    @property
    def translation_information(self) -> float:
        """Scalar translation weight derived from the declared covariance."""
        return self._covariance_block_information(
            self.covariance,
            slice(0, 3),
            self.information,
        )

    @property
    def rotation_information(self) -> float:
        """Scalar rotation weight derived from the declared covariance."""
        return self._covariance_block_information(
            self.covariance,
            slice(3, 6),
            self.information,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "destination": self.destination,
            "translation_m": self.translation_m.tolist(),
            "quaternion_xyzw": self.quaternion_xyzw.tolist(),
            "covariance": self.covariance.tolist(),
            "information": self.information,
            "kind": self.kind,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SE3PoseConstraint:
        return cls(
            str(payload["source"]),
            str(payload["destination"]),
            np.asarray(payload["translation_m"], dtype=np.float64),
            np.asarray(payload["quaternion_xyzw"], dtype=np.float64),
            np.asarray(payload["covariance"], dtype=np.float64),
            float(payload["information"]),
            str(payload["kind"]),
        )


@dataclass(frozen=True)
class SE3PoseGraphResult:
    poses: dict[str, npt.NDArray[np.float64]]
    covariances: dict[str, npt.NDArray[np.float64]]
    components: dict[str, int]
    rejected_constraints: tuple[int, ...]
    translation_rmse_m: float
    rotation_rmse_rad: float
    revision: int = 0
    constraint_count: int = 0
    translation_rationalized: bool = False
    rationalization_constraint_count: int = 0
    rotation_rationalized: bool = False
    rotation_rationalization_constraint_count: int = 0
    rotation_rationalization_iterations: int = 0


class RobustSE3PoseGraph:
    """Robust forest initialization with optional all-factor SE(3) rationalization."""

    def __init__(
        self,
        anchor: str,
        *,
        max_constraints: int = 16_384,
        max_results: int = 32,
    ) -> None:
        if not anchor or max_constraints <= 0 or max_results <= 0:
            raise ValueError("pose graph identity and bounds must be valid")
        self.anchor = anchor
        self.max_constraints = max_constraints
        self.max_results = max_results
        self._constraints: list[SE3PoseConstraint] = []
        self._constraint_keys: set[tuple[object, ...]] = set()
        self._revision = 0
        self._history: deque[SE3PoseGraphResult] = deque()
        self.metrics = {
            "constraints": 0,
            "duplicates": 0,
            "optimizations": 0,
            "restores": 0,
        }

    @property
    def revision(self) -> int:
        return self._revision

    @property
    def history(self) -> tuple[SE3PoseGraphResult, ...]:
        return tuple(self._history)

    def add_constraint(self, constraint: SE3PoseConstraint) -> bool:
        key = self._constraint_key(constraint)
        if key in self._constraint_keys:
            self.metrics["duplicates"] += 1
            return False
        if len(self._constraints) >= self.max_constraints:
            raise ValueError("pose graph constraint capacity exceeded")
        self._constraints.append(constraint)
        self._constraint_keys.add(key)
        self._revision += 1
        self.metrics["constraints"] += 1
        return True

    def optimize(
        self,
        *,
        translation_gate_m: float = 0.75,
        rotation_gate_rad: float = 0.35,
        rationalize_translations: bool = False,
        rationalize_se3: bool = False,
        max_rotation_iterations: int = 50,
        rotation_tolerance_rad: float = 1e-7,
    ) -> SE3PoseGraphResult:
        if (
            translation_gate_m <= 0
            or rotation_gate_rad <= 0
            or max_rotation_iterations <= 0
            or not np.isfinite(rotation_tolerance_rad)
            or rotation_tolerance_rad <= 0
        ):
            raise ValueError("SE(3) consistency gates must be positive")
        nodes = sorted(
            {self.anchor}
            | {constraint.source for constraint in self._constraints}
            | {constraint.destination for constraint in self._constraints}
        )
        parent = {node: node for node in nodes}

        def find(node: str) -> str:
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        tree_indices: set[int] = set()
        for index, constraint in sorted(
            enumerate(self._constraints), key=lambda item: (-item[1].information, item[0])
        ):
            left, right = find(constraint.source), find(constraint.destination)
            if left != right:
                parent[right] = left
                tree_indices.add(index)

        adjacency: dict[str, list[tuple[str, npt.NDArray[np.float64], npt.NDArray[np.float64]]]] = {
            node: [] for node in nodes
        }
        for index in tree_indices:
            constraint = self._constraints[index]
            matrix = constraint.matrix
            adjacency[constraint.source].append(
                (constraint.destination, matrix, constraint.covariance)
            )
            adjacency[constraint.destination].append(
                (constraint.source, np.linalg.inv(matrix), constraint.covariance)
            )

        poses: dict[str, npt.NDArray[np.float64]] = {}
        covariances: dict[str, npt.NDArray[np.float64]] = {}
        components: dict[str, int] = {}
        roots = [self.anchor, *(node for node in nodes if node != self.anchor)]
        component = 0
        for root in roots:
            if root in poses:
                continue
            queue = deque([root])
            poses[root] = np.eye(4)
            covariances[root] = np.zeros((6, 6))
            components[root] = component
            while queue:
                source = queue.popleft()
                for destination, relative, covariance in adjacency[source]:
                    if destination in poses:
                        continue
                    poses[destination] = poses[source] @ relative
                    covariances[destination] = covariances[source] + covariance
                    components[destination] = component
                    queue.append(destination)
            component += 1

        translation_residuals, rotation_residuals = self._residuals(poses)
        rejected: list[int] = []
        for index, (translation_error, rotation_error) in enumerate(
            zip(translation_residuals, rotation_residuals, strict=True)
        ):
            if index not in tree_indices and (
                translation_error > translation_gate_m or rotation_error > rotation_gate_rad
            ):
                rejected.append(index)
        rotation_rationalization_constraint_count = 0
        rotation_rationalization_iterations = 0
        if rationalize_se3:
            (
                rotation_rationalization_constraint_count,
                rotation_rationalization_iterations,
            ) = self._rationalize_rotations(
                poses,
                components,
                frozenset(rejected),
                max_iterations=max_rotation_iterations,
                tolerance_rad=rotation_tolerance_rad,
            )
        translation_rationalized = rationalize_translations or rationalize_se3
        rationalization_constraint_count = 0
        if translation_rationalized:
            rationalization_constraint_count = self._rationalize_translations(
                poses,
                components,
                frozenset(rejected),
            )
            translation_residuals, rotation_residuals = self._residuals(poses)
        accepted_translation = [
            residual
            for index, residual in enumerate(translation_residuals)
            if index not in rejected
        ]
        accepted_rotation = [
            residual for index, residual in enumerate(rotation_residuals) if index not in rejected
        ]
        result = SE3PoseGraphResult(
            poses,
            covariances,
            components,
            tuple(rejected),
            float(np.sqrt(np.mean(np.square(accepted_translation))))
            if accepted_translation
            else 0.0,
            float(np.sqrt(np.mean(np.square(accepted_rotation)))) if accepted_rotation else 0.0,
            self._revision,
            len(self._constraints),
            translation_rationalized,
            rationalization_constraint_count,
            rationalize_se3,
            rotation_rationalization_constraint_count,
            rotation_rationalization_iterations,
        )
        self._history.append(result)
        while len(self._history) > self.max_results:
            self._history.popleft()
        self.metrics["optimizations"] += 1
        return result

    def _residuals(
        self,
        poses: dict[str, npt.NDArray[np.float64]],
    ) -> tuple[list[float], list[float]]:
        translation_residuals: list[float] = []
        rotation_residuals: list[float] = []
        for constraint in self._constraints:
            predicted = np.linalg.inv(poses[constraint.source]) @ poses[constraint.destination]
            error = np.linalg.inv(constraint.matrix) @ predicted
            translation_residuals.append(float(np.linalg.norm(error[:3, 3])))
            rotation_residuals.append(_rotation_angle(error[:3, :3]))
        return translation_residuals, rotation_residuals

    def _rationalize_rotations(
        self,
        poses: dict[str, npt.NDArray[np.float64]],
        components: dict[str, int],
        rejected: frozenset[int],
        *,
        max_iterations: int,
        tolerance_rad: float,
    ) -> tuple[int, int]:
        """Coordinate-descent chordal averaging over all accepted rotation factors."""
        neighbors: dict[
            str,
            list[tuple[str, npt.NDArray[np.float64], float]],
        ] = {node: [] for node in poses}
        used_constraints: set[int] = set()
        for index, constraint in enumerate(self._constraints):
            if index in rejected:
                continue
            relative_rotation = constraint.matrix[:3, :3]
            neighbors[constraint.destination].append(
                (
                    constraint.source,
                    relative_rotation,
                    constraint.rotation_information,
                )
            )
            neighbors[constraint.source].append(
                (
                    constraint.destination,
                    relative_rotation.T,
                    constraint.rotation_information,
                )
            )
            used_constraints.add(index)

        iterations = 0
        for component in sorted(set(components.values())):
            nodes = sorted(node for node, value in components.items() if value == component)
            root = self.anchor if self.anchor in nodes else nodes[0]
            variables = [node for node in nodes if node != root]
            if not variables:
                continue
            for iteration in range(1, max_iterations + 1):
                maximum_update_rad = 0.0
                for node in variables:
                    weighted_candidates = np.zeros((3, 3), dtype=np.float64)
                    for neighbor, relative, information in neighbors[node]:
                        weighted_candidates += information * poses[neighbor][:3, :3] @ relative
                    if not np.any(weighted_candidates):
                        continue
                    previous = poses[node][:3, :3]
                    updated = _project_rotation(weighted_candidates)
                    maximum_update_rad = max(
                        maximum_update_rad,
                        _rotation_angle(previous.T @ updated),
                    )
                    poses[node][:3, :3] = updated
                iterations = max(iterations, iteration)
                if maximum_update_rad <= tolerance_rad:
                    break
        return len(used_constraints), iterations

    def _rationalize_translations(
        self,
        poses: dict[str, npt.NDArray[np.float64]],
        components: dict[str, int],
        rejected: frozenset[int],
    ) -> int:
        """Solve all accepted translation factors while holding rotations fixed."""
        used_constraints = 0
        for component in sorted(set(components.values())):
            nodes = sorted(node for node, value in components.items() if value == component)
            root = self.anchor if self.anchor in nodes else nodes[0]
            variables = [node for node in nodes if node != root]
            columns = {node: index for index, node in enumerate(variables)}
            rows: list[npt.NDArray[np.float64]] = []
            targets: list[npt.NDArray[np.float64]] = []
            for index, constraint in enumerate(self._constraints):
                if (
                    index in rejected
                    or components[constraint.source] != component
                    or components[constraint.destination] != component
                ):
                    continue
                weight = np.sqrt(constraint.translation_information)
                row = np.zeros(len(variables), dtype=np.float64)
                if constraint.source != root:
                    row[columns[constraint.source]] = -weight
                if constraint.destination != root:
                    row[columns[constraint.destination]] = weight
                target = (poses[constraint.source][:3, :3] @ constraint.translation_m) * weight
                if constraint.source == root:
                    target += poses[root][:3, 3] * weight
                if constraint.destination == root:
                    target -= poses[root][:3, 3] * weight
                rows.append(row)
                targets.append(target)
                used_constraints += 1
            if not variables or not rows:
                continue
            solution, _, _, _ = np.linalg.lstsq(np.vstack(rows), np.vstack(targets), rcond=None)
            for node, column in columns.items():
                poses[node][:3, 3] = solution[column]
        return used_constraints

    def write_json(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": SNAPSHOT_SCHEMA,
            "anchor": self.anchor,
            "max_constraints": self.max_constraints,
            "max_results": self.max_results,
            "revision": self._revision,
            "constraints": [item.to_dict() for item in self._constraints],
        }
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
                json.dump(payload, stream, indent=2, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return destination

    @classmethod
    def read_json(cls, path: str | Path) -> RobustSE3PoseGraph:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("schema") != SNAPSHOT_SCHEMA:
            raise ValueError("unsupported SE(3) pose graph snapshot schema")
        graph = cls(
            str(payload["anchor"]),
            max_constraints=int(payload["max_constraints"]),
            max_results=int(payload["max_results"]),
        )
        constraints = [
            SE3PoseConstraint.from_dict(item)
            for item in cast(list[dict[str, Any]], payload["constraints"])
        ]
        if len(constraints) > graph.max_constraints:
            raise ValueError("pose graph snapshot exceeds configured bounds")
        for constraint in constraints:
            graph.add_constraint(constraint)
        if graph.revision != int(payload["revision"]):
            raise ValueError("pose graph snapshot revision is inconsistent")
        graph.metrics["restores"] = 1
        return graph

    @staticmethod
    def _constraint_key(constraint: SE3PoseConstraint) -> tuple[object, ...]:
        return (
            constraint.source,
            constraint.destination,
            tuple(constraint.translation_m),
            tuple(constraint.quaternion_xyzw),
            tuple(constraint.covariance.ravel()),
            constraint.information,
            constraint.kind,
        )
