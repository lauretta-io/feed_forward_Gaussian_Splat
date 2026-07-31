"""Robust incremental translation pose graph used as a CPU reference backend."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


def _translation(value: npt.ArrayLike, name: str) -> npt.NDArray[np.float64]:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite three-vector")
    return vector.copy()


@dataclass(frozen=True)
class PoseConstraint:
    source: str
    destination: str
    translation_m: npt.NDArray[np.float64]
    information: float = 1.0
    kind: str = "odometry"

    def __post_init__(self) -> None:
        if not self.source or not self.destination or self.source == self.destination:
            raise ValueError("constraint endpoints must be distinct and non-empty")
        if not np.isfinite(self.information) or self.information <= 0:
            raise ValueError("information must be finite and positive")
        object.__setattr__(self, "translation_m", _translation(self.translation_m, "translation_m"))


@dataclass(frozen=True)
class PoseGraphResult:
    positions_m: dict[str, npt.NDArray[np.float64]]
    rmse_m: float
    rejected_constraints: tuple[int, ...]
    iterations: int


class IncrementalPoseGraph:
    """IRLS pose graph reference; production adapters can replace it with GTSAM or Ceres."""

    def __init__(self, anchor: str, anchor_position_m: npt.ArrayLike = (0.0, 0.0, 0.0)) -> None:
        if not anchor:
            raise ValueError("anchor must not be empty")
        self.anchor = anchor
        self.anchor_position_m = _translation(anchor_position_m, "anchor_position_m")
        self._constraints: list[PoseConstraint] = []

    def add_constraint(self, constraint: PoseConstraint) -> None:
        self._constraints.append(constraint)

    def optimize(self, *, max_iterations: int = 10, huber_delta_m: float = 0.5) -> PoseGraphResult:
        if max_iterations <= 0 or huber_delta_m <= 0:
            raise ValueError("optimizer controls must be positive")
        nodes = sorted(
            {self.anchor}
            | {constraint.source for constraint in self._constraints}
            | {constraint.destination for constraint in self._constraints}
        )
        variable_nodes = [node for node in nodes if node != self.anchor]
        if not variable_nodes:
            return PoseGraphResult({self.anchor: self.anchor_position_m.copy()}, 0.0, (), 0)
        node_columns = {node: index for index, node in enumerate(variable_nodes)}
        weights = np.asarray([constraint.information for constraint in self._constraints])
        solution = np.zeros((len(variable_nodes), 3), dtype=np.float64)
        iteration_count = 0
        for iteration in range(1, max_iterations + 1):
            iteration_count = iteration
            rows: list[npt.NDArray[np.float64]] = []
            targets: list[npt.NDArray[np.float64]] = []
            for index, constraint in enumerate(self._constraints):
                row = np.zeros(len(variable_nodes), dtype=np.float64)
                target = constraint.translation_m.copy()
                if constraint.source == self.anchor:
                    target += self.anchor_position_m
                else:
                    row[node_columns[constraint.source]] -= 1.0
                if constraint.destination == self.anchor:
                    target = self.anchor_position_m - constraint.translation_m
                else:
                    row[node_columns[constraint.destination]] += 1.0
                scale = np.sqrt(weights[index])
                rows.append(row * scale)
                targets.append(target * scale)
            matrix = np.stack(rows) if rows else np.zeros((0, len(variable_nodes)))
            target_matrix = np.stack(targets) if targets else np.zeros((0, 3))
            previous = solution
            solution = np.linalg.lstsq(matrix, target_matrix, rcond=None)[0]
            positions = {
                self.anchor: self.anchor_position_m,
                **dict(zip(variable_nodes, solution, strict=True)),
            }
            residuals = np.asarray(
                [
                    np.linalg.norm(
                        positions[item.destination] - positions[item.source] - item.translation_m
                    )
                    for item in self._constraints
                ]
            )
            robust = np.where(
                residuals <= huber_delta_m,
                1.0,
                huber_delta_m / np.maximum(residuals, np.finfo(np.float64).eps),
            )
            new_weights = robust * np.asarray(
                [constraint.information for constraint in self._constraints]
            )
            if np.allclose(solution, previous, atol=1e-8) and np.allclose(
                weights, new_weights, atol=1e-8
            ):
                weights = new_weights
                break
            weights = new_weights
        positions = {
            self.anchor: self.anchor_position_m.copy(),
            **dict(zip(variable_nodes, solution, strict=True)),
        }
        residuals = np.asarray(
            [
                np.linalg.norm(
                    positions[item.destination] - positions[item.source] - item.translation_m
                )
                for item in self._constraints
            ]
        )
        rejected = tuple(
            index for index, residual in enumerate(residuals) if residual > 2 * huber_delta_m
        )
        rmse = float(np.sqrt(np.mean(residuals**2))) if residuals.size else 0.0
        return PoseGraphResult(positions, rmse, rejected, iteration_count)
