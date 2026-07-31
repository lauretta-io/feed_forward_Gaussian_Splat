"""Static, asynchronous fusion of dense Gaussian PLY contributions.

The dense fusion contract deliberately separates spatial registration from time.
Every contribution must provide an explicit rigid local-to-global transform, but
capture timestamps are provenance only: inputs are neither sorted nor rejected
when their time ranges do not overlap.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, BinaryIO

import numpy as np
import numpy.typing as npt

MANIFEST_SCHEMA = "ariadne.static-asynchronous-global-gaussians.v1"

_PLY_TYPES: dict[str, str] = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}


def _as_transform(value: npt.ArrayLike) -> npt.NDArray[np.float64]:
    transform = np.asarray(value, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("local-to-global transform must be a finite 4x4 matrix")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise ValueError("local-to-global transform must be homogeneous")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not math.isclose(
        float(np.linalg.det(rotation)), 1.0, abs_tol=1e-5
    ):
        raise ValueError("local-to-global transform must contain a proper rigid rotation")
    return transform


@dataclass(frozen=True)
class DenseGaussianContribution:
    agent_id: str
    capture_timestamp_ns: int | None
    ply_path: Path
    local_to_global: npt.NDArray[np.float64]
    registration_method: str
    registration_verified: bool = False

    def __post_init__(self) -> None:
        if not self.agent_id.strip():
            raise ValueError("Gaussian contribution agent_id cannot be empty")
        if self.capture_timestamp_ns is not None and self.capture_timestamp_ns < 0:
            raise ValueError("Gaussian contribution timestamp cannot be negative")
        if not self.registration_method.strip():
            raise ValueError("Gaussian contribution registration method cannot be empty")
        object.__setattr__(self, "ply_path", Path(self.ply_path))
        object.__setattr__(self, "local_to_global", _as_transform(self.local_to_global))


@dataclass(frozen=True)
class DenseFusionResult:
    output_ply: Path
    manifest_path: Path
    input_gaussians: int
    output_gaussians: int
    filtered_gaussians: int
    global_registration_verified: bool


@dataclass(frozen=True)
class _PlyVertexData:
    vertices: npt.NDArray[Any]
    format_name: str
    property_types: tuple[tuple[str, str], ...]


def _read_header(stream: BinaryIO) -> tuple[list[str], int]:
    lines: list[str] = []
    size = 0
    while True:
        raw = stream.readline()
        if not raw:
            raise ValueError("PLY header is truncated")
        size += len(raw)
        try:
            line = raw.decode("ascii").rstrip("\r\n")
        except UnicodeDecodeError as error:
            raise ValueError("PLY header must be ASCII") from error
        lines.append(line)
        if line == "end_header":
            return lines, size


def _parse_vertex_header(lines: Sequence[str]) -> tuple[str, int, tuple[tuple[str, str], ...]]:
    if not lines or lines[0] != "ply":
        raise ValueError("input is not a PLY file")
    format_name = ""
    vertex_count: int | None = None
    properties: list[tuple[str, str]] = []
    current_element = ""
    elements_before_vertex = 0
    for line in lines[1:]:
        fields = line.split()
        if not fields or fields[0] in {"comment", "obj_info", "end_header"}:
            continue
        if fields[0] == "format" and len(fields) >= 2:
            format_name = fields[1]
        elif fields[0] == "element" and len(fields) == 3:
            current_element = fields[1]
            if current_element == "vertex":
                vertex_count = int(fields[2])
            elif vertex_count is None and int(fields[2]) != 0:
                elements_before_vertex += int(fields[2])
        elif fields[0] == "property" and current_element == "vertex":
            if len(fields) != 3 or fields[1] == "list":
                raise ValueError("dense Gaussian PLY requires scalar vertex properties")
            if fields[1] not in _PLY_TYPES:
                raise ValueError(f"unsupported PLY property type: {fields[1]}")
            properties.append((fields[2], fields[1]))
    if format_name not in {"ascii", "binary_little_endian"}:
        raise ValueError("only ASCII and binary little-endian PLY files are supported")
    if vertex_count is None or vertex_count <= 0 or not properties:
        raise ValueError("PLY must contain a non-empty vertex element")
    if elements_before_vertex:
        raise ValueError("PLY elements before vertex are not supported")
    required = {"x", "y", "z", "rot_0", "rot_1", "rot_2", "rot_3"}
    names = {name for name, _ in properties}
    missing = required - names
    if missing:
        raise ValueError(f"Gaussian PLY is missing properties: {', '.join(sorted(missing))}")
    return format_name, vertex_count, tuple(properties)


def _structured_dtype(
    properties: Sequence[tuple[str, str]], *, byte_order: str
) -> np.dtype[Any]:
    return np.dtype([(name, byte_order + _PLY_TYPES[kind]) for name, kind in properties])


def _read_ply_vertices(path: Path) -> _PlyVertexData:
    if not path.is_file():
        raise FileNotFoundError(f"Gaussian contribution does not exist: {path}")
    with path.open("rb") as stream:
        lines, _ = _read_header(stream)
        format_name, count, properties = _parse_vertex_header(lines)
        if format_name == "binary_little_endian":
            vertices = np.fromfile(
                stream,
                dtype=_structured_dtype(properties, byte_order="<"),
                count=count,
            )
        else:
            matrix = np.loadtxt(stream, dtype=np.float64, max_rows=count, ndmin=2)
            if matrix.shape != (count, len(properties)):
                raise ValueError("ASCII PLY vertex payload does not match its header")
            vertices = np.empty(count, dtype=_structured_dtype(properties, byte_order="="))
            for index, (name, _) in enumerate(properties):
                vertices[name] = matrix[:, index]
    if len(vertices) != count:
        raise ValueError("PLY vertex payload is truncated")
    return _PlyVertexData(vertices, format_name, properties)


def _rotation_quaternion_wxyz(rotation: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.asarray(
            [0.25 * scale, (rotation[2, 1] - rotation[1, 2]) / scale,
             (rotation[0, 2] - rotation[2, 0]) / scale,
             (rotation[1, 0] - rotation[0, 1]) / scale]
        )
    else:
        axis = int(np.argmax(np.diag(rotation)))
        if axis == 0:
            scale = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            quaternion = np.asarray(
                [(rotation[2, 1] - rotation[1, 2]) / scale, 0.25 * scale,
                 (rotation[0, 1] + rotation[1, 0]) / scale,
                 (rotation[0, 2] + rotation[2, 0]) / scale]
            )
        elif axis == 1:
            scale = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            quaternion = np.asarray(
                [(rotation[0, 2] - rotation[2, 0]) / scale,
                 (rotation[0, 1] + rotation[1, 0]) / scale, 0.25 * scale,
                 (rotation[1, 2] + rotation[2, 1]) / scale]
            )
        else:
            scale = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            quaternion = np.asarray(
                [(rotation[1, 0] - rotation[0, 1]) / scale,
                 (rotation[0, 2] + rotation[2, 0]) / scale,
                 (rotation[1, 2] + rotation[2, 1]) / scale, 0.25 * scale]
            )
    return quaternion / np.linalg.norm(quaternion)


def _quaternion_multiply_wxyz(
    left: npt.NDArray[np.float64], right: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right.T
    return np.column_stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        )
    )


def _finite_vertex_mask(vertices: npt.NDArray[Any]) -> npt.NDArray[np.bool_]:
    mask = np.ones(len(vertices), dtype=np.bool_)
    for name in vertices.dtype.names or ():
        if np.issubdtype(vertices.dtype[name], np.floating):
            mask &= np.isfinite(vertices[name])
    quaternion = np.column_stack([vertices[f"rot_{index}"] for index in range(4)])
    mask &= np.linalg.norm(quaternion.astype(np.float64), axis=1) > 1e-12
    return mask


def _transform_vertices(
    vertices: npt.NDArray[Any], transform: npt.NDArray[np.float64]
) -> npt.NDArray[Any]:
    output = vertices.copy()
    positions = np.column_stack([vertices[name] for name in ("x", "y", "z")]).astype(
        np.float64
    )
    positions = (transform[:3, :3] @ positions.T).T + transform[:3, 3]
    for index, name in enumerate(("x", "y", "z")):
        output[name] = positions[:, index]
    if all(name in (vertices.dtype.names or ()) for name in ("nx", "ny", "nz")):
        normals = np.column_stack([vertices[name] for name in ("nx", "ny", "nz")]).astype(
            np.float64
        )
        normals = (transform[:3, :3] @ normals.T).T
        for index, name in enumerate(("nx", "ny", "nz")):
            output[name] = normals[:, index]
    local_quaternions = np.column_stack(
        [vertices[f"rot_{index}"] for index in range(4)]
    ).astype(np.float64)
    local_quaternions /= np.linalg.norm(local_quaternions, axis=1, keepdims=True)
    global_quaternion = _rotation_quaternion_wxyz(transform[:3, :3])
    rotated = _quaternion_multiply_wxyz(global_quaternion, local_quaternions)
    rotated /= np.linalg.norm(rotated, axis=1, keepdims=True)
    for index in range(4):
        output[f"rot_{index}"] = rotated[:, index]
    return output


def _write_binary_ply(
    path: Path,
    vertices: npt.NDArray[Any],
    properties: Sequence[tuple[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "ply",
        "format binary_little_endian 1.0",
        f"comment schema {MANIFEST_SCHEMA}",
        "comment temporal_alignment none",
        "comment scene_assumption static",
        f"element vertex {len(vertices)}",
        *(f"property {kind} {name}" for name, kind in properties),
        "end_header",
        "",
    ]
    target_dtype = _structured_dtype(properties, byte_order="<")
    payload = vertices.astype(target_dtype, copy=False)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            "wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write("\n".join(header).encode("ascii"))
            payload.tofile(stream)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def fuse_static_gaussian_plys(
    contributions: Sequence[DenseGaussianContribution],
    output_ply: str | Path,
    *,
    manifest_path: str | Path | None = None,
) -> DenseFusionResult:
    """Fuse spatially registered PLYs without cross-contribution time alignment."""
    if not contributions:
        raise ValueError("at least one dense Gaussian contribution is required")
    output = Path(output_ply)
    manifest = Path(manifest_path) if manifest_path is not None else output.with_suffix(".json")
    property_contract: tuple[tuple[str, str], ...] | None = None
    transformed: list[npt.NDArray[Any]] = []
    source_records: list[dict[str, object]] = []
    input_count = 0
    filtered_count = 0
    directional_appearance_rotation_required = False
    offset = 0
    for contribution in contributions:
        source = _read_ply_vertices(contribution.ply_path)
        if property_contract is None:
            property_contract = source.property_types
        elif source.property_types != property_contract:
            raise ValueError("all Gaussian contributions must have identical PLY properties")
        input_count += len(source.vertices)
        mask = _finite_vertex_mask(source.vertices)
        clean = source.vertices[mask]
        filtered = len(source.vertices) - len(clean)
        filtered_count += filtered
        if not len(clean):
            raise ValueError(
                f"Gaussian contribution has no finite primitives: {contribution.ply_path}"
            )
        fused = _transform_vertices(clean, contribution.local_to_global)
        rest_properties = tuple(
            name for name in (clean.dtype.names or ()) if name.startswith("f_rest_")
        )
        has_directional_harmonics = any(
            bool(np.any(np.abs(clean[name].astype(np.float64)) > 1e-12))
            for name in rest_properties
        )
        rotation_is_identity = np.allclose(
            contribution.local_to_global[:3, :3], np.eye(3), atol=1e-9
        )
        source_requires_harmonic_rotation = bool(
            has_directional_harmonics and not rotation_is_identity
        )
        directional_appearance_rotation_required |= source_requires_harmonic_rotation
        transformed.append(fused)
        source_records.append(
            {
                "agent_id": contribution.agent_id,
                "capture_timestamp_ns": contribution.capture_timestamp_ns,
                "source_ply": str(contribution.ply_path),
                "source_sha256": _sha256(contribution.ply_path),
                "registration_method": contribution.registration_method,
                "registration_verified": contribution.registration_verified,
                "directional_harmonics_present": has_directional_harmonics,
                "directional_harmonics_rotation_required": (
                    source_requires_harmonic_rotation
                ),
                "local_to_global": contribution.local_to_global.tolist(),
                "input_gaussians": len(source.vertices),
                "filtered_nonfinite_gaussians": filtered,
                "output_index_start": offset,
                "output_index_end_exclusive": offset + len(fused),
            }
        )
        offset += len(fused)
    assert property_contract is not None
    merged = np.concatenate(transformed)
    _write_binary_ply(output, merged, property_contract)
    positions = np.column_stack([merged[name] for name in ("x", "y", "z")])
    timestamps = [item.capture_timestamp_ns for item in contributions]
    known_timestamps = [value for value in timestamps if value is not None]
    verified = all(item.registration_verified for item in contributions)
    warnings = []
    if not verified:
        warnings.append(
            "At least one local-to-global transform is unverified; this artifact is a fusion/atlas "
            "attempt and not evidence of metric global co-registration."
        )
    if directional_appearance_rotation_required:
        warnings.append(
            "At least one rotated source contains directional spherical harmonics. Gaussian "
            "geometry and orientations were rotated, but SH coefficients remain in their source "
            "basis; view-dependent appearance is diagnostic until SH rotation is implemented."
        )
    payload: dict[str, object] = {
        "schema": MANIFEST_SCHEMA,
        "mode": "static-asynchronous",
        "scene_assumption": "static",
        "dynamic_elements_modeled": False,
        "temporal_alignment": "none",
        "temporal_overlap_required": False,
        "timestamps_preserved_as_provenance": True,
        "input_order_preserved": True,
        "inputs_arrived_in_timestamp_order": (
            timestamps == sorted(known_timestamps)
            if len(known_timestamps) == len(timestamps)
            else None
        ),
        "global_registration_verified": verified,
        "global_metric_claim_eligible": verified,
        "directional_harmonics_rotation_applied": False,
        "appearance_claim_eligible": not directional_appearance_rotation_required,
        "input_gaussians": input_count,
        "output_gaussians": len(merged),
        "filtered_gaussians": filtered_count,
        "bounds_m": {
            "minimum": np.min(positions, axis=0).astype(float).tolist(),
            "maximum": np.max(positions, axis=0).astype(float).tolist(),
        },
        "output_ply": str(output),
        "output_sha256": _sha256(output),
        "sources": source_records,
        "warnings": warnings,
    }
    _atomic_json(manifest, payload)
    return DenseFusionResult(output, manifest, input_count, len(merged), filtered_count, verified)


def contributions_from_manifest(path: str | Path) -> tuple[DenseGaussianContribution, ...]:
    """Load the small input-spec format used by the fusion CLI."""
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("contributions"), list):
        raise ValueError("fusion input spec must contain a contributions array")
    base = manifest_path.parent
    contributions: list[DenseGaussianContribution] = []
    for raw in payload["contributions"]:
        if not isinstance(raw, dict):
            raise ValueError("fusion contribution must be an object")
        source = Path(str(raw["ply_path"]))
        if not source.is_absolute():
            source = base / source
        contributions.append(
            DenseGaussianContribution(
                agent_id=str(raw["agent_id"]),
                capture_timestamp_ns=(
                    int(raw["capture_timestamp_ns"])
                    if raw.get("capture_timestamp_ns") is not None
                    else None
                ),
                ply_path=source,
                local_to_global=np.asarray(raw["local_to_global"], dtype=np.float64),
                registration_method=str(raw["registration_method"]),
                registration_verified=bool(raw.get("registration_verified", False)),
            )
        )
    return tuple(contributions)
