"""Real dataset replay sources normalized for ARIADNE model adapters."""

from __future__ import annotations

import csv
import importlib
import io
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import numpy.typing as npt

from ariadne.common import Timestamp
from ariadne.replay.synchronizer import ImageFrame, ImuSample


def _vector3(message: Any) -> npt.NDArray[np.float64]:
    return np.asarray([message.x, message.y, message.z], dtype=np.float64)


def _decode_image(payload: bytes) -> npt.NDArray[np.generic]:
    try:
        image_module = importlib.import_module("PIL.Image")
    except ImportError as error:
        raise RuntimeError("install ARIADNE with the evaluation extra to decode images") from error
    with image_module.open(io.BytesIO(payload)) as image:
        decoded = np.asarray(image.convert("RGB"))
    return cast(npt.NDArray[np.generic], decoded)


@dataclass(frozen=True)
class GroundTruthPose:
    timestamp: Timestamp
    position_m: npt.NDArray[np.float64]
    quaternion_xyzw: npt.NDArray[np.float64]

    def __post_init__(self) -> None:
        position = np.asarray(self.position_m, dtype=np.float64)
        quaternion = np.asarray(self.quaternion_xyzw, dtype=np.float64)
        if position.shape != (3,) or quaternion.shape != (4,):
            raise ValueError("ground truth position and quaternion shapes are invalid")
        if not np.all(np.isfinite(position)) or not np.all(np.isfinite(quaternion)):
            raise ValueError("ground truth pose must be finite")
        object.__setattr__(self, "position_m", position.copy())
        object.__setattr__(self, "quaternion_xyzw", quaternion.copy())


@dataclass(frozen=True)
class ReplayBatch:
    dataset: str
    agent_id: str
    camera_streams: dict[str, tuple[ImageFrame, ...]]
    imu_samples: tuple[ImuSample, ...]
    ground_truth: tuple[GroundTruthPose, ...]
    calibration: dict[str, Any] = field(default_factory=dict)
    source_path: Path | None = None
    warnings: tuple[str, ...] = ()

    @property
    def primary_images(self) -> tuple[ImageFrame, ...]:
        if not self.camera_streams:
            return ()
        return next(iter(self.camera_streams.values()))


class ReplaySource(Protocol):
    def load(
        self, agent_id: str, *, start_frame: int = 0, max_frames: int = 100
    ) -> ReplayBatch: ...


def _validate_window(start_frame: int, max_frames: int) -> None:
    if start_frame < 0 or max_frames <= 0:
        raise ValueError("start_frame must be non-negative and max_frames must be positive")


def _seconds_to_timestamp(value: str | float) -> Timestamp:
    return Timestamp(int(float(value) * 1e9))


class MiluvReplaySource:
    def __init__(self, archive: Path) -> None:
        self.archive = archive

    def load(self, agent_id: str, *, start_frame: int = 0, max_frames: int = 100) -> ReplayBatch:
        _validate_window(start_frame, max_frames)
        with zipfile.ZipFile(self.archive) as handle:
            names = handle.namelist()
            prefix = next(
                (
                    name.rsplit("/", 1)[0]
                    for name in names
                    if name.endswith(f"/{agent_id}/imu_cam.csv")
                ),
                None,
            )
            if prefix is None:
                raise ValueError(f"MILUV agent not found: {agent_id}")
            agent_root = prefix.removesuffix(f"/{agent_id}") + f"/{agent_id}"
            image_names = sorted(
                (
                    name
                    for name in names
                    if name.startswith(f"{agent_root}/color/") and Path(name).suffix
                ),
                key=lambda name: float(Path(name).stem),
            )[start_frame : start_frame + max_frames]
            images = tuple(
                ImageFrame(
                    _seconds_to_timestamp(Path(name).stem),
                    agent_id,
                    _decode_image(handle.read(name)),
                    start_frame + index,
                )
                for index, name in enumerate(image_names)
            )
            imu = self._read_imu(handle, f"{agent_root}/imu_cam.csv", agent_id, images)
            truth = self._read_truth(handle, f"{agent_root}/mocap.csv", images)
        return ReplayBatch(
            "miluv",
            agent_id,
            {"color": images},
            imu,
            truth,
            source_path=self.archive,
            warnings=("MILUV selected archive does not publish camera intrinsics.",),
        )

    @staticmethod
    def _read_imu(
        handle: zipfile.ZipFile, member: str, agent_id: str, images: tuple[ImageFrame, ...]
    ) -> tuple[ImuSample, ...]:
        if not images:
            return ()
        start_ns = images[0].timestamp.monotonic_ns - 100_000_000
        end_ns = images[-1].timestamp.monotonic_ns
        with handle.open(member) as raw, io.TextIOWrapper(raw, encoding="utf-8") as text:
            rows = csv.DictReader(text)
            return tuple(
                ImuSample(
                    timestamp,
                    agent_id,
                    np.asarray(
                        [
                            row["linear_acceleration.x"],
                            row["linear_acceleration.y"],
                            row["linear_acceleration.z"],
                        ],
                        dtype=np.float64,
                    ),
                    np.asarray(
                        [
                            row["angular_velocity.x"],
                            row["angular_velocity.y"],
                            row["angular_velocity.z"],
                        ],
                        dtype=np.float64,
                    ),
                )
                for row in rows
                if start_ns
                <= (timestamp := _seconds_to_timestamp(row["timestamp"])).monotonic_ns
                <= end_ns
            )

    @staticmethod
    def _read_truth(
        handle: zipfile.ZipFile, member: str, images: tuple[ImageFrame, ...]
    ) -> tuple[GroundTruthPose, ...]:
        if not images:
            return ()
        start_ns = images[0].timestamp.monotonic_ns - 100_000_000
        end_ns = images[-1].timestamp.monotonic_ns + 100_000_000
        with handle.open(member) as raw, io.TextIOWrapper(raw, encoding="utf-8") as text:
            rows = csv.DictReader(text)
            return tuple(
                GroundTruthPose(
                    timestamp,
                    np.asarray(
                        [
                            row["pose.position.x"],
                            row["pose.position.y"],
                            row["pose.position.z"],
                        ],
                        dtype=np.float64,
                    ),
                    np.asarray(
                        [
                            row["pose.orientation.x"],
                            row["pose.orientation.y"],
                            row["pose.orientation.z"],
                            row["pose.orientation.w"],
                        ],
                        dtype=np.float64,
                    ),
                )
                for row in rows
                if start_ns
                <= (timestamp := _seconds_to_timestamp(row["timestamp"])).monotonic_ns
                <= end_ns
            )


def _opencv_yaml_scalars(path: Path) -> dict[str, float | int | str]:
    values: dict[str, float | int | str] = {}
    pattern = re.compile(r"^([A-Za-z][A-Za-z0-9_.]*):\s*[\"']?([^#\"']+)")
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line.strip())
        if match is None:
            continue
        key, raw_value = match.groups()
        value = raw_value.strip()
        try:
            numeric = float(value)
            values[key] = int(numeric) if numeric.is_integer() else numeric
        except ValueError:
            values[key] = value
    return values


class RosbagReplaySource:
    def __init__(
        self,
        bag: Path,
        *,
        dataset: str,
        calibration_path: Path | None = None,
        ground_truth_path: Path | None = None,
    ) -> None:
        self.bag = bag
        self.dataset = dataset
        self.calibration_path = calibration_path
        self.ground_truth_path = ground_truth_path

    def load(self, agent_id: str, *, start_frame: int = 0, max_frames: int = 100) -> ReplayBatch:
        _validate_window(start_frame, max_frames)
        AnyReader = importlib.import_module("rosbags.highlevel").AnyReader
        kwargs: dict[str, Any] = {}
        if self.bag.suffix == ".db3":
            typesys = importlib.import_module("rosbags.typesys")
            kwargs["default_typestore"] = typesys.get_typestore(typesys.Stores.ROS2_HUMBLE)
        with AnyReader([self.bag], **kwargs) as reader:
            topics = self._topics(agent_id)
            connections = [
                connection
                for connection in reader.connections
                if connection.topic in set(topics.values())
            ]
            messages: list[tuple[str, int, Any]] = []
            left_seen = 0
            end_ns: int | None = None
            for connection, timestamp_ns, raw in reader.messages(connections=connections):
                if end_ns is not None and timestamp_ns > end_ns:
                    break
                if connection.topic == topics["left"]:
                    if left_seen < start_frame:
                        left_seen += 1
                        continue
                    if left_seen >= start_frame + max_frames:
                        end_ns = timestamp_ns
                        continue
                    left_seen += 1
                messages.append(
                    (connection.topic, timestamp_ns, reader.deserialize(raw, connection.msgtype))
                )
        streams: dict[str, list[ImageFrame]] = {"left": [], "right": []}
        imu: list[ImuSample] = []
        embedded_truth: list[GroundTruthPose] = []
        for topic, timestamp_ns, message in messages:
            if topic == topics["imu"]:
                imu.append(
                    ImuSample(
                        Timestamp(timestamp_ns),
                        agent_id,
                        _vector3(message.linear_acceleration),
                        _vector3(message.angular_velocity),
                    )
                )
            elif topic in (topics["left"], topics["right"]):
                camera = "left" if topic == topics["left"] else "right"
                payload = cast(npt.NDArray[np.uint8], message.data).tobytes()
                streams[camera].append(
                    ImageFrame(
                        Timestamp(timestamp_ns),
                        agent_id,
                        _decode_image(payload),
                        len(streams[camera]),
                    )
                )
            elif topic == topics.get("truth"):
                pose = message.pose if hasattr(message, "pose") else message.transform
                translation = pose.position if hasattr(pose, "position") else pose.translation
                rotation = pose.orientation if hasattr(pose, "orientation") else pose.rotation
                embedded_truth.append(
                    GroundTruthPose(
                        Timestamp(timestamp_ns),
                        _vector3(translation),
                        np.asarray([rotation.x, rotation.y, rotation.z, rotation.w]),
                    )
                )
        if not streams["left"]:
            raise ValueError(f"no frames found for {self.dataset}:{agent_id}")
        first_image_ns = streams["left"][0].timestamp.monotonic_ns
        last_image_ns = streams["left"][-1].timestamp.monotonic_ns
        selected_streams = {
            name: tuple(
                frame
                for frame in frames
                if first_image_ns <= frame.timestamp.monotonic_ns <= last_image_ns
            )
            for name, frames in streams.items()
            if frames
        }
        imu = [
            sample
            for sample in imu
            if first_image_ns - 100_000_000 <= sample.timestamp.monotonic_ns <= last_image_ns
        ]
        all_truth = self._external_truth() or tuple(embedded_truth)
        truth = tuple(
            pose
            for pose in all_truth
            if first_image_ns - 1_000_000_000
            <= pose.timestamp.monotonic_ns
            <= last_image_ns + 1_000_000_000
        )
        calibration = (
            _opencv_yaml_scalars(self.calibration_path)
            if self.calibration_path is not None and self.calibration_path.is_file()
            else {}
        )
        warnings = () if calibration else ("camera calibration was not available",)
        return ReplayBatch(
            self.dataset,
            agent_id,
            selected_streams,
            tuple(imu),
            truth,
            calibration,
            self.bag,
            warnings,
        )

    def _topics(self, agent_id: str) -> dict[str, str]:
        if self.dataset == "s3e":
            topic_agent = agent_id.capitalize()
            return {
                "left": f"/{topic_agent}/left_camera/compressed",
                "right": f"/{topic_agent}/right_camera/compressed",
                "imu": f"/{topic_agent}/imu/data",
            }
        return {
            "left": "/cam0/image_raw/compressed",
            "right": "/cam1/image_raw/compressed",
            "imu": "/imu0",
            "truth": "/calib_pose",
        }

    def _external_truth(self) -> tuple[GroundTruthPose, ...]:
        if self.ground_truth_path is None or not self.ground_truth_path.is_file():
            return ()
        poses: list[GroundTruthPose] = []
        for line in self.ground_truth_path.read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            values = [float(value) for value in line.replace(",", " ").split()]
            if len(values) < 8:
                continue
            poses.append(
                GroundTruthPose(
                    _seconds_to_timestamp(values[0]),
                    np.asarray(values[1:4]),
                    np.asarray(values[4:8]),
                )
            )
        return tuple(poses)


class S3EReplaySource(RosbagReplaySource):
    def __init__(self, bag: Path, calibration_root: Path, ground_truth_root: Path) -> None:
        super().__init__(bag, dataset="s3e")
        self.calibration_root = calibration_root
        self.ground_truth_root = ground_truth_root

    def load(self, agent_id: str, *, start_frame: int = 0, max_frames: int = 100) -> ReplayBatch:
        self.calibration_path = self.calibration_root / f"{agent_id.lower()}.yaml"
        self.ground_truth_path = self.ground_truth_root / f"{agent_id.lower()}_gt.txt"
        return super().load(agent_id, start_frame=start_frame, max_frames=max_frames)


class D2SlamReplaySource(RosbagReplaySource):
    def __init__(self, root: Path, sequence: int) -> None:
        if sequence not in range(1, 6):
            raise ValueError("D2SLAM sequence must be between one and five")
        bag = root / f"dataset-corridor{sequence}_512_16-sync-comp.bag"
        super().__init__(
            bag,
            dataset="d2slam",
            calibration_path=root / "Configs/tum/left_pinhole.yaml",
            ground_truth_path=root / f"groundtruth_{sequence}.csv",
        )
        self.sequence = sequence

    def load(
        self, agent_id: str = "d2slam", *, start_frame: int = 0, max_frames: int = 100
    ) -> ReplayBatch:
        return super().load(agent_id, start_frame=start_frame, max_frames=max_frames)
