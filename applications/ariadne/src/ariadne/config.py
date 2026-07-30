"""Typed runtime configuration loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, cast

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ariadne.common import FrameId


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RuntimeConfig(StrictModel):
    role: Literal["wingman", "intelligence", "simulation"]
    node_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_-]+$")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    json_logs: bool = False
    output_dir: Path


class PreprocessingConfig(StrictModel):
    width: int = Field(default=64, ge=2)
    height: int = Field(default=64, ge=2)
    min_blur_score: float = Field(default=0.002, ge=0)
    min_exposure_score: float = Field(default=0.15, ge=0, le=1)
    max_occlusion_fraction: float = Field(default=0.55, ge=0, le=1)


class SaliencyConfig(StrictModel):
    backend: Literal["u2net", "gradient_contrast"] = "u2net"
    model_path: Path = Path(".cache/ariadne/models/u2net/u2net.pth")
    device: str = Field(default="auto", min_length=1)
    input_size: int = Field(default=320, ge=32, multiple_of=32)
    threshold: float = Field(default=0.5, gt=0, lt=1)
    quantile: float = Field(default=0.82, gt=0, lt=1)
    contrast_weight: float = Field(default=0.45, ge=0, le=1)
    min_area_px: int = Field(default=4, ge=1)
    max_regions: int = Field(default=32, ge=1)


class ObjectStoreConfig(StrictModel):
    max_objects: int = Field(default=256, ge=1)
    max_keyframes_per_object: int = Field(default=4, ge=1)
    snapshot_path: Path | None = None


class TransportConfig(StrictModel):
    queue_capacity: int = Field(default=128, ge=1)
    max_payload_bytes: int = Field(default=64_000, ge=129)
    packet_ttl_seconds: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=0, ge=0)
    retry_interval_seconds: float = Field(default=0.1, gt=0)
    dedupe_entries: int = Field(default=1024, ge=1)

    @model_validator(mode="after")
    def dedupe_window_covers_queue(self) -> TransportConfig:
        if self.dedupe_entries < self.queue_capacity:
            raise ValueError("dedupe_entries must be at least queue_capacity")
        return self


class SplattingConfig(StrictModel):
    backend: Literal["reference", "resplat", "mvsplat", "anysplat"] = "reference"
    device: str = Field(default="cpu", min_length=1)
    max_observations_per_update: int = Field(default=64, ge=1)
    max_objects_per_update: int = Field(default=32, ge=1)
    max_estimated_memory_bytes: int = Field(default=268_435_456, ge=1)
    max_registered_backends: int = Field(default=16, ge=1)
    max_primitives: int = Field(default=10_000, ge=1)
    max_history: int = Field(default=16, ge=1)
    snapshot_directory: Path | None = None
    max_persisted_snapshots: int = Field(default=16, ge=1)


class PoseGraphConfig(StrictModel):
    translation_gate_m: float = Field(default=0.75, gt=0)
    rotation_gate_rad: float = Field(default=0.35, gt=0)
    max_constraints: int = Field(default=16_384, ge=1)
    max_results: int = Field(default=32, ge=1)
    snapshot_path: Path | None = None


class AssociationConfig(StrictModel):
    max_distance_m: float = Field(default=1.5, gt=0)
    min_cosine_similarity: float = Field(default=0.8, ge=0, le=1)
    max_objects: int = Field(default=4096, ge=1)
    max_evidence: int = Field(default=8192, ge=1)
    snapshot_path: Path | None = None


class CorrectionConfig(StrictModel):
    max_translation_step_m: float = Field(default=0.5, gt=0)
    max_total_translation_m: float = Field(default=10.0, gt=0)
    max_generated_history: int = Field(default=1024, ge=1)
    max_applied_history: int = Field(default=4096, ge=1)
    generator_snapshot_path: Path | None = None
    applier_snapshot_path: Path | None = None
    ttl_seconds: float = Field(default=60.0, gt=0)

    @model_validator(mode="after")
    def total_correction_covers_step(self) -> CorrectionConfig:
        if self.max_total_translation_m < self.max_translation_step_m:
            raise ValueError(
                "max_total_translation_m must be at least max_translation_step_m"
            )
        return self


class ContextConfig(StrictModel):
    max_age_seconds: float = Field(default=5.0, gt=0)


class PlanningConfig(StrictModel):
    minimum_battery_fraction: float = Field(default=0.2, ge=0, lt=1)
    minimum_link_quality: float = Field(default=0.15, ge=0, le=1)
    handoff_ttl_seconds: float = Field(default=60.0, gt=0)
    fail_closed_on_degraded_context: bool = True


class TelemetryConfig(StrictModel):
    max_distribution_samples: int = Field(default=1024, ge=1)
    max_metrics: int = Field(default=256, ge=1)
    max_events: int = Field(default=1024, ge=1)


class SecurityConfig(StrictModel):
    max_envelope_ttl_seconds: float = Field(default=60.0, gt=0)
    replay_window_entries: int = Field(default=4096, ge=1)


class DeploymentConfig(StrictModel):
    profile: str = Field(default="cpu_reference", min_length=1)


class WingmanConfig(StrictModel):
    local_frame: FrameId
    body_frame: FrameId = FrameId("body")
    preprocessing: PreprocessingConfig = PreprocessingConfig()
    saliency: SaliencyConfig = SaliencyConfig()
    object_store: ObjectStoreConfig = ObjectStoreConfig()
    transport: TransportConfig = TransportConfig()
    correction: CorrectionConfig = CorrectionConfig()
    telemetry: TelemetryConfig = TelemetryConfig()
    security: SecurityConfig = SecurityConfig()
    deployment: DeploymentConfig = DeploymentConfig()

    @field_validator("local_frame", "body_frame", mode="before")
    @classmethod
    def parse_frame(cls, value: object) -> FrameId:
        return value if isinstance(value, FrameId) else FrameId(str(value))


class IntelligenceConfig(StrictModel):
    global_frame: FrameId = FrameId("global")
    observation_retention_seconds: float = Field(default=60.0, gt=0)
    observation_max_entries: int = Field(default=8192, ge=1)
    observation_sequence_history: int = Field(default=4096, ge=1)
    observation_max_future_skew_seconds: float = Field(default=1.0, ge=0)
    observation_snapshot_path: Path | None = None
    observation_journal_directory: Path | None = None
    observation_journal_entries_per_segment: int = Field(default=1024, ge=1)
    observation_journal_max_segments: int = Field(default=8, ge=1)
    transport: TransportConfig = TransportConfig()
    splatting: SplattingConfig = SplattingConfig()
    association: AssociationConfig = AssociationConfig()
    pose_graph: PoseGraphConfig = PoseGraphConfig()
    correction: CorrectionConfig = CorrectionConfig()
    context: ContextConfig = ContextConfig()
    planning: PlanningConfig = PlanningConfig()
    telemetry: TelemetryConfig = TelemetryConfig()
    security: SecurityConfig = SecurityConfig()
    deployment: DeploymentConfig = DeploymentConfig()

    @field_validator("global_frame", mode="before")
    @classmethod
    def parse_frame(cls, value: object) -> FrameId:
        return value if isinstance(value, FrameId) else FrameId(str(value))


class SimulationConfig(StrictModel):
    seed: int = 0
    wingman_count: int = Field(default=2, ge=1)
    duration_seconds: float = Field(default=1.0, gt=0)
    packet_loss_probability: float = Field(default=0.12, ge=0, le=1)
    partition_duration_seconds: float = Field(default=60.0, ge=60.0)


class AriadneConfig(StrictModel):
    runtime: RuntimeConfig
    wingman: WingmanConfig | None = None
    intelligence: IntelligenceConfig | None = None
    simulation: SimulationConfig | None = None

    @model_validator(mode="after")
    def role_section_is_present(self) -> AriadneConfig:
        section = getattr(self, self.runtime.role)
        if section is None:
            raise ValueError(f"configuration for role {self.runtime.role!r} is required")
        return self


def load_config(path: str | Path) -> AriadneConfig:
    config_path = Path(path).expanduser()
    if not config_path.is_file():
        raise FileNotFoundError(f"configuration file does not exist: {config_path}")
    raw = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if not isinstance(raw, dict):
        raise ValueError(f"configuration root must be a mapping: {config_path}")
    return AriadneConfig.model_validate(cast(dict[str, Any], raw))
