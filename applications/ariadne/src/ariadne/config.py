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


class WingmanConfig(StrictModel):
    local_frame: FrameId
    body_frame: FrameId = FrameId("body")

    @field_validator("local_frame", "body_frame", mode="before")
    @classmethod
    def parse_frame(cls, value: object) -> FrameId:
        return value if isinstance(value, FrameId) else FrameId(str(value))


class IntelligenceConfig(StrictModel):
    global_frame: FrameId = FrameId("global")

    @field_validator("global_frame", mode="before")
    @classmethod
    def parse_frame(cls, value: object) -> FrameId:
        return value if isinstance(value, FrameId) else FrameId(str(value))


class SimulationConfig(StrictModel):
    seed: int = 0
    wingman_count: int = Field(default=2, ge=1)
    duration_seconds: float = Field(default=1.0, gt=0)


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
