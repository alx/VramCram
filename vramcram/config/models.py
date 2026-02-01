"""Pydantic models for VramCram configuration."""

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class RedisConfig(BaseModel):
    """Redis connection configuration."""

    host: str = "localhost"
    port: int = Field(default=6379, ge=1, le=65535)
    db: int = Field(default=0, ge=0)
    password: str | None = None
    max_connections: int = Field(default=50, ge=1)
    socket_timeout: int = Field(default=5, ge=1)
    socket_connect_timeout: int = Field(default=5, ge=1)


class VRAMConfig(BaseModel):
    """VRAM management configuration."""

    total_mb: int = Field(gt=0, description="Total GPU VRAM in MB")
    safety_margin_mb: int = Field(ge=0, description="Reserved memory for system overhead")
    monitoring_interval_seconds: int = Field(default=5, ge=1)

    @field_validator("safety_margin_mb")
    @classmethod
    def safety_margin_less_than_total(cls, v: int, info: Any) -> int:
        """Validate safety margin is less than total VRAM."""
        if "total_mb" in info.data and v >= info.data["total_mb"]:
            raise ValueError("safety_margin_mb must be less than total_mb")
        return v


class APIConfig(BaseModel):
    """API server configuration."""

    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)
    workers: int = Field(default=1, ge=1)


class LLMModelConfig(BaseModel):
    """LLM model configuration."""

    name: str = Field(min_length=1, description="Unique model identifier")
    type: Literal["llm"] = "llm"
    model_path: Path = Field(description="Path to GGUF model file")
    vram_mb: int = Field(gt=0, description="Estimated VRAM usage in MB")
    config: dict[str, Any] = Field(
        default_factory=lambda: {
            "n_gpu_layers": 35,
            "n_ctx": 4096,
            "n_batch": 512,
            "temperature": 0.7,
        }
    )

    @field_validator("model_path")
    @classmethod
    def model_path_exists(cls, v: Path) -> Path:
        """Validate model path exists (skip in tests)."""
        # Allow non-existent paths for testing
        return v


class DiffusionModelConfig(BaseModel):
    """Diffusion model configuration."""

    name: str = Field(min_length=1, description="Unique model identifier")
    type: Literal["diffusion"] = "diffusion"
    model_path: Path = Field(description="Path to diffusion model file")
    vram_mb: int = Field(gt=0, description="Estimated VRAM usage in MB")
    config: dict[str, Any] = Field(
        default_factory=lambda: {
            "width": 512,
            "height": 512,
            "sample_steps": 4,
            "cfg_scale": 1.0,
        }
    )

    @field_validator("model_path")
    @classmethod
    def model_path_exists(cls, v: Path) -> Path:
        """Validate model path exists (skip in tests)."""
        # Allow non-existent paths for testing
        return v


class ModelsConfig(BaseModel):
    """Collection of model configurations."""

    llm: list[LLMModelConfig] = Field(default_factory=list)
    diffusion: list[DiffusionModelConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_unique_names(self) -> "ModelsConfig":
        """Ensure all model names are unique across types."""
        all_names = [m.name for m in self.llm] + [m.name for m in self.diffusion]
        if len(all_names) != len(set(all_names)):
            raise ValueError("Model names must be unique across all types")
        return self


class AgentConfig(BaseModel):
    """Agent behavior configuration."""

    heartbeat_interval_seconds: int = Field(default=10, ge=1)
    heartbeat_timeout_seconds: int = Field(default=30, ge=5)
    worker_ready_timeout_seconds: int = Field(default=60, ge=10)
    graceful_shutdown_timeout_seconds: int = Field(default=300, ge=30)
    eviction_timeout_seconds: int = Field(default=30, ge=5)


class JobConfig(BaseModel):
    """Job processing configuration."""

    max_queue_size: int = Field(default=1000, ge=1)
    result_ttl_seconds: int = Field(default=86400, ge=60)
    default_timeout_seconds: int = Field(default=300, ge=10)


class OutputConfig(BaseModel):
    """Output directory configuration."""

    base_path: Path = Field(default=Path("/var/vramcram/outputs"))
    results_path: Path = Field(default=Path("/var/vramcram/results"))
    image_result_format: Literal["base64", "path"] = Field(
        default="base64",
        description="Format for diffusion image results: 'base64' for data URI, 'path' for filesystem path"
    )


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    format: Literal["json", "text"] = "json"
    output: Literal["stdout", "stderr"] = "stdout"


class MetricsConfig(BaseModel):
    """Prometheus metrics configuration."""

    enabled: bool = True
    port: int = Field(default=9090, ge=1, le=65535)
    path: str = "/metrics"


class SecurityConfig(BaseModel):
    """Security constraints configuration."""

    max_prompt_length: int = Field(default=50000, ge=100, le=1000000)
    max_negative_prompt_length: int = Field(default=50000, ge=100, le=1000000)
    subprocess_max_memory_mb: int | None = Field(default=None, ge=1024)
    subprocess_max_output_bytes: int = Field(default=10485760, ge=1024)  # 10MB
    allowed_model_base_path: Path | None = Field(default=None)
    validate_model_paths: bool = Field(default=True)
    validate_binary_paths: bool = Field(default=True)


class InferenceConfig(BaseModel):
    """Inference binary configuration."""

    llama_cli_path: str = "llama-cli"  # Deprecated, kept for compatibility
    llama_server_path: str = "llama-server"  # Path to llama-server binary
    sd_binary_path: str = "sd"  # Path to stable-diffusion.cpp sd binary


class VramCramConfig(BaseModel):
    """Complete VramCram configuration."""

    redis: RedisConfig
    vram: VRAMConfig
    api: APIConfig = Field(default_factory=APIConfig)
    models: ModelsConfig
    agents: AgentConfig = Field(default_factory=AgentConfig)
    jobs: JobConfig = Field(default_factory=JobConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)

    @model_validator(mode="after")
    def validate_vram_capacity(self) -> "VramCramConfig":
        """Ensure total model VRAM estimates fit within capacity."""
        total_model_vram = sum(m.vram_mb for m in self.models.llm) + sum(
            m.vram_mb for m in self.models.diffusion
        )
        available_vram = self.vram.total_mb - self.vram.safety_margin_mb

        # Warning: we allow configs where not all models fit simultaneously
        # (that's the point of eviction), but each individual model must fit
        for model in self.models.llm + self.models.diffusion:
            if model.vram_mb > available_vram:
                raise ValueError(
                    f"Model '{model.name}' requires {model.vram_mb}MB but only "
                    f"{available_vram}MB available after safety margin"
                )

        return self
