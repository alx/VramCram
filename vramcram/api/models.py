"""API request and response models."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator


class JobSubmitRequest(BaseModel):
    """Request to submit a new job."""

    model: str = Field(description="Target model name")
    prompt: str = Field(description="Inference prompt/input")
    params: dict[str, Any] = Field(
        default_factory=dict, description="Model-specific parameters"
    )

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, v: str) -> str:
        """Validate prompt doesn't contain null bytes and isn't empty."""
        if "\x00" in v:
            raise ValueError("Prompt contains null bytes which are not allowed")
        if not v.strip():
            raise ValueError("Prompt cannot be empty or whitespace-only")
        return v


class JobSubmitResponse(BaseModel):
    """Response from job submission."""

    job_id: str = Field(description="Unique job identifier")
    model: str = Field(description="Target model name")
    status: str = Field(description="Initial job status (queued)")
    created_at: datetime = Field(description="Job creation timestamp")


class JobStatusResponse(BaseModel):
    """Response for job status query."""

    job_id: str = Field(description="Job identifier")
    model: str = Field(description="Target model name")
    status: str = Field(description="Current job status")
    created_at: datetime = Field(description="Job creation timestamp")
    dispatched_at: datetime | None = Field(
        default=None, description="When job was dispatched"
    )
    started_at: datetime | None = Field(
        default=None, description="When processing started"
    )
    completed_at: datetime | None = Field(
        default=None, description="When processing completed"
    )
    duration_ms: int | None = Field(default=None, description="Processing duration in ms")
    error: str | None = Field(default=None, description="Error message if failed")


class JobResultResponse(BaseModel):
    """Response for job result query."""

    job_id: str = Field(description="Job identifier")
    status: str = Field(description="Job status")
    result: str | None = Field(default=None, description="Job result data")
    error: str | None = Field(default=None, description="Error message if failed")


class ModelInfo(BaseModel):
    """Information about an available model."""

    name: str = Field(description="Model identifier")
    type: str = Field(description="Model type (llm or diffusion)")
    vram_mb: int = Field(description="Estimated VRAM usage in MB")
    loaded: bool = Field(
        default=False, description="Whether model is currently loaded"
    )


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(description="Overall system status")
    redis_connected: bool = Field(description="Redis connection status")
    models_available: int = Field(description="Number of available models")
    jobs_queued: int = Field(default=0, description="Number of queued jobs")
