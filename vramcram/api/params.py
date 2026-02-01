"""Parameter validation schemas for API requests."""

from pydantic import BaseModel, ConfigDict, Field, field_validator


class BaseLLMParams(BaseModel):
    """Base parameters for LLM inference."""

    model_config = ConfigDict(extra="forbid")  # Reject unknown fields

    max_tokens: int = Field(default=512, ge=1, le=32768)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    top_k: int = Field(default=40, ge=0, le=200)


class BaseDiffusionParams(BaseModel):
    """Base parameters for diffusion inference."""

    model_config = ConfigDict(extra="forbid")  # Reject unknown fields

    negative_prompt: str = Field(default="", max_length=50000)
    width: int = Field(default=512, ge=64, le=2048)
    height: int = Field(default=512, ge=64, le=2048)
    sample_steps: int = Field(default=4, ge=1, le=150)
    cfg_scale: float = Field(default=1.0, ge=0.0, le=30.0)

    @field_validator("negative_prompt")
    @classmethod
    def validate_negative_prompt(cls, v: str) -> str:
        """Validate negative prompt doesn't contain null bytes."""
        if "\x00" in v:
            raise ValueError("Negative prompt contains null bytes which are not allowed")
        return v

    @field_validator("width", "height")
    @classmethod
    def validate_dimensions(cls, v: int) -> int:
        """Validate dimensions are multiples of 8."""
        if v % 8 != 0:
            raise ValueError(f"Dimension must be a multiple of 8, got {v}")
        return v
