"""Integration tests for security validation in API and workers."""

import pytest
from pydantic import ValidationError

from vramcram.api.models import JobSubmitRequest
from vramcram.api.params import BaseDiffusionParams, BaseLLMParams


class TestAPIRequestValidation:
    """Test API request validation."""

    def test_valid_job_submission(self):
        """Test that valid job submissions are accepted."""
        request = JobSubmitRequest(
            model="test-model",
            prompt="This is a valid prompt",
            params={"max_tokens": 100},
        )
        assert request.prompt == "This is a valid prompt"

    def test_null_byte_in_prompt_rejection(self):
        """Test that prompts with null bytes are rejected."""
        with pytest.raises(ValidationError, match="null bytes"):
            JobSubmitRequest(
                model="test-model",
                prompt="Invalid\x00prompt",
                params={},
            )

    def test_empty_prompt_rejection(self):
        """Test that empty prompts are rejected."""
        with pytest.raises(ValidationError, match="empty"):
            JobSubmitRequest(
                model="test-model",
                prompt="",
                params={},
            )

    def test_whitespace_only_prompt_rejection(self):
        """Test that whitespace-only prompts are rejected."""
        with pytest.raises(ValidationError, match="empty"):
            JobSubmitRequest(
                model="test-model",
                prompt="   \n\t  ",
                params={},
            )


class TestLLMParamsValidation:
    """Test LLM parameter validation."""

    def test_valid_llm_params(self):
        """Test that valid LLM params are accepted."""
        params = BaseLLMParams(
            max_tokens=512,
            temperature=0.7,
            top_p=0.95,
            top_k=40,
        )
        assert params.max_tokens == 512

    def test_max_tokens_range(self):
        """Test max_tokens range validation."""
        # Valid
        BaseLLMParams(max_tokens=1)
        BaseLLMParams(max_tokens=32768)

        # Invalid - too low
        with pytest.raises(ValidationError):
            BaseLLMParams(max_tokens=0)

        # Invalid - too high
        with pytest.raises(ValidationError):
            BaseLLMParams(max_tokens=32769)

    def test_temperature_range(self):
        """Test temperature range validation."""
        # Valid
        BaseLLMParams(temperature=0.0)
        BaseLLMParams(temperature=2.0)

        # Invalid
        with pytest.raises(ValidationError):
            BaseLLMParams(temperature=-0.1)
        with pytest.raises(ValidationError):
            BaseLLMParams(temperature=2.1)

    def test_top_p_range(self):
        """Test top_p range validation."""
        # Valid
        BaseLLMParams(top_p=0.0)
        BaseLLMParams(top_p=1.0)

        # Invalid
        with pytest.raises(ValidationError):
            BaseLLMParams(top_p=-0.1)
        with pytest.raises(ValidationError):
            BaseLLMParams(top_p=1.1)

    def test_unknown_field_rejection(self):
        """Test that unknown fields are rejected."""
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            BaseLLMParams(unknown_field="value")


class TestDiffusionParamsValidation:
    """Test diffusion parameter validation."""

    def test_valid_diffusion_params(self):
        """Test that valid diffusion params are accepted."""
        params = BaseDiffusionParams(
            negative_prompt="bad quality",
            width=512,
            height=512,
            sample_steps=4,
            cfg_scale=1.0,
        )
        assert params.width == 512

    def test_null_byte_in_negative_prompt_rejection(self):
        """Test that negative prompts with null bytes are rejected."""
        with pytest.raises(ValidationError, match="null bytes"):
            BaseDiffusionParams(negative_prompt="Invalid\x00prompt")

    def test_dimension_multiple_of_8(self):
        """Test that dimensions must be multiples of 8."""
        # Valid
        BaseDiffusionParams(width=512, height=512)
        BaseDiffusionParams(width=64, height=64)
        BaseDiffusionParams(width=2048, height=2048)

        # Invalid - not multiple of 8
        with pytest.raises(ValidationError, match="multiple of 8"):
            BaseDiffusionParams(width=511)

        with pytest.raises(ValidationError, match="multiple of 8"):
            BaseDiffusionParams(height=513)

    def test_dimension_range(self):
        """Test dimension range validation."""
        # Valid
        BaseDiffusionParams(width=64, height=64)
        BaseDiffusionParams(width=2048, height=2048)

        # Invalid - too small
        with pytest.raises(ValidationError):
            BaseDiffusionParams(width=56)

        # Invalid - too large
        with pytest.raises(ValidationError):
            BaseDiffusionParams(width=2056)

    def test_sample_steps_range(self):
        """Test sample_steps range validation."""
        # Valid
        BaseDiffusionParams(sample_steps=1)
        BaseDiffusionParams(sample_steps=150)

        # Invalid
        with pytest.raises(ValidationError):
            BaseDiffusionParams(sample_steps=0)
        with pytest.raises(ValidationError):
            BaseDiffusionParams(sample_steps=151)

    def test_cfg_scale_range(self):
        """Test cfg_scale range validation."""
        # Valid
        BaseDiffusionParams(cfg_scale=0.0)
        BaseDiffusionParams(cfg_scale=30.0)

        # Invalid
        with pytest.raises(ValidationError):
            BaseDiffusionParams(cfg_scale=-0.1)
        with pytest.raises(ValidationError):
            BaseDiffusionParams(cfg_scale=30.1)

    def test_unknown_field_rejection(self):
        """Test that unknown fields are rejected."""
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            BaseDiffusionParams(unknown_field="value")

    def test_empty_negative_prompt(self):
        """Test that empty negative prompts are allowed."""
        params = BaseDiffusionParams(negative_prompt="")
        assert params.negative_prompt == ""

    def test_negative_prompt_max_length(self):
        """Test negative prompt max length."""
        # Valid
        BaseDiffusionParams(negative_prompt="x" * 50000)

        # Invalid - too long
        with pytest.raises(ValidationError):
            BaseDiffusionParams(negative_prompt="x" * 50001)
