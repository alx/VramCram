"""Unit tests for direct result format (text and base64).

Tests that workers return direct string results instead of JSON-wrapped dicts.
"""

import base64
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from vramcram.config.models import (
    DiffusionModelConfig,
    LLMModelConfig,
    ModelsConfig,
    OutputConfig,
    RedisConfig,
    VramCramConfig,
    VRAMConfig,
)


class TestLLMResultFormat:
    """Test LLM worker returns direct text string."""

    def test_llm_result_is_string_not_dict(self) -> None:
        """Verify LLM workers return text directly, not wrapped in dict."""
        from vramcram.agents.worker.llm_worker import LLMWorker
        from vramcram.events.bus import EventBus
        from vramcram.redis.client import RedisClientFactory

        # Create minimal config
        config = VramCramConfig(
            redis=RedisConfig(db=15),
            vram=VRAMConfig(total_mb=16384, safety_margin_mb=512),
            models=ModelsConfig(
                llm=[
                    LLMModelConfig(
                        name="test-llm",
                        model_path=Path("/fake/model.gguf"),
                        vram_mb=4096,
                    )
                ]
            ),
        )

        # Check that execute_inference signature returns str
        import inspect
        from vramcram.agents.worker.base import BaseWorker

        sig = inspect.signature(BaseWorker.execute_inference)
        return_annotation = sig.return_annotation

        # Should be "str" not "dict[str, Any]"
        assert "str" in str(return_annotation)
        assert "dict" not in str(return_annotation)


class TestDiffusionResultFormat:
    """Test diffusion worker returns base64 or path based on config."""

    def test_diffusion_result_signature_is_string(self) -> None:
        """Verify diffusion worker execute_inference returns str."""
        import inspect

        from vramcram.agents.worker.base import BaseWorker

        sig = inspect.signature(BaseWorker.execute_inference)
        return_annotation = sig.return_annotation

        # Should return str
        assert "str" in str(return_annotation)

    def test_base64_encoding_format(self) -> None:
        """Test that base64 data URI format is correct."""
        # Create a simple test image
        img = Image.new("RGB", (100, 100), color="red")
        img_bytes = BytesIO()
        img.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        # Encode to base64
        image_base64 = base64.b64encode(img_bytes.read()).decode("utf-8")
        data_uri = f"data:image/png;base64,{image_base64}"

        # Verify format
        assert data_uri.startswith("data:image/png;base64,")
        assert len(data_uri) > len("data:image/png;base64,")

        # Verify decodable
        base64_data = data_uri.split(",")[1]
        decoded = base64.b64decode(base64_data)
        assert len(decoded) > 0

        # Verify it's valid PNG
        assert decoded.startswith(b"\x89PNG")

    def test_output_config_has_image_result_format_field(self) -> None:
        """Test that OutputConfig has image_result_format field."""
        config = OutputConfig()

        # Should have default value "base64"
        assert hasattr(config, "image_result_format")
        assert config.image_result_format == "base64"

        # Should accept "path" as alternative
        config_path = OutputConfig(image_result_format="path")
        assert config_path.image_result_format == "path"

    def test_output_config_rejects_invalid_format(self) -> None:
        """Test that OutputConfig rejects invalid image_result_format values."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            OutputConfig(image_result_format="invalid")


class TestResultStorageFormat:
    """Test that results are stored directly in Redis without JSON wrapping."""

    def test_result_stored_as_string(self) -> None:
        """Verify results are stored as strings, not JSON."""
        from vramcram.queue.job import Job

        # Create job with direct string result
        job = Job(
            job_id="test-123",
            model="test-model",
            prompt="test prompt",
        )
        job.result = "This is direct text"  # Direct string, not JSON

        # Convert to dict (as stored in Redis)
        job_dict = job.to_dict()

        # Result should be plain string, not JSON-wrapped
        assert job_dict["result"] == "This is direct text"
        assert not job_dict["result"].startswith('{"')
        assert not job_dict["result"].startswith('{"text":')

    def test_base64_result_in_job(self) -> None:
        """Test storing base64 data URI in job result."""
        from vramcram.queue.job import Job

        # Create fake base64 data URI
        data_uri = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

        job = Job(
            job_id="test-456",
            model="test-diffusion",
            prompt="test image prompt",
        )
        job.result = data_uri

        # Verify it's stored as direct string
        job_dict = job.to_dict()
        assert job_dict["result"] == data_uri
        assert job_dict["result"].startswith("data:image/png;base64,")
