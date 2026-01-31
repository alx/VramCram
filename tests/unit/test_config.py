"""Tests for configuration system."""

import tempfile
from pathlib import Path

import pytest

from vramcram.config.loader import load_config
from vramcram.config.models import (
    APIConfig,
    DiffusionModelConfig,
    LLMModelConfig,
    ModelsConfig,
    RedisConfig,
    VRAMConfig,
    VramCramConfig,
)


def test_redis_config_defaults() -> None:
    """Test RedisConfig with default values."""
    config = RedisConfig()
    assert config.host == "localhost"
    assert config.port == 6379
    assert config.db == 0
    assert config.password is None
    assert config.max_connections == 50


def test_redis_config_validation() -> None:
    """Test RedisConfig port validation."""
    with pytest.raises(Exception):
        RedisConfig(port=0)  # Invalid port

    with pytest.raises(Exception):
        RedisConfig(port=70000)  # Port too high


def test_vram_config() -> None:
    """Test VRAMConfig."""
    config = VRAMConfig(total_mb=24576, safety_margin_mb=2048)
    assert config.total_mb == 24576
    assert config.safety_margin_mb == 2048
    assert config.monitoring_interval_seconds == 5


def test_vram_config_safety_margin_validation() -> None:
    """Test safety margin must be less than total."""
    with pytest.raises(Exception):
        VRAMConfig(total_mb=8192, safety_margin_mb=8192)

    with pytest.raises(Exception):
        VRAMConfig(total_mb=8192, safety_margin_mb=10000)


def test_api_config_defaults() -> None:
    """Test APIConfig defaults."""
    config = APIConfig()
    assert config.host == "0.0.0.0"
    assert config.port == 8000
    assert config.workers == 1


def test_llm_model_config() -> None:
    """Test LLMModelConfig."""
    config = LLMModelConfig(
        name="test-model",
        model_path=Path("/models/test.gguf"),
        vram_mb=4096,
    )
    assert config.name == "test-model"
    assert config.type == "llm"
    assert config.model_path == Path("/models/test.gguf")
    assert config.vram_mb == 4096
    assert "n_gpu_layers" in config.config


def test_diffusion_model_config() -> None:
    """Test DiffusionModelConfig."""
    config = DiffusionModelConfig(
        name="test-sd",
        model_path=Path("/models/sd.safetensors"),
        vram_mb=4096,
    )
    assert config.name == "test-sd"
    assert config.type == "diffusion"
    assert config.vram_mb == 4096
    assert "width" in config.config


def test_models_config_unique_names() -> None:
    """Test that model names must be unique."""
    llm1 = LLMModelConfig(
        name="duplicate",
        model_path=Path("/models/test1.gguf"),
        vram_mb=4096,
    )
    llm2 = LLMModelConfig(
        name="duplicate",
        model_path=Path("/models/test2.gguf"),
        vram_mb=4096,
    )

    with pytest.raises(Exception):
        ModelsConfig(llm=[llm1, llm2])


def test_models_config_unique_across_types() -> None:
    """Test that model names must be unique across LLM and diffusion."""
    llm = LLMModelConfig(
        name="duplicate",
        model_path=Path("/models/test.gguf"),
        vram_mb=4096,
    )
    diffusion = DiffusionModelConfig(
        name="duplicate",
        model_path=Path("/models/sd.safetensors"),
        vram_mb=4096,
    )

    with pytest.raises(Exception):
        ModelsConfig(llm=[llm], diffusion=[diffusion])


def test_vramcram_config_validates_model_vram() -> None:
    """Test that individual models must fit in available VRAM."""
    redis_cfg = RedisConfig()
    vram_cfg = VRAMConfig(total_mb=8192, safety_margin_mb=2048)
    models_cfg = ModelsConfig(
        llm=[
            LLMModelConfig(
                name="too-large",
                model_path=Path("/models/huge.gguf"),
                vram_mb=10000,  # Exceeds available VRAM (8192 - 2048 = 6144)
            )
        ]
    )

    with pytest.raises(Exception):
        VramCramConfig(redis=redis_cfg, vram=vram_cfg, models=models_cfg)


def test_vramcram_config_allows_total_exceeding_vram() -> None:
    """Test that total model VRAM can exceed capacity (eviction handles this)."""
    redis_cfg = RedisConfig()
    vram_cfg = VRAMConfig(total_mb=8192, safety_margin_mb=2048)
    # Total is 8192MB, but available is 6144MB after safety margin
    # Two models at 4096MB each = 8192MB total, but each fits individually
    models_cfg = ModelsConfig(
        llm=[
            LLMModelConfig(
                name="model1",
                model_path=Path("/models/model1.gguf"),
                vram_mb=4096,
            ),
            LLMModelConfig(
                name="model2",
                model_path=Path("/models/model2.gguf"),
                vram_mb=4096,
            ),
        ]
    )

    # This should succeed - eviction will handle swapping
    config = VramCramConfig(redis=redis_cfg, vram=vram_cfg, models=models_cfg)
    assert len(config.models.llm) == 2


def test_load_config_file_not_found() -> None:
    """Test load_config with non-existent file."""
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/config.yaml")


def test_load_config_empty_file() -> None:
    """Test load_config with empty file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("")
        temp_path = f.name

    try:
        with pytest.raises(ValueError, match="empty"):
            load_config(temp_path)
    finally:
        Path(temp_path).unlink()


def test_load_config_valid() -> None:
    """Test load_config with valid YAML."""
    config_yaml = """
redis:
  host: localhost
  port: 6379
  db: 0

vram:
  total_mb: 24576
  safety_margin_mb: 2048

models:
  llm:
    - name: test-llm
      type: llm
      model_path: /models/test.gguf
      vram_mb: 4096
  diffusion:
    - name: test-sd
      type: diffusion
      model_path: /models/sd.safetensors
      vram_mb: 4096
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(config_yaml)
        temp_path = f.name

    try:
        config = load_config(temp_path)
        assert config.redis.host == "localhost"
        assert config.vram.total_mb == 24576
        assert len(config.models.llm) == 1
        assert len(config.models.diffusion) == 1
        assert config.models.llm[0].name == "test-llm"
    finally:
        Path(temp_path).unlink()


def test_load_config_invalid_yaml() -> None:
    """Test load_config with invalid configuration."""
    config_yaml = """
redis:
  host: localhost
  port: 99999  # Invalid port

vram:
  total_mb: 8192
  safety_margin_mb: 1024

models:
  llm: []
  diffusion: []
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(config_yaml)
        temp_path = f.name

    try:
        with pytest.raises(ValueError, match="Invalid configuration"):
            load_config(temp_path)
    finally:
        Path(temp_path).unlink()
