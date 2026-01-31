"""Unit tests for ModelRegistry."""

from datetime import datetime, timedelta, timezone

import pytest

from vramcram.queue.registry import LoadedModel, ModelRegistry, ModelState


class TestModelRegistry:
    """Test suite for ModelRegistry."""

    def test_add_model(self) -> None:
        """Test adding a model to registry."""
        registry = ModelRegistry()

        registry.add_model("model-1", vram_mb=4096, worker_pid=123)

        assert "model-1" in registry.loaded_models
        model = registry.loaded_models["model-1"]
        assert model.model_name == "model-1"
        assert model.vram_mb == 4096
        assert model.worker_pid == 123
        assert model.state == ModelState.LOADED
        assert isinstance(model.last_used, datetime)

    def test_remove_model(self) -> None:
        """Test removing a model from registry."""
        registry = ModelRegistry()
        registry.add_model("model-1", vram_mb=4096)

        assert "model-1" in registry.loaded_models

        registry.remove_model("model-1")

        assert "model-1" not in registry.loaded_models

    def test_remove_nonexistent_model(self) -> None:
        """Test removing nonexistent model doesn't error."""
        registry = ModelRegistry()

        # Should not raise
        registry.remove_model("nonexistent")

    def test_update_last_used(self) -> None:
        """Test updating last used timestamp."""
        registry = ModelRegistry()
        registry.add_model("model-1", vram_mb=4096)

        original_time = registry.loaded_models["model-1"].last_used

        # Wait a bit
        import time

        time.sleep(0.01)

        registry.update_last_used("model-1")

        updated_time = registry.loaded_models["model-1"].last_used
        assert updated_time > original_time

    def test_update_state(self) -> None:
        """Test updating model state."""
        registry = ModelRegistry()
        registry.add_model("model-1", vram_mb=4096, state=ModelState.LOADING)

        assert registry.loaded_models["model-1"].state == ModelState.LOADING

        registry.update_state("model-1", ModelState.LOADED)

        assert registry.loaded_models["model-1"].state == ModelState.LOADED

    def test_get_model(self) -> None:
        """Test getting model info."""
        registry = ModelRegistry()
        registry.add_model("model-1", vram_mb=4096)

        model = registry.get_model("model-1")

        assert model is not None
        assert model.model_name == "model-1"
        assert model.vram_mb == 4096

    def test_get_nonexistent_model(self) -> None:
        """Test getting nonexistent model returns None."""
        registry = ModelRegistry()

        model = registry.get_model("nonexistent")

        assert model is None

    def test_is_loaded(self) -> None:
        """Test checking if model is loaded."""
        registry = ModelRegistry()
        registry.add_model("model-1", vram_mb=4096, state=ModelState.LOADED)
        registry.add_model("model-2", vram_mb=2048, state=ModelState.LOADING)

        assert registry.is_loaded("model-1") is True
        assert registry.is_loaded("model-2") is False
        assert registry.is_loaded("nonexistent") is False

    def test_get_lru_model(self) -> None:
        """Test getting least recently used model."""
        registry = ModelRegistry()

        # Add models with different last_used times
        now = datetime.now(timezone.utc)

        registry.add_model("model-1", vram_mb=4096, state=ModelState.LOADED)
        registry.loaded_models["model-1"].last_used = now - timedelta(hours=3)

        registry.add_model("model-2", vram_mb=2048, state=ModelState.LOADED)
        registry.loaded_models["model-2"].last_used = now - timedelta(hours=1)

        registry.add_model("model-3", vram_mb=3072, state=ModelState.LOADED)
        registry.loaded_models["model-3"].last_used = now - timedelta(hours=2)

        # model-1 has oldest last_used time
        lru = registry.get_lru_model()
        assert lru == "model-1"

    def test_get_lru_model_only_loaded(self) -> None:
        """Test get_lru_model only considers LOADED models."""
        registry = ModelRegistry()

        now = datetime.now(timezone.utc)

        # Oldest but not loaded
        registry.add_model("model-1", vram_mb=4096, state=ModelState.LOADING)
        registry.loaded_models["model-1"].last_used = now - timedelta(hours=5)

        # Loaded
        registry.add_model("model-2", vram_mb=2048, state=ModelState.LOADED)
        registry.loaded_models["model-2"].last_used = now - timedelta(hours=2)

        registry.add_model("model-3", vram_mb=3072, state=ModelState.LOADED)
        registry.loaded_models["model-3"].last_used = now - timedelta(hours=1)

        # Should return model-2, not model-1
        lru = registry.get_lru_model()
        assert lru == "model-2"

    def test_get_lru_model_empty(self) -> None:
        """Test get_lru_model with no loaded models."""
        registry = ModelRegistry()

        lru = registry.get_lru_model()
        assert lru is None

    def test_get_total_vram_used(self) -> None:
        """Test calculating total VRAM used."""
        registry = ModelRegistry()

        registry.add_model("model-1", vram_mb=4096, state=ModelState.LOADED)
        registry.add_model("model-2", vram_mb=2048, state=ModelState.LOADING)
        registry.add_model("model-3", vram_mb=1024, state=ModelState.EVICTING)

        # Should count LOADED and LOADING, not EVICTING
        total = registry.get_total_vram_used()
        assert total == 6144  # 4096 + 2048

    def test_list_models(self) -> None:
        """Test listing all models."""
        registry = ModelRegistry()

        registry.add_model("model-1", vram_mb=4096)
        registry.add_model("model-2", vram_mb=2048)

        models = registry.list_models()

        assert len(models) == 2
        assert any(m.model_name == "model-1" for m in models)
        assert any(m.model_name == "model-2" for m in models)
