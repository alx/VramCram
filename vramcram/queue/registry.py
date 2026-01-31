"""Model registry for tracking loaded models."""

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

import structlog

logger = structlog.get_logger()


class ModelState(str, Enum):
    """Model lifecycle state."""

    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    EVICTING = "evicting"


@dataclass
class LoadedModel:
    """Tracks a loaded model's state.

    Attributes:
        model_name: Unique model identifier.
        vram_mb: VRAM usage in megabytes.
        last_used: Timestamp of most recent use (for LRU eviction).
        state: Current model state.
        worker_pid: Process ID of worker managing this model.
    """

    model_name: str
    vram_mb: int
    last_used: datetime
    state: ModelState
    worker_pid: int | None = None


class ModelRegistry:
    """Registry for tracking loaded models and LRU eviction.

    The registry maintains the set of currently loaded models and tracks
    their last usage time for Least Recently Used (LRU) eviction decisions.
    """

    def __init__(self) -> None:
        """Initialize model registry."""
        self.loaded_models: dict[str, LoadedModel] = {}
        self.logger = logger.bind(component="model_registry")

    def add_model(
        self,
        model_name: str,
        vram_mb: int,
        worker_pid: int | None = None,
        state: ModelState = ModelState.LOADED,
    ) -> None:
        """Add or update a model in the registry.

        Args:
            model_name: Model identifier.
            vram_mb: VRAM usage in MB.
            worker_pid: Worker process ID.
            state: Model state.
        """
        now = datetime.now(timezone.utc)

        if model_name in self.loaded_models:
            self.logger.debug("updating_model", model=model_name, state=state.value)
        else:
            self.logger.info("adding_model", model=model_name, vram_mb=vram_mb)

        self.loaded_models[model_name] = LoadedModel(
            model_name=model_name,
            vram_mb=vram_mb,
            last_used=now,
            state=state,
            worker_pid=worker_pid,
        )

    def remove_model(self, model_name: str) -> None:
        """Remove a model from the registry.

        Args:
            model_name: Model identifier.
        """
        if model_name in self.loaded_models:
            self.logger.info("removing_model", model=model_name)
            del self.loaded_models[model_name]
        else:
            self.logger.warning("remove_nonexistent_model", model=model_name)

    def update_last_used(self, model_name: str) -> None:
        """Update the last used timestamp for a model.

        This is called when a job is dispatched to the model, maintaining
        accurate LRU ordering for eviction decisions.

        Args:
            model_name: Model identifier.
        """
        if model_name in self.loaded_models:
            self.loaded_models[model_name].last_used = datetime.now(timezone.utc)
            self.logger.debug("updated_last_used", model=model_name)
        else:
            self.logger.warning("update_nonexistent_model", model=model_name)

    def update_state(self, model_name: str, state: ModelState) -> None:
        """Update model state.

        Args:
            model_name: Model identifier.
            state: New state.
        """
        if model_name in self.loaded_models:
            self.loaded_models[model_name].state = state
            self.logger.debug("updated_model_state", model=model_name, state=state.value)
        else:
            self.logger.warning("update_nonexistent_model", model=model_name)

    def get_model(self, model_name: str) -> LoadedModel | None:
        """Get model information.

        Args:
            model_name: Model identifier.

        Returns:
            LoadedModel instance or None if not found.
        """
        return self.loaded_models.get(model_name)

    def is_loaded(self, model_name: str) -> bool:
        """Check if a model is loaded.

        Args:
            model_name: Model identifier.

        Returns:
            True if model is in LOADED state, False otherwise.
        """
        model = self.get_model(model_name)
        return model is not None and model.state == ModelState.LOADED

    def is_loading(self, model_name: str) -> bool:
        """Check if a model is currently loading.

        Args:
            model_name: Model identifier.

        Returns:
            True if model is in LOADING state, False otherwise.
        """
        model = self.get_model(model_name)
        return model is not None and model.state == ModelState.LOADING

    def get_lru_model(self) -> str | None:
        """Get the least recently used loaded model.

        This is used by the Coordinator for eviction decisions. Only models
        in the LOADED state are considered for eviction.

        Returns:
            Model name of LRU model, or None if no loaded models.
        """
        loaded_only = {
            name: model
            for name, model in self.loaded_models.items()
            if model.state == ModelState.LOADED
        }

        if not loaded_only:
            return None

        # Find model with oldest last_used timestamp
        lru_name = min(loaded_only.items(), key=lambda item: item[1].last_used)[0]

        self.logger.debug("lru_model_selected", model=lru_name)
        return lru_name

    def get_total_vram_used(self) -> int:
        """Calculate total VRAM used by all loaded models.

        Returns:
            Total VRAM in MB.
        """
        return sum(
            model.vram_mb
            for model in self.loaded_models.values()
            if model.state in (ModelState.LOADED, ModelState.LOADING)
        )

    def list_models(self) -> list[LoadedModel]:
        """Get list of all models in registry.

        Returns:
            List of LoadedModel instances.
        """
        return list(self.loaded_models.values())
