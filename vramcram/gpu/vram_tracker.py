"""VRAM monitoring using pynvml."""

from dataclasses import dataclass
from datetime import datetime, timezone

import structlog

try:
    import pynvml
except ImportError:
    pynvml = None  # type: ignore

logger = structlog.get_logger()


@dataclass
class VRAMState:
    """Snapshot of GPU memory state.

    Attributes:
        free_mb: Available VRAM in megabytes.
        used_mb: Used VRAM in megabytes.
        total_mb: Total VRAM in megabytes.
        timestamp: When this snapshot was taken.
    """

    free_mb: int
    used_mb: int
    total_mb: int
    timestamp: datetime


class VRAMTracker:
    """Tracks GPU VRAM usage using pynvml.

    This class wraps NVIDIA Management Library (NVML) to query GPU memory
    information. It provides a simplified interface for checking available
    VRAM and determining if sufficient memory exists for model loading.
    """

    def __init__(self, gpu_index: int = 0) -> None:
        """Initialize VRAM tracker.

        Args:
            gpu_index: GPU device index to monitor (default: 0).

        Raises:
            RuntimeError: If pynvml is not available or initialization fails.
        """
        if pynvml is None:
            raise RuntimeError(
                "pynvml not available. Install with: pip install pynvml"
            )

        self.gpu_index = gpu_index
        self.logger = logger.bind(gpu_index=gpu_index)

        try:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            self.logger.info("vram_tracker_initialized")
        except pynvml.NVMLError as e:
            self.logger.error("vram_tracker_init_failed", error=str(e))
            raise RuntimeError(f"Failed to initialize NVML: {e}") from e

    def get_vram_state(self) -> VRAMState:
        """Get current VRAM state.

        Returns:
            VRAMState with current memory usage.

        Raises:
            RuntimeError: If NVML query fails.
        """
        try:
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)

            # Convert bytes to MB
            total_mb = mem_info.total // (1024 * 1024)
            used_mb = mem_info.used // (1024 * 1024)
            free_mb = mem_info.free // (1024 * 1024)

            state = VRAMState(
                free_mb=free_mb,
                used_mb=used_mb,
                total_mb=total_mb,
                timestamp=datetime.now(timezone.utc),
            )

            self.logger.debug(
                "vram_state_queried",
                free_mb=free_mb,
                used_mb=used_mb,
                total_mb=total_mb,
            )

            return state
        except pynvml.NVMLError as e:
            self.logger.error("vram_query_failed", error=str(e))
            raise RuntimeError(f"Failed to query VRAM: {e}") from e

    def has_sufficient_vram(self, required_mb: int, safety_margin_mb: int = 0) -> bool:
        """Check if sufficient VRAM is available.

        Args:
            required_mb: Required VRAM in megabytes.
            safety_margin_mb: Additional safety margin to reserve.

        Returns:
            True if sufficient VRAM is available, False otherwise.
        """
        state = self.get_vram_state()
        available = state.free_mb - safety_margin_mb
        sufficient = available >= required_mb

        self.logger.debug(
            "vram_sufficiency_check",
            required_mb=required_mb,
            safety_margin_mb=safety_margin_mb,
            available_mb=available,
            sufficient=sufficient,
        )

        return sufficient

    def __del__(self) -> None:
        """Cleanup NVML on destruction."""
        try:
            if pynvml is not None:
                pynvml.nvmlShutdown()
        except Exception:
            # Ignore errors during cleanup
            pass
