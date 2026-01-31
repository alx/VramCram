"""Unit tests for VRAM tracker."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from vramcram.gpu.vram_tracker import VRAMState, VRAMTracker


class TestVRAMTracker:
    """Test suite for VRAMTracker."""

    @patch("vramcram.gpu.vram_tracker.pynvml")
    def test_init_success(self, mock_pynvml: MagicMock) -> None:
        """Test successful initialization."""
        mock_handle = MagicMock()
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle

        tracker = VRAMTracker(gpu_index=0)

        assert tracker.gpu_index == 0
        assert tracker.handle == mock_handle
        mock_pynvml.nvmlInit.assert_called_once()
        mock_pynvml.nvmlDeviceGetHandleByIndex.assert_called_once_with(0)

    @patch("vramcram.gpu.vram_tracker.pynvml", None)
    def test_init_no_pynvml(self) -> None:
        """Test initialization fails when pynvml not available."""
        with pytest.raises(RuntimeError, match="pynvml not available"):
            VRAMTracker()

    @patch("vramcram.gpu.vram_tracker.pynvml")
    def test_get_vram_state(self, mock_pynvml: MagicMock) -> None:
        """Test getting VRAM state."""
        mock_handle = MagicMock()
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle

        # Mock memory info
        mock_mem_info = MagicMock()
        mock_mem_info.total = 16 * 1024 * 1024 * 1024  # 16GB in bytes
        mock_mem_info.used = 8 * 1024 * 1024 * 1024  # 8GB in bytes
        mock_mem_info.free = 8 * 1024 * 1024 * 1024  # 8GB in bytes
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mock_mem_info

        tracker = VRAMTracker(gpu_index=0)
        state = tracker.get_vram_state()

        assert isinstance(state, VRAMState)
        assert state.total_mb == 16 * 1024
        assert state.used_mb == 8 * 1024
        assert state.free_mb == 8 * 1024
        assert isinstance(state.timestamp, datetime)

    @patch("vramcram.gpu.vram_tracker.pynvml")
    def test_has_sufficient_vram_true(self, mock_pynvml: MagicMock) -> None:
        """Test sufficient VRAM check returns True."""
        mock_handle = MagicMock()
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle

        mock_mem_info = MagicMock()
        mock_mem_info.total = 16 * 1024 * 1024 * 1024
        mock_mem_info.used = 4 * 1024 * 1024 * 1024
        mock_mem_info.free = 12 * 1024 * 1024 * 1024  # 12GB free
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mock_mem_info

        tracker = VRAMTracker(gpu_index=0)

        # Need 4GB, have 12GB - should be sufficient
        assert tracker.has_sufficient_vram(4 * 1024, safety_margin_mb=0) is True

        # Need 4GB with 2GB margin, have 12GB - should be sufficient
        assert tracker.has_sufficient_vram(4 * 1024, safety_margin_mb=2 * 1024) is True

    @patch("vramcram.gpu.vram_tracker.pynvml")
    def test_has_sufficient_vram_false(self, mock_pynvml: MagicMock) -> None:
        """Test insufficient VRAM check returns False."""
        mock_handle = MagicMock()
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle

        mock_mem_info = MagicMock()
        mock_mem_info.total = 16 * 1024 * 1024 * 1024
        mock_mem_info.used = 14 * 1024 * 1024 * 1024
        mock_mem_info.free = 2 * 1024 * 1024 * 1024  # 2GB free
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mock_mem_info

        tracker = VRAMTracker(gpu_index=0)

        # Need 4GB, have 2GB - should be insufficient
        assert tracker.has_sufficient_vram(4 * 1024, safety_margin_mb=0) is False

        # Need 1GB with 2GB margin, have 2GB - should be insufficient
        assert tracker.has_sufficient_vram(1 * 1024, safety_margin_mb=2 * 1024) is False
