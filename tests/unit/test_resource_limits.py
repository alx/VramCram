"""Unit tests for resource limiting."""

import platform

import pytest

from vramcram.security.resource_limits import create_resource_limiter


class TestResourceLimiter:
    """Test resource limiter creation."""

    def test_no_limit_returns_none(self):
        """Test that None is returned when no limit is specified."""
        result = create_resource_limiter(max_memory_mb=None)
        if platform.system() == "Linux":
            assert result is None
        else:
            assert result is None

    def test_non_linux_returns_none(self):
        """Test that non-Linux systems return None."""
        # This test will pass on all platforms
        result = create_resource_limiter(max_memory_mb=1024)
        if platform.system() != "Linux":
            assert result is None

    @pytest.mark.skipif(platform.system() != "Linux", reason="Linux-specific test")
    def test_linux_returns_callable(self):
        """Test that Linux systems return a callable."""
        result = create_resource_limiter(max_memory_mb=1024)
        assert callable(result)

    @pytest.mark.skipif(platform.system() != "Linux", reason="Linux-specific test")
    def test_limiter_sets_rlimit(self):
        """Test that the limiter actually sets resource limits."""
        import subprocess

        # Test in a subprocess to avoid affecting the test process
        test_code = """
import resource
from vramcram.security.resource_limits import create_resource_limiter

limiter = create_resource_limiter(max_memory_mb=1024)
if limiter is not None:
    limiter()
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    expected_bytes = 1024 * 1024 * 1024
    assert soft == expected_bytes, f"Soft limit {soft} != {expected_bytes}"
    assert hard == expected_bytes, f"Hard limit {hard} != {expected_bytes}"
    print("OK")
"""
        result = subprocess.run(
            ["uv", "run", "python", "-c", test_code],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Test failed: {result.stderr}"
        assert "OK" in result.stdout
