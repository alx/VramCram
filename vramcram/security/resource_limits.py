"""Resource limiting utilities for subprocess execution."""

import platform
from collections.abc import Callable


def create_resource_limiter(
    max_memory_mb: int | None = None,
) -> Callable[[], None] | None:
    """Create preexec_fn for subprocess resource limits (Linux only).

    Args:
        max_memory_mb: Maximum memory in megabytes, or None for no limit

    Returns:
        Callable to use as preexec_fn in subprocess, or None if not on Linux
    """
    if platform.system() != "Linux":
        return None

    if max_memory_mb is None:
        return None

    def set_limits() -> None:
        """Set resource limits for the subprocess."""
        import resource

        max_bytes = max_memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (max_bytes, max_bytes))

    return set_limits
