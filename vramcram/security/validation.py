"""Security validation functions for user inputs and file paths."""

import os
from pathlib import Path


def validate_prompt(prompt: str, max_length: int) -> str:
    """Validate user prompt for security.

    Args:
        prompt: User-provided prompt string
        max_length: Maximum allowed prompt length

    Returns:
        Validated prompt string

    Raises:
        ValueError: If prompt contains null bytes, exceeds max length, or is empty
    """
    # Check for null bytes
    if "\x00" in prompt:
        raise ValueError("Prompt contains null bytes which are not allowed")

    # Enforce maximum length
    if len(prompt) > max_length:
        raise ValueError(
            f"Prompt exceeds maximum length of {max_length} characters "
            f"(got {len(prompt)} characters)"
        )

    # Strip whitespace and reject empty prompts
    stripped = prompt.strip()
    if not stripped:
        raise ValueError("Prompt cannot be empty or whitespace-only")

    return prompt


def validate_path(path: Path, allowed_base: Path | None = None) -> Path:
    """Validate file path for security.

    Args:
        path: Path to validate
        allowed_base: Optional base directory that path must be within

    Returns:
        Resolved absolute path

    Raises:
        ValueError: If path contains directory traversal, is outside allowed base,
                   doesn't exist, or isn't readable
    """
    # Resolve to absolute path
    try:
        resolved_path = path.resolve(strict=False)
    except (OSError, RuntimeError) as e:
        raise ValueError(f"Failed to resolve path {path}: {e}")

    # Check for directory traversal patterns
    # After resolution, the path should not escape the allowed base
    if ".." in resolved_path.parts:
        raise ValueError(f"Path contains directory traversal: {path}")

    # Verify within allowed_base if provided
    allowed_base_resolved = None
    if allowed_base is not None:
        try:
            allowed_base_resolved = allowed_base.resolve(strict=False)
        except (OSError, RuntimeError) as e:
            raise ValueError(f"Failed to resolve allowed base {allowed_base}: {e}")

        try:
            resolved_path.relative_to(allowed_base_resolved)
        except ValueError:
            raise ValueError(
                f"Path {resolved_path} is not within allowed base directory "
                f"{allowed_base_resolved}"
            )

    # Verify exists and is readable
    if not resolved_path.exists():
        raise ValueError(f"Path does not exist: {resolved_path}")

    if not os.access(resolved_path, os.R_OK):
        raise ValueError(f"Path is not readable: {resolved_path}")

    # Check for symlinks pointing outside allowed areas
    if allowed_base_resolved is not None and resolved_path.is_symlink():
        link_target = resolved_path.readlink()
        if link_target.is_absolute():
            try:
                link_target.relative_to(allowed_base_resolved)
            except ValueError:
                raise ValueError(
                    f"Symlink {resolved_path} points outside allowed base: "
                    f"{link_target}"
                )

    return resolved_path


def validate_binary_path(binary_path: str) -> Path:
    """Validate binary executable path.

    Args:
        binary_path: Path to binary executable

    Returns:
        Resolved absolute path to binary

    Raises:
        ValueError: If path contains "..", doesn't exist, or isn't executable
    """
    # Convert to Path object
    path = Path(binary_path)

    # Reject paths containing ".." before resolution to prevent traversal
    if ".." in path.parts:
        raise ValueError(f"Binary path contains '..': {binary_path}")

    # Resolve to absolute path
    try:
        resolved_path = path.resolve(strict=False)
    except (OSError, RuntimeError) as e:
        raise ValueError(f"Failed to resolve binary path {binary_path}: {e}")

    # Verify exists
    if not resolved_path.exists():
        raise ValueError(f"Binary does not exist: {resolved_path}")

    # Verify is executable
    if not os.access(resolved_path, os.X_OK):
        raise ValueError(f"Path is not executable: {resolved_path}")

    # Verify is a file (not a directory)
    if not resolved_path.is_file():
        raise ValueError(f"Path is not a regular file: {resolved_path}")

    return resolved_path
