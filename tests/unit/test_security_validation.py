"""Unit tests for security validation functions."""

import os
from pathlib import Path

import pytest

from vramcram.security.validation import (
    validate_binary_path,
    validate_path,
    validate_prompt,
)


class TestValidatePrompt:
    """Test prompt validation."""

    def test_valid_prompt(self):
        """Test that valid prompts pass validation."""
        prompt = "This is a valid prompt"
        result = validate_prompt(prompt, max_length=1000)
        assert result == prompt

    def test_null_byte_rejection(self):
        """Test that prompts with null bytes are rejected."""
        prompt = "This has a null byte\x00here"
        with pytest.raises(ValueError, match="null bytes"):
            validate_prompt(prompt, max_length=1000)

    def test_length_limit_enforcement(self):
        """Test that oversized prompts are rejected."""
        prompt = "x" * 1001
        with pytest.raises(ValueError, match="exceeds maximum length"):
            validate_prompt(prompt, max_length=1000)

    def test_empty_prompt_rejection(self):
        """Test that empty prompts are rejected."""
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_prompt("", max_length=1000)

    def test_whitespace_only_rejection(self):
        """Test that whitespace-only prompts are rejected."""
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_prompt("   \n\t  ", max_length=1000)

    def test_prompt_with_leading_trailing_whitespace(self):
        """Test that prompts with whitespace are preserved."""
        prompt = "  valid prompt  "
        result = validate_prompt(prompt, max_length=1000)
        assert result == prompt


class TestValidatePath:
    """Test path validation."""

    def test_valid_absolute_path(self, tmp_path):
        """Test that valid absolute paths pass validation."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")

        result = validate_path(test_file)
        assert result == test_file.resolve()

    def test_valid_relative_path(self, tmp_path):
        """Test that valid relative paths are resolved."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")

        # Change to tmp_path and use relative path
        original_cwd = Path.cwd()
        try:
            os.chdir(tmp_path)
            result = validate_path(Path("test.txt"))
            assert result == test_file.resolve()
        finally:
            os.chdir(original_cwd)

    def test_nonexistent_path_rejection(self, tmp_path):
        """Test that non-existent paths are rejected."""
        nonexistent = tmp_path / "does_not_exist.txt"
        with pytest.raises(ValueError, match="does not exist"):
            validate_path(nonexistent)

    def test_directory_traversal_rejection(self, tmp_path):
        """Test that directory traversal is detected."""
        # Create a file outside the allowed base
        outside_dir = tmp_path / "outside"
        outside_dir.mkdir()
        outside_file = outside_dir / "file.txt"
        outside_file.write_text("test")

        # Try to access it via traversal
        inside_dir = tmp_path / "inside"
        inside_dir.mkdir()

        with pytest.raises(ValueError, match="not within allowed base"):
            validate_path(outside_file, allowed_base=inside_dir)

    def test_allowed_base_enforcement(self, tmp_path):
        """Test that paths must be within allowed base."""
        allowed_dir = tmp_path / "allowed"
        allowed_dir.mkdir()
        allowed_file = allowed_dir / "file.txt"
        allowed_file.write_text("test")

        # Should pass when within allowed base
        result = validate_path(allowed_file, allowed_base=allowed_dir)
        assert result == allowed_file.resolve()

        # Should fail when outside allowed base
        outside_dir = tmp_path / "outside"
        outside_dir.mkdir()
        outside_file = outside_dir / "file.txt"
        outside_file.write_text("test")

        with pytest.raises(ValueError, match="not within allowed base"):
            validate_path(outside_file, allowed_base=allowed_dir)

    def test_symlink_within_allowed_base(self, tmp_path):
        """Test symlinks pointing within allowed base are allowed."""
        allowed_dir = tmp_path / "allowed"
        allowed_dir.mkdir()

        target = allowed_dir / "target.txt"
        target.write_text("test")

        link = allowed_dir / "link.txt"
        link.symlink_to(target)

        result = validate_path(link, allowed_base=allowed_dir)
        assert result == link.resolve()

    def test_symlink_outside_allowed_base(self, tmp_path):
        """Test symlinks pointing outside allowed base are rejected."""
        allowed_dir = tmp_path / "allowed"
        allowed_dir.mkdir()

        outside_dir = tmp_path / "outside"
        outside_dir.mkdir()
        target = outside_dir / "target.txt"
        target.write_text("test")

        link = allowed_dir / "link.txt"
        link.symlink_to(target)

        with pytest.raises(ValueError, match="not within allowed base"):
            validate_path(link, allowed_base=allowed_dir)


class TestValidateBinaryPath:
    """Test binary path validation."""

    def test_valid_executable(self, tmp_path):
        """Test that valid executables pass validation."""
        binary = tmp_path / "test_binary"
        binary.write_text("#!/bin/sh\necho test")
        binary.chmod(0o755)

        result = validate_binary_path(str(binary))
        assert result == binary.resolve()

    def test_directory_traversal_rejection(self):
        """Test that paths with .. are rejected."""
        with pytest.raises(ValueError, match=r"contains '\.\.'"):
            validate_binary_path("../../bin/sh")

    def test_nonexistent_binary_rejection(self, tmp_path):
        """Test that non-existent binaries are rejected."""
        nonexistent = tmp_path / "does_not_exist"
        with pytest.raises(ValueError, match="does not exist"):
            validate_binary_path(str(nonexistent))

    def test_non_executable_rejection(self, tmp_path):
        """Test that non-executable files are rejected."""
        non_exec = tmp_path / "not_executable.txt"
        non_exec.write_text("test")
        non_exec.chmod(0o644)

        with pytest.raises(ValueError, match="not executable"):
            validate_binary_path(str(non_exec))

    def test_directory_rejection(self, tmp_path):
        """Test that directories are rejected."""
        directory = tmp_path / "dir"
        directory.mkdir()
        directory.chmod(0o755)

        with pytest.raises(ValueError, match="not a regular file"):
            validate_binary_path(str(directory))

    def test_executable_in_path(self):
        """Test validating executables in PATH."""
        # /bin/sh should exist on most Unix systems
        result = validate_binary_path("/bin/sh")
        # Result may be resolved symlink, so just check it's a valid path
        assert result.exists()
        assert result.is_file()
        assert os.access(result, os.X_OK)
