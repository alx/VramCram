"""Configuration loader for VramCram."""

import os
from pathlib import Path

import yaml

from vramcram.config.models import VramCramConfig


def load_config(config_path: str | Path | None = None) -> VramCramConfig:
    """Load VramCram configuration from YAML file.

    Args:
        config_path: Path to config file. If None, uses VRAMCRAM_CONFIG env var.

    Returns:
        Validated VramCramConfig instance.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If config is invalid.
    """
    if config_path is None:
        config_path = os.environ.get("VRAMCRAM_CONFIG", "config.yaml")

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        config_data = yaml.safe_load(f)

    if config_data is None:
        raise ValueError(f"Config file is empty: {config_path}")

    try:
        return VramCramConfig(**config_data)
    except Exception as e:
        raise ValueError(f"Invalid configuration: {e}") from e
