"""
Configuration loader module
Handles loading and validating YAML configuration
"""


import yaml
import os
from pathlib import Path
from typing import Dict, Any


class Config:
    """Configuration wrapper class"""

    def __init__(self, config_dict: Dict[str, Any]):
        self._config = config_dict

    def __getattr__(self, key):
        if key in self._config:
            value = self._config[key]
            if isinstance(value, dict):
                return Config(value)
            return value
        raise AttributeError(f"Config has no attribute '{key}'")

    def __getitem__(self, key):
        return self._config[key]

    def get(self, key, default=None):
        return self._config.get(key, default)

    def to_dict(self) -> Dict:
        return self._config


def load_config(config_path: str = None) -> Config:
    """
    Load configuration from YAML file

    Args:
        config_path: Path to config.yaml file. If None, looks in current directory

    Returns:
        Config object with nested attribute access

    Example:
        >>> config = load_config('config.yaml')
        >>> print(config.data.sequence_length)
        8
        >>> print(config.model.gru.hidden_units)
        [128, 32]
    """
    if config_path is None:
        # Try to find config.yaml in current directory or parent
        current_dir = Path(__file__).parent
        config_path = current_dir / "config.yaml"

        if not config_path.exists():
            config_path = current_dir.parent / "config.yaml"

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)

    # Validate required sections
    required_sections = ['data', 'model', 'training']
    for section in required_sections:
        if section not in config_dict:
            raise ValueError(f"Config missing required section: '{section}'")

    return Config(config_dict)


def resolve_path(path: str, base_dir: str = None) -> Path:
    """
    Resolve relative paths relative to config file location

    Args:
        path: File path (can be relative or absolute)
        base_dir: Base directory for resolving relative paths

    Returns:
        Resolved absolute Path
    """
    path = Path(path)

    if path.is_absolute():
        return path

    if base_dir is None:
        base_dir = Path(__file__).parent

    return (Path(base_dir) / path).resolve()
