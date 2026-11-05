"""Configuration: `configs/base.yaml` merged with one stage file and optional overrides."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return `base` updated with `override`; nested mappings merge, everything else replaces."""
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_config(stage: str | None = None, overrides: dict[str, Any] | None = None) -> dict:
    """Load `base.yaml`, then `<stage>.yaml`, then `overrides`, later values winning."""
    config = _read_yaml(CONFIG_DIR / "base.yaml")
    if stage is not None:
        config = deep_merge(config, _read_yaml(CONFIG_DIR / f"{stage}.yaml"))
    if overrides:
        config = deep_merge(config, overrides)
    return config


def resolve_path(config: dict[str, Any], key: str) -> Path:
    """Resolve a `paths` entry, treating relative paths as relative to the project root."""
    path = Path(config["paths"][key])
    return path if path.is_absolute() else PROJECT_ROOT / path


def save_tuned(config: dict[str, Any], name: str, params: dict[str, Any]) -> Path:
    """Write selected hyperparameters to `<paths.tuned>/<name>.yaml`."""
    directory = resolve_path(config, "tuned")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.yaml"
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(params, handle, sort_keys=True)
    return path


def load_tuned(config: dict[str, Any], name: str) -> dict[str, Any]:
    """Read selected hyperparameters, failing clearly if the tuning stage has not been run."""
    path = resolve_path(config, "tuned") / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Run the tuning stages first (make stage1 stage2)."
        )
    return _read_yaml(path)
