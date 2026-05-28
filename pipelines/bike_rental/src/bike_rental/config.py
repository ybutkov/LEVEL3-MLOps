"""Typed pipeline configuration loaded from per-stand YAML files."""

import os
from copy import deepcopy
from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]  # .../pipelines/bike_rental
CONFIG_DIR = ROOT / "config"


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` into `base`; override wins at the leaves."""
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(result.get(key), dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


class Layout(BaseModel):
    """Env-invariant data layout under the data root (from base.yaml)."""

    source: str
    processed: str
    quarantine: str


class AppConfig(BaseModel):
    """Typed pipeline config: per-stand roots + shared layout, resolved to paths.

    Stand is chosen by DAGSTER_DEPLOYMENT (default 'local'): base.yaml supplies
    the layout, {stand}.yaml supplies the roots. Relative paths are anchored to
    the pipeline root so resolution does not depend on the process CWD.
    """

    data_root: str
    dagster_dir: str
    layout: Layout

    @classmethod
    def load(cls) -> Self:
        """Load and merge base + per-stand config for the active deployment."""
        env = os.getenv("DAGSTER_DEPLOYMENT", "local")
        env_path = CONFIG_DIR / f"{env}.yaml"
        if not env_path.exists():
            raise FileNotFoundError(f"No config for DAGSTER_DEPLOYMENT={env}: {env_path}")
        base = yaml.safe_load((CONFIG_DIR / "base.yaml").read_text()) or {}
        env_cfg = yaml.safe_load(env_path.read_text()) or {}
        return cls(**_deep_merge(base, env_cfg))

    def _path(self, *parts: str) -> str:
        return str(ROOT / Path(*parts))

    @property
    def source_dir(self) -> str:
        """Absolute path to the raw source data directory."""
        return self._path(self.data_root, self.layout.source)

    @property
    def processed_dir(self) -> str:
        """Absolute path to the processed output directory."""
        return self._path(self.data_root, self.layout.processed)

    @property
    def quarantine_dir(self) -> str:
        """Absolute path to the quarantine output directory."""
        return self._path(self.data_root, self.layout.quarantine)

    @property
    def dagster_storage_dir(self) -> str:
        """Absolute path to Dagster's local storage directory."""
        return self._path(self.dagster_dir)
