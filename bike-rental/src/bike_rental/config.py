"""Typed pipeline configuration loaded from per-stand YAML files."""

import os
from copy import deepcopy
from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
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


class MlflowSettings(BaseModel):
    """MLflow tracking/registry settings; ``tracking_uri`` is stand-specific."""

    tracking_uri: str
    experiment_name: str
    registered_model: str


class LakeFSSettings(BaseModel):
    """LakeFS source settings (non-secret); credentials come from env vars."""

    host: str
    repo: str
    ref: str = "main"
    raw_prefix: str = "raw"
    ingest_branch: str = "ingest"


class AppConfig(BaseModel):
    """Typed pipeline config: per-stand roots + shared layout, resolved to paths.

    Stand is chosen by DAGSTER_DEPLOYMENT (default 'local'): base.yaml supplies
    the layout, {stand}.yaml supplies the roots. Relative paths are anchored to
    the pipeline root so resolution does not depend on the process CWD.
    """

    data_root: str
    dagster_dir: str
    layout: Layout
    mlflow: MlflowSettings
    lakefs: LakeFSSettings

    @classmethod
    def load(cls) -> Self:
        """Load and merge base + per-deployment config for the active stand.

        The deployment is chosen by the ``DAGSTER_DEPLOYMENT`` env var (default
        ``"local"``); its YAML is deep-merged over ``base.yaml``.

        Returns
        -------
        AppConfig
            The merged configuration.

        Raises
        ------
        FileNotFoundError
            If no config file exists for the active deployment.
        """
        env = os.getenv("DAGSTER_DEPLOYMENT", "local")
        env_path = CONFIG_DIR / f"{env}.yaml"
        if not env_path.exists():
            raise FileNotFoundError(f"No config for DAGSTER_DEPLOYMENT={env}: {env_path}")
        base = yaml.safe_load((CONFIG_DIR / "base.yaml").read_text()) or {}
        env_cfg = yaml.safe_load(env_path.read_text()) or {}
        return cls(**_deep_merge(base, env_cfg))

    def _path(self, *parts: str) -> str:
        """Return an absolute path under the project root for the given parts."""
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
