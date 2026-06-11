"""Recipe configuration resource: loads named preprocessing recipes from YAML."""

from __future__ import annotations

from functools import cached_property
from pathlib import Path

import dagster as dg
import yaml

from bike_rental.config import CONFIG_DIR

_FALLBACK_RECIPES: dict[str, dict] = {
    "dataset": {
        "target": "total_rentals",
        "steps": [
            {"kind": "select", "columns": [
                "datetime_hourly", "hour_of_day", "month", "day_of_week", "is_weekend",
                "is_holiday", "conditions", "temperature_c", "perceived_temperature_c",
                "humidity", "windspeed_kmh", "days_since_launch", "total_rentals",
            ]},
        ],
    },
    "linear": {
        "target": "total_rentals",
        "steps": [
            {"kind": "cyclic", "periods": {"hour_of_day": 24, "month": 12, "day_of_week": 7}},
            {"kind": "scale", "columns": [
                "conditions", "temperature_c", "perceived_temperature_c",
                "humidity", "windspeed_kmh", "days_since_launch",
            ]},
        ],
    },
    "tree": {
        "target": "total_rentals",
        "steps": [],
    },
}

_DEFAULT_SPLIT = {"train_frac": 0.70, "val_frac": 0.15}

class RecipeConfigError(Exception):
        """A recipe file is present but structurally invalid."""

class RecipeConfig(dg.ConfigurableResource):
    """Load named recipes from ``recipes.yaml`` (or a built-in fallback)."""

    config_recipe_dir: str = str(CONFIG_DIR)
    config_recipe_file: str = "recipes.yaml"

    @cached_property
    def _recipes(self) -> dict:
        path = Path(self.config_recipe_dir) / self.config_recipe_file
        if not path.exists():
            return _FALLBACK_RECIPES
        try:
            return yaml.safe_load(path.read_text()) or {}
        except yaml.YAMLError as e:
            raise RecipeConfigError(
                f"{self.config_recipe_file} is not valid YAML: {e}"
            ) from e

    def get_recipe(self, name: str) -> dict:
        """Return the recipe dict for ``name``, or raise if it isn't defined."""
        if name not in self._recipes:
            raise RecipeConfigError(f"recipe '{name}' not found in {self.config_recipe_file}")
        return self._recipes.get(name) or {}
