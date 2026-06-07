from __future__ import annotations
from functools import cached_property
from pathlib import Path

import yaml
import dagster as dg

from bike_rental.config import CONFIG_DIR


_FALLBACK_RECIPES: dict[str, dict] = {
    "linear": {
        "target": "total_rentals",
        "steps": [
            {"kind": "select", "columns": [
                "datetime_hourly", "hour_of_day", "month", "day_of_week",
                "conditions", "temperature_c", "perceived_temperature_c",
                "humidity", "windspeed_kmh", "days_since_launch",
                "is_weekend", "is_holiday", "total_rentals",
            ]},
            {"kind": "cyclic", "periods": {"hour_of_day": 24, "month": 12, "day_of_week": 7}},
            {"kind": "scale", "columns": [
                "conditions", "temperature_c", "perceived_temperature_c",
                "humidity", "windspeed_kmh", "days_since_launch",
            ]},
        ],
    },
    "tree": {
        "target": "total_rentals",
        "steps": [
            {"kind": "select", "columns": [
                "datetime_hourly", "month", "hour_of_day", "day_of_week", "is_weekend",
                "is_holiday", "conditions", "temperature_c", "perceived_temperature_c",
                "humidity", "windspeed_kmh", "days_since_launch", "total_rentals",
            ]},
        ],
    },
}

_DEFAULT_SPLIT = {"train_frac": 0.70, "val_frac": 0.15}

class RecipeConfigError(Exception):
        """A recipe file is present but structurally invalid."""

class RecipeConfig(dg.ConfigurableResource):
    config_recipe_dir: str = str(CONFIG_DIR)
    config_recipe_file: str = "recipes.yaml"

    @cached_property
    def _recipes(self) -> dict:
        path = Path(self.config_recipe_dir) / self.config_recipe_file
        # path = self.config_recipe_dir / self.config_recipe_file
        if not path.exists():
            return _FALLBACK_RECIPES
        try:
            return yaml.safe_load(path.read_text()) or {}
        except yaml.YAMLError as e:
            raise RecipeConfigError(
                f"{self.config_recipe_file} is not valid YAML: {e}"
            ) from e

    def get_recipe(self, name: str) -> dict:
        if name not in self._recipes:
            raise RecipeConfigError(f"recipe '{name}' not found in {self.config_recipe_file}")
        return self._recipes.get(name) or {}
    
    def get_split_settings(self):
        split = (self._recipes or {}).get("split") or _DEFAULT_SPLIT
        train, val = split.get("train_frac"), split.get("val_frac")
        valid = (
            isinstance(train, (int, float))
            and isinstance(val, (int, float))
            and 0 < train < 1
            and 0 < val
            and round(train + val, 6) <= 1
        )
        if not valid:
            raise RecipeConfigError(
                "split: need 0<train_frac<1, 0<val_frac, train_frac+val_frac<=1; "
                f"got {split}"
            )
        return {"train_frac": float(train), "val_frac": float(val)}