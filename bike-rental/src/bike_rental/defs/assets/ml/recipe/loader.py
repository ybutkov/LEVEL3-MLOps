"""Load dataset recipes from YAML, with built-in fallbacks and validation.

A recipe is `{target, steps}`: the target column plus an ordered list of dataset
steps. Recipes live in `config/recipes.yaml` at the project root (alongside
base.yaml). If that file is absent, we fall back to the built-in
`_FALLBACK_RECIPES` below. If the file is
*present but invalid* (bad YAML, missing recipe, missing/empty 'target', missing
'steps', unknown kind, missing required field), we raise a clear
`RecipeConfigError` instead of leaking a raw IO / YAML / pydantic error — so the
failure reads as a config problem.
"""

from __future__ import annotations

import yaml
from pydantic import ValidationError

from bike_rental.config import CONFIG_DIR
from bike_rental.defs.assets.ml.recipe.schema import DatasetStep

_RECIPES_PATH = CONFIG_DIR / "recipes.yaml"

# Used ONLY when the YAML file is absent. Keep in sync with config/recipes.yaml.
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


class RecipeConfigError(Exception):
    """A recipe file is present but structurally invalid."""


def _read_recipes_file() -> dict | None:
    """Parse the YAML file; return its dict, or None if the file is absent."""
    if not _RECIPES_PATH.exists():
        return None
    try:
        return yaml.safe_load(_RECIPES_PATH.read_text()) or {}
    except yaml.YAMLError as e:
        raise RecipeConfigError(f"{_RECIPES_PATH.name} is not valid YAML: {e}") from e


def _validate_steps(name: str, steps: object) -> None:
    """Check `steps` is a non-empty list of valid DatasetStep dicts."""
    if not isinstance(steps, list) or not steps:
        raise RecipeConfigError(f"recipe '{name}': 'steps' must be a non-empty list")
    try:
        for s in steps:
            DatasetStep(**s)  # runs the per-kind validator
    except (ValidationError, TypeError, ValueError) as e:
        raise RecipeConfigError(f"recipe '{name}' has an invalid step: {e}") from e


def load_recipe(name: str) -> dict:
    """Return the named recipe as ``{"target": str, "steps": [dict, ...]}``, validated.

    Parameters
    ----------
    name : str
        Recipe name (e.g. ``"linear"`` or ``"tree"``).

    Returns
    -------
    dict
        ``{"target": str, "steps": list of dict}``.

    Raises
    ------
    RecipeConfigError
        If the file is present but the recipe is missing, or its ``target`` /
        ``steps`` are missing or invalid. (File absent → built-in fallback.)
    """
    data = _read_recipes_file()
    if data is None:
        return _FALLBACK_RECIPES[name]
    if name not in data:
        raise RecipeConfigError(f"recipe '{name}' not found in {_RECIPES_PATH.name}")
    recipe = data.get(name) or {}

    target = recipe.get("target")
    if not isinstance(target, str) or not target:
        raise RecipeConfigError(f"recipe '{name}': 'target' must be a non-empty string")

    steps = recipe.get("steps")
    if steps is None:
        raise RecipeConfigError(f"recipe '{name}': missing 'steps' key")
    _validate_steps(name, steps)

    return {"target": target, "steps": steps}


# Global split fractions (top-level `split` key). One setting for all models so
# their validation metrics stay comparable. train + val == 1 means no test set.
_DEFAULT_SPLIT = {"train_frac": 0.70, "val_frac": 0.15}


def load_split() -> dict:
    """Return the global split fractions ``{"train_frac", "val_frac"}``, validated.

    Returns
    -------
    dict
        ``{"train_frac": float, "val_frac": float}``; falls back to defaults if
        the ``split`` key is absent.

    Raises
    ------
    RecipeConfigError
        Unless ``0 < train_frac < 1``, ``0 < val_frac`` and
        ``train_frac + val_frac <= 1`` (== 1 means a 2-way split, empty test set).
    """
    data = _read_recipes_file()
    split = (data or {}).get("split") or _DEFAULT_SPLIT
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
