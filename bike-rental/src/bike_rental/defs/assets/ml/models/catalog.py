"""Model catalog: maps a ``model_type`` to its estimator class and recipe.

One place to answer "what builds an ``hgb`` model and on which preprocessing
recipe". Hyperparameter defaults stay in the per-model ``dg.Config`` blocks;
this catalog only knows the estimator class and the recipe pairing.
"""

from dataclasses import dataclass

from sklearn.base import BaseEstimator
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression


@dataclass(frozen=True)
class ModelSpec:
    """What builds a model and which recipe it trains on."""

    estimator_cls: type[BaseEstimator]
    recipe: str


MODEL_CATALOG: dict[str, ModelSpec] = {
    "linear": ModelSpec(LinearRegression, "linear"),
    "rf": ModelSpec(RandomForestRegressor, "tree"),
    "hgb": ModelSpec(HistGradientBoostingRegressor, "tree"),
}


def _spec(model_type: str) -> ModelSpec:
    if model_type not in MODEL_CATALOG:
        raise KeyError(f"unknown model_type {model_type!r}; have {list(MODEL_CATALOG)}")
    return MODEL_CATALOG[model_type]


def build_estimator(model_type: str, params: dict) -> BaseEstimator:
    """Instantiate the estimator for ``model_type`` from already-resolved params."""
    return _spec(model_type).estimator_cls(**params)


def recipe_name_for(model_type: str) -> str:
    """Preprocessing recipe paired with ``model_type``."""
    return _spec(model_type).recipe
