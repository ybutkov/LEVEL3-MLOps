"""Apply a dataset recipe — both sides of the train/val/test split.

A recipe (`DatasetConfig.steps`) mixes two buckets, separated by leakage:

- STATELESS (`select`, `cyclic`) -> `build_dataset`: run eagerly on the DataFrame
  BEFORE the split (they learn nothing from the data).
- STATEFUL (`scale`, `one_hot`) -> `build_preprocessor`: assembled into an
  unfitted sklearn ColumnTransformer that the model fits AFTER the split, so test
  statistics never leak.

Validation checks references against the REAL columns present at each point (no
reconstruction of post-transform names): each stateless step's inputs must exist
when it runs, and `target` + stateful columns must exist in the assembled dataset.
"""

import dagster as dg
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from bike_rental.defs.assets.ml.recipe.features import add_cyclic_features
from bike_rental.defs.assets.ml.recipe.schema import (
    _STATEFUL_KINDS,
    _STATELESS_KINDS,
    DatasetConfig,
    DatasetStep,
)
from bike_rental.defs.assets.ml.training.guards import assert_no_target_leak

# --- stateless: executed by the builder, before the split ------------------


def _apply_cyclic(df: pd.DataFrame, step: DatasetStep) -> pd.DataFrame:
    """Apply the ``cyclic`` step: sin/cos-encode the configured columns."""
    return add_cyclic_features(df, step.periods)


def _apply_select(df: pd.DataFrame, step: DatasetStep) -> pd.DataFrame:
    """Apply the ``select`` step: keep only the configured columns."""
    return df[step.columns].copy()


STATELESS_TRANSFORMS = {
    "cyclic": _apply_cyclic,
    "select": _apply_select,
}
assert set(STATELESS_TRANSFORMS) == _STATELESS_KINDS, (
    "STATELESS_TRANSFORMS must cover exactly the stateless kinds"
)


def _step_inputs(step: DatasetStep) -> list[str]:
    """Columns a stateless step reads from the frame it runs on."""
    return list(step.periods) if step.kind == "cyclic" else step.columns


# --- stateful: assembled here, fit by the model, after the split -----------


def _make_scale(step: DatasetStep) -> StandardScaler:
    """Build the transformer for a ``scale`` step (a StandardScaler)."""
    return StandardScaler()


def _make_one_hot(step: DatasetStep) -> OneHotEncoder:
    """Build the transformer for a ``one_hot`` step (an OneHotEncoder)."""
    return OneHotEncoder(handle_unknown="ignore")


STATEFUL_TRANSFORMERS = {
    "scale": _make_scale,
    "one_hot": _make_one_hot,
}
assert set(STATEFUL_TRANSFORMERS) == _STATEFUL_KINDS, (
    "STATEFUL_TRANSFORMERS must cover exactly the stateful kinds"
)


def stateful_columns(config: DatasetConfig) -> set[str]:
    """Return the columns referenced by the recipe's stateful steps.

    Parameters
    ----------
    config : DatasetConfig
        The recipe.

    Returns
    -------
    set of str
        Columns named by ``scale`` / ``one_hot`` steps; these must exist in the
        assembled dataset.
    """
    return {
        col
        for step in config.steps
        if step.kind in _STATEFUL_KINDS
        for col in step.columns
    }


def assert_recipe_columns(config: DatasetConfig, columns) -> None:
    """Fail if ``target`` or any stateful-step column is absent from ``columns``.

    Checked against the real columns of the assembled dataset (no name guessing).

    Parameters
    ----------
    config : DatasetConfig
        The recipe (target + steps).
    columns : iterable of str
        The actual columns of the assembled dataset.

    Raises
    ------
    dagster.Failure
        If any required column is missing.
    """
    needed = stateful_columns(config) | {config.target}
    missing = sorted(needed - set(columns))
    if missing:
        raise dg.Failure(
            description=(
                f"Recipe references columns absent from the dataset: {missing}. "
                "Add them to the select step (or fix the target)."
            ),
            metadata={"missing": dg.MetadataValue.text(", ".join(missing))},
        )


# --- entry points ----------------------------------------------------------


def build_dataset(df: pd.DataFrame, config: DatasetConfig) -> pd.DataFrame:
    """Apply the recipe's stateless steps in order; skip the stateful ones.

    Each stateless step's inputs are checked against the columns present when it
    runs; after assembly, ``target`` plus stateful columns must exist and the
    no-target-leak rule is enforced — so a leaky dataset is never written.
    Stateful steps are only declared here (running them before the split would
    leak); ``build_preprocessor`` handles them inside the model.

    Parameters
    ----------
    df : pandas.DataFrame
        Source frame (e.g. the ``hourly_total`` base dataset).
    config : DatasetConfig
        The recipe to apply.

    Returns
    -------
    pandas.DataFrame
        The assembled, leakage-checked feature table.

    Raises
    ------
    dagster.Failure
        If a step's inputs are missing, a required column is absent, or a target
        component would leak into the features.
    """
    for step in config.steps:
        if step.kind in _STATEFUL_KINDS:
            continue
        missing = sorted(set(_step_inputs(step)) - set(df.columns))
        if missing:
            raise dg.Failure(
                description=(
                    f"Recipe step '{step.kind}' needs columns not available at that "
                    f"point: {missing}."
                ),
                metadata={"missing": dg.MetadataValue.text(", ".join(missing))},
            )
        df = STATELESS_TRANSFORMS[step.kind](df, step)

    assert_recipe_columns(config, df.columns)
    features = [c for c in df.columns if c not in (config.target, "datetime_hourly")]
    assert_no_target_leak(features, config.target)
    return df


def build_preprocessor(config: DatasetConfig) -> ColumnTransformer:
    """Assemble an unfitted ColumnTransformer from the recipe's stateful steps.

    Each stateful step becomes one transformer over its ``columns``; all other
    columns pass through. With no stateful steps this is a pure passthrough. The
    model fits it on the train fold.

    Parameters
    ----------
    config : DatasetConfig
        The recipe whose stateful steps (``scale`` / ``one_hot``) are assembled.

    Returns
    -------
    sklearn.compose.ColumnTransformer
        Unfitted transformer with ``remainder="passthrough"``.
    """
    transformers = [
        (f"{step.kind}_{i}", STATEFUL_TRANSFORMERS[step.kind](step), step.columns)
        for i, step in enumerate(config.steps)
        if step.kind in _STATEFUL_KINDS
    ]
    return ColumnTransformer(transformers, remainder="passthrough")
