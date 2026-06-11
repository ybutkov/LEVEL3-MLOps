"""Build the dataset (select step) and the preprocessor (transform steps) from a recipe."""

import dagster as dg
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from bike_rental.defs.assets.ml.guards import assert_no_target_leak
from bike_rental.defs.assets.ml.recipes.schema import DatasetConfig, DatasetStep
from bike_rental.defs.assets.ml.recipes.transformers import CyclicEncoder

TIME_KEY = "datetime_hourly"

_PREPROCESSOR_BUILDERS = {
    "cyclic":  lambda s: (CyclicEncoder(s.periods), list(s.periods)),
    "scale":   lambda s: (StandardScaler(), s.columns),
    "one_hot": lambda s: (OneHotEncoder(handle_unknown="ignore"), s.columns),
}


def _assert_columns_present(needed, available, what: str) -> None:
    """Raise dg.Failure if any column in `needed` is absent from `available`."""
    missing = sorted(set(needed) - set(available))
    if missing:
        raise dg.Failure(
            description=f"{what}: columns absent from the dataset: {missing}.",
            metadata={"missing": dg.MetadataValue.text(", ".join(missing))},
        )


def preprocessor_input_columns(config: DatasetConfig) -> set[str]:
    """Raw columns the preprocessor steps consume (must exist after select)."""
    cols: set[str] = set()
    for step in config.steps:
        build = _PREPROCESSOR_BUILDERS.get(step.kind)
        if build is not None:
            cols |= set(build(step)[1])
    return cols


def assert_recipe_columns(config: DatasetConfig, columns) -> None:
    """Fail if target or any preprocessor-input column is absent from `columns`."""
    needed = preprocessor_input_columns(config) | {config.target}
    _assert_columns_present(needed, columns, "Recipe references")


def restrict_to_features(config: DatasetConfig, features) -> DatasetConfig:
    """Restrict a recipe config to ``features`` (target and time are always kept).

    The ``select`` step is intersected with ``features ∪ {target, time}``;
    preprocessor steps (cyclic/scale/one_hot) are intersected with ``features``;
    steps left with no columns are dropped. Lets a run train on a subset of
    features without editing the recipe.

    Parameters
    ----------
    config : DatasetConfig
        Recipe config to restrict.
    features : Iterable[str]
        Active feature columns to keep.

    Returns
    -------
    DatasetConfig
        A new config referencing only the active columns.
    """
    active = set(features)
    keep = active | {config.target, TIME_KEY}
    new_steps: list[DatasetStep] = []
    for step in config.steps:
        if step.kind == "select":
            cols = [c for c in step.columns if c in keep]
            new_steps.append(DatasetStep(kind="select", columns=cols))
        elif step.kind == "cyclic":
            periods = {k: v for k, v in step.periods.items() if k in active}
            if periods:
                new_steps.append(DatasetStep(kind="cyclic", periods=periods))
        else:  # scale / one_hot
            cols = [c for c in step.columns if c in active]
            if cols:
                new_steps.append(DatasetStep(kind=step.kind, columns=cols))
    return DatasetConfig(target=config.target, steps=new_steps)


def build_dataset(df: pd.DataFrame, config: DatasetConfig) -> pd.DataFrame:
    """Apply the recipe's `select` steps; leakage-check the result."""
    for step in config.steps:
        if step.kind != "select":
            continue
        _assert_columns_present(step.columns, df.columns, "select step")
        df = df[step.columns].copy()

    assert_recipe_columns(config, df.columns)
    features = [c for c in df.columns if c not in (config.target, TIME_KEY)]
    assert_no_target_leak(features, config.target)
    return df


def build_preprocessor(config: DatasetConfig) -> ColumnTransformer:
    """Assemble the recipe's preprocessor steps into one unfitted ColumnTransformer."""
    transformers = []
    for i, step in enumerate(config.steps):
        if step.kind == "select":
            continue
        build = _PREPROCESSOR_BUILDERS.get(step.kind)
        if build is None:  # non-preprocessor kind (select) — handled in build_da
            continue
        estimator, columns = build(step)
        transformers.append((f"{step.kind}_{i}", estimator, columns))
    return ColumnTransformer(transformers, remainder="passthrough")
