"""Experiment-level configuration shared across model-training assets."""

import dagster as dg

from bike_rental.defs import schemas


class ExperimentConfig(dg.ConfigurableResource):
    """Run-level knobs shared by every model asset, set once in the launchpad.

    ``features`` is the active feature set. Removing one here drops it from every
    model's dataset (and therefore its preprocessing) without editing recipes or
    assets — the recipe defines *how* each column is transformed, this defines
    *which* columns are active.
    """

    features: list[str] = list(schemas.HOURLY_FEATURES)
