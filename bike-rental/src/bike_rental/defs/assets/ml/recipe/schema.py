"""Typed step configs for the config-driven dataset builder.

A recipe is an ordered list of steps. Steps split into two buckets by leakage:
- STATELESS (`select`, `cyclic`): run in the builder BEFORE the split.
- STATEFUL (`scale`, `one_hot`): only DECLARED here; executed by the model AFTER
  the split (fit on train), so test statistics never leak in.

Dagster's config layer can't resolve `list[A | B]`, so a step is one flat model
tagged by `kind` with each step's params as optional fields; per-kind required
fields are enforced by the validator. The builder / model dispatch on `kind`.
"""

from __future__ import annotations

import dagster as dg
from pydantic import model_validator

# kind -> the field that kind requires
_REQUIRED_FIELD = {
    "select": "columns",   # stateless: keep these columns
    "cyclic": "periods",   # stateless: sin/cos encode, drop raw source
    "scale": "columns",    # stateful: StandardScaler, fit on train (in model)
    "one_hot": "columns",  # stateful: OneHotEncoder, fit on train (in model)
}

# Who executes a step is decided by its bucket (see module docstring).
_STATELESS_KINDS = {"select", "cyclic"}
_STATEFUL_KINDS = {"scale", "one_hot"}
assert _STATELESS_KINDS | _STATEFUL_KINDS == set(_REQUIRED_FIELD)


class DatasetStep(dg.Config):
    """One feature step; `kind` selects the op, the rest are its params."""

    kind: str
    periods: dict[str, int] | None = None   # kind="cyclic"
    columns: list[str] | None = None        # kind in {"select", "scale", "one_hot"}

    @model_validator(mode="after")
    def _check_required(self) -> DatasetStep:
        if self.kind not in _REQUIRED_FIELD:
            raise ValueError(f"unknown step kind: {self.kind!r}")
        field = _REQUIRED_FIELD[self.kind]
        if getattr(self, field) is None:
            raise ValueError(f"step kind '{self.kind}' requires '{field}'")
        return self


class DatasetConfig(dg.Config):
    """Target plus the ordered feature-engineering recipe (stateless + stateful)."""

    # TODO target list ?
    target: str = "total_rentals"
    steps: list[DatasetStep] = []
