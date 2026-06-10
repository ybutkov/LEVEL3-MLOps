from __future__ import annotations

from pydantic import BaseModel, Field, model_validator, ValidationError

from bike_rental.defs.assets.ml.recipes.recipe_config import RecipeConfigError

_REQUIRED_FIELD = {
    "select": "columns",
    "cyclic": "periods",
    "scale": "columns",
    "one_hot": "columns",
}


class DatasetStep(BaseModel):

    kind: str
    periods: dict[str, int] | None = None
    columns: list[str] | None = None

    @model_validator(mode="after")
    def _check_required(self) -> DatasetStep:
        """Validate the step kind and that its required field is present."""
        if self.kind not in _REQUIRED_FIELD:
            raise ValueError(f"unknown step kind: {self.kind!r}")
        field = _REQUIRED_FIELD[self.kind]
        if getattr(self, field) is None:
            raise ValueError(f"step kind '{self.kind}' requires '{field}'")
        return self


class DatasetConfig(BaseModel):

    # TODO target list ?
    target: str = Field("total_rentals", min_length=1)
    steps: list[DatasetStep] = Field(min_length=1)

    @classmethod
    def from_recipe(cls, recipe_config, name: str) -> DatasetConfig:
        recipe = recipe_config.get_recipe(name)
        try:
            return cls(**recipe)
        except ValidationError as e:
            raise RecipeConfigError(f"recipe '{name}': {e}") from e
