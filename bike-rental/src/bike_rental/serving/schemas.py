"""Request/response schemas for the prediction API."""

from datetime import datetime

from pydantic import BaseModel, Field, field_validator

from bike_rental.defs.schemas import WEATHER_CONDITIONS


class PredictionRequest(BaseModel):
    """One demand query: when + weather.

    The caller supplies only a timestamp and the weather; calendar, holiday and
    trend features are derived server-side (see ``features.build_feature_row``),
    keeping the contract aligned with the business case.
    """

    timestamp: datetime
    conditions: str
    temperature_c: float = Field(ge=-50, le=60)
    perceived_temperature_c: float = Field(ge=-50, le=60)
    humidity: float = Field(ge=0, le=100)
    windspeed_kmh: float = Field(ge=0, le=300)

    @field_validator("conditions")
    @classmethod
    def _known_condition(cls, value: str) -> str:
        """Reject conditions the model was never trained on."""
        if value not in WEATHER_CONDITIONS:
            raise ValueError(f"conditions must be one of {WEATHER_CONDITIONS}, got {value!r}")
        return value


class PredictionResponse(BaseModel):
    """Predicted hourly demand plus the lineage of the model that produced it."""

    timestamp: str
    predicted_rentals: int
    model_version: str
    data_commit: str | None


class HealthResponse(BaseModel):
    """Liveness and which model version / data commit is currently served."""

    status: str
    model_name: str
    model_version: str
    data_commit: str | None
