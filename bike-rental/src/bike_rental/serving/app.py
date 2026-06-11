"""FastAPI service exposing the production bike-rental demand model.

A separate service from the Dagster pipeline: it loads the model behind the
``production`` registry alias at startup and predicts hourly demand for a given
time and weather. The model is cached and re-downloaded only when the alias moves
to a new version (checked cheaply per request), so promoting a new model is just
an alias move — no ``/reload`` or restart. Calendar/holiday/trend features are
derived server-side to match the pipeline, so a caller sends only a timestamp and
weather.

Run: ``uvicorn bike_rental.serving.app:app``.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from bike_rental.config import AppConfig
from bike_rental.serving.champion import ChampionCache
from bike_rental.serving.features import build_feature_row
from bike_rental.serving.holidays import HolidayRepository
from bike_rental.serving.schemas import HealthResponse, PredictionRequest, PredictionResponse

SERVING_ALIAS = "production"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Set up the holiday lookup and the champion cache (primed once) at startup."""
    config = AppConfig.load()
    app.state.config = config
    app.state.holidays = HolidayRepository()
    app.state.champion = ChampionCache(
        config.mlflow.tracking_uri, config.mlflow.registered_model, SERVING_ALIAS
    )
    app.state.champion.get()  # load once now; later calls only reload on version change
    yield


app = FastAPI(title="Bike Rental Demand API", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Report liveness and the served model's version + data commit."""
    model = app.state.champion.get()
    return HealthResponse(
        status="ok",
        model_name=app.state.config.mlflow.registered_model,
        model_version=model.version,
        data_commit=model.data_commit,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Predict hourly bike-rental demand for the requested time and weather."""
    model = app.state.champion.get()
    features = build_feature_row(request, app.state.holidays, model.feature_columns)
    try:
        prediction = float(model.pipeline.predict(features)[0])
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {error}") from error

    return PredictionResponse(
        timestamp=request.timestamp.isoformat(),
        predicted_rentals=int(round(prediction)),
        model_version=model.version,
        data_commit=model.data_commit,
    )
