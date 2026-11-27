
import json
import logging
from datetime import datetime, timezone
import math
import os
import time
from datetime import date
from functools import lru_cache

import mlflow
import mlflow.sklearn
import pandas as pd

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field


# ==================================================
# LOGGING CONFIGURATION
# ==================================================

# ==================================================
# LOGGING CONFIGURATION
# ==================================================

class JSONFormatter(logging.Formatter):

    def format(self, record):

        log_data = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage()
        }

        if record.exc_info:
            log_data["exception"] = self.formatException(
                record.exc_info
            )

        return json.dumps(log_data)


handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.handlers.clear()
logger.addHandler(handler)
logger.propagate = False


# ==================================================
# FASTAPI APPLICATION
# ==================================================

app = FastAPI(
    title="MP172 PG Rent Prediction API",
    description="Predict monthly PG room rent in Jaipur.",
    version="1.0.0"
)


# ==================================================
# REQUEST SCHEMA
# ==================================================

class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    distance_to_campus_km: float = Field(ge=0)
    room_area_sqft: float
    sharing: int
    meals_per_day: int
    building_age_yr: float
    wifi_mbps: float = Field(le=550)
    furnishing: str
    locality: str


# ==================================================
# RESPONSE SCHEMA
# ==================================================

class PredictionResponse(BaseModel):
    prediction: float


# ==================================================
# LOAD REGISTERED MODEL
# ==================================================

@lru_cache(maxsize=1)
def load_model():

    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        "http://127.0.0.1:5001"
    )

    model_uri = os.getenv(
        "MODEL_URI",
        "models:/RentalPricePredictor@champion"
    )

    mlflow.set_tracking_uri(tracking_uri)

    logger.info("Loading registered model: %s", model_uri)

    model = mlflow.sklearn.load_model(model_uri)

    logger.info("ML model loaded successfully")

    return model


# ==================================================
# REQUEST LOGGING MIDDLEWARE
# ==================================================

@app.middleware("http")
async def request_logging(request: Request, call_next):

    start_time = time.perf_counter()

    logger.info(
        "Request started | method=%s | path=%s",
        request.method,
        request.url.path
    )

    try:
        response = await call_next(request)

        duration = (time.perf_counter() - start_time) * 1000

        logger.info(
            "Request completed | status=%s | duration_ms=%.2f",
            response.status_code,
            duration
        )

        return response

    except Exception:
        logger.exception("Unhandled API request error")
        raise


# ==================================================
# HEALTH ENDPOINT
# ==================================================

@app.get("/health")
def health():

    return {"status": "ok"}


# ==================================================
# PREDICTION ENDPOINT
# ==================================================

@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(request: PredictionRequest):

    # Load the registered model
    try:
        model = load_model()

    except Exception as exc:
        logger.exception("Failed to load prediction model")

        raise HTTPException(
            status_code=503,
            detail="Prediction model is currently unavailable"
        ) from exc

    try:
        # Convert request into a dictionary
        input_data = request.model_dump()

        # Generate date features internally
        current_date = date.today()
        weekday = current_date.weekday()

        input_data.update({

            # Champion v1 date features
            "month": current_date.month,
            "day_of_week": weekday,
            "day_of_month": current_date.day,
            "is_weekend": int(weekday >= 5),

            # Candidate v2 date features
            "date_ordinal": current_date.toordinal(),

            "month_sin": math.sin(
                2 * math.pi * current_date.month / 12
            ),

            "month_cos": math.cos(
                2 * math.pi * current_date.month / 12
            ),

            "weekday_sin": math.sin(
                2 * math.pi * weekday / 7
            ),

            "weekday_cos": math.cos(
                2 * math.pi * weekday / 7
            )
        })

        # Create input DataFrame
        features = pd.DataFrame([input_data])

        # Select features expected by the loaded model
        expected_features = getattr(
            model,
            "feature_names_in_",
            None
        )

        if expected_features is None:
            raise ValueError(
                "Model does not expose expected feature names"
            )

        missing_features = (
            set(expected_features) - set(features.columns)
        )

        if missing_features:
            raise ValueError(
                f"Missing model features: {missing_features}"
            )

        features = features.loc[
            :, list(expected_features)
        ]

        # Generate prediction
        prediction = float(
            model.predict(features)[0]
        )

        if not math.isfinite(prediction):
            raise ValueError(
                "Model returned an invalid prediction"
            )

        logger.info(
            "Prediction successful | predicted_rent=%.2f",
            prediction
        )

        return PredictionResponse(
            prediction=prediction
        )

    except Exception as exc:
        logger.exception("Prediction failed")

        raise HTTPException(
            status_code=500,
            detail="Prediction failed"
        ) from exc
