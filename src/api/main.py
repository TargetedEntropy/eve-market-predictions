"""FastAPI application for price predictions"""

import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List, Optional

import torch
from fastapi import FastAPI, HTTPException, Response, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from src.config import settings
from src.constants import item_name
from src.database import init_db
from src.models.artifacts import available_models, load_metadata
from src.models.pipeline import (
    InsufficientData,
    ModelNotTrained,
    generate_predictions,
)
from src.utils.timeutils import utcnow

# Configure logging
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

API_VERSION = "0.2.0"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set during startup so /health can report why the service is degraded
_startup_errors: List[str] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown (replaces the deprecated @app.on_event hooks)."""
    logger.info("Starting up EVE Price Prediction API...")
    _startup_errors.clear()

    try:
        await init_db()
        logger.info("Database initialized")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        _startup_errors.append(f"database: {e}")

    models = available_models()
    if models:
        logger.info(f"{len(models)} trained model(s) available: {models}")
    else:
        logger.warning(
            f"No trained models found in {settings.model_dir}. "
            "Run scripts/train_model.py to create some."
        )

    yield

    logger.info("Shutting down EVE Price Prediction API")


app = FastAPI(
    title="EVE Price Prediction API",
    description="Machine learning API for predicting EVE Online market prices",
    version=API_VERSION,
    lifespan=lifespan,
)

# CORS middleware. Wildcard origins combined with credentials is rejected by
# the settings validator, so this can never be the foot-gun default again.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.api_cors_allow_credentials,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# Pydantic models
class PredictionRequest(BaseModel):
    """Request model for price prediction"""

    type_id: int = Field(..., description="Item type ID", ge=1)
    region_id: Optional[int] = Field(
        default=None, description="Region ID (default: first configured region)"
    )
    horizon_days: int = Field(
        default=1,
        ge=1,
        le=90,
        description=(
            "Days ahead to forecast. The model predicts one day at a time; "
            "longer horizons are produced autoregressively and compound error."
        ),
    )


class DailyPrediction(BaseModel):
    day: int
    date: str
    predicted_price: float


class PredictionResponse(BaseModel):
    """Response model for price prediction"""

    # "model_version" is a legitimate field name here, not a pydantic namespace clash
    model_config = ConfigDict(protected_namespaces=())

    type_id: int
    item_name: str
    region_id: int
    current_price: Optional[float] = None
    predicted_price: float
    horizon_days: int
    #: 1 - validation MAPE of the serving model, not a calibrated probability.
    confidence: Optional[float] = None
    timestamp: datetime
    model_version: str
    daily: List[DailyPrediction]


class ModelInfo(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    type_id: int
    item_name: str
    region_id: int
    model_version: str
    trained_at: str
    lookback_window: int
    feature_columns: List[str]
    metrics: dict


class HealthResponse(BaseModel):
    """Health check response"""

    status: str
    models_loaded: int
    timestamp: datetime
    device: str
    version: str
    errors: List[str] = []


@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "message": "EVE Price Prediction API",
        "version": API_VERSION,
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check(response: Response):
    """Health check.

    Returns 503 when no model is servable, so orchestrator probes fail on a
    non-functional API instead of passing on a "degraded" 200.
    """
    models = available_models()
    healthy = bool(models) and not _startup_errors

    if not healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return HealthResponse(
        status="healthy" if healthy else "degraded",
        models_loaded=len(models),
        timestamp=utcnow(),
        device=str(device),
        version=API_VERSION,
        errors=list(_startup_errors),
    )


@app.get("/models", response_model=List[ModelInfo])
async def list_models():
    """List the trained models this instance can serve"""
    infos = []
    for type_id, region_id in available_models():
        metadata = load_metadata(type_id, region_id)
        if metadata is None:
            continue
        infos.append(
            ModelInfo(
                type_id=type_id,
                item_name=item_name(type_id),
                region_id=region_id,
                model_version=metadata.model_version,
                trained_at=metadata.trained_at,
                lookback_window=metadata.lookback_window,
                feature_columns=metadata.feature_columns,
                metrics=metadata.metrics,
            )
        )
    return infos


@app.post("/predict", response_model=PredictionResponse)
async def predict_price(request: PredictionRequest):
    """
    Predict future price for an item.

    Uses the checkpoint trained for this exact ``(type_id, region_id)`` and its
    persisted scaler: the last ``lookback_window`` daily prices are scaled the
    same way they were in training, run through the model, and the output is
    inverse-transformed back to ISK.
    """
    region_id = request.region_id or settings.region_ids[0]

    try:
        result = await generate_predictions(
            type_id=request.type_id,
            region_id=region_id,
            horizon_days=request.horizon_days,
            store=settings.api_store_predictions,
        )
    except ModelNotTrained as e:
        raise HTTPException(status_code=503, detail=str(e))
    except InsufficientData as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        # Never leak internals to the caller; the detail goes to the log.
        logger.exception(f"Prediction failed for type_id={request.type_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

    return PredictionResponse(
        type_id=result["type_id"],
        item_name=result["item_name"],
        region_id=result["region_id"],
        current_price=result["current_price"],
        predicted_price=result["predicted_price"],
        horizon_days=result["horizon_days"],
        confidence=result["confidence"],
        timestamp=result["prediction_time"],
        model_version=result["model_version"],
        daily=[DailyPrediction(**row) for row in result["predictions"]],
    )


@app.get("/items")
async def list_tracked_items():
    """List all tracked items"""
    from sqlalchemy import select

    from src.database import Item, get_session

    async with get_session() as session:
        result = await session.execute(select(Item))
        items = result.scalars().all()

        return {
            "items": [
                {
                    "type_id": item.type_id,
                    "name": item.name,
                    "category": item.category,
                    "liquidity_tier": item.liquidity_tier,
                }
                for item in items
            ]
        }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
