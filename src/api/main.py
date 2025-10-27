"""FastAPI application for price predictions"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import settings
from src.models import MarketLSTM
from src.features import FeatureEngine
from src.database import init_db

# Configure logging
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="EVE Price Prediction API",
    description="Machine learning API for predicting EVE Online market prices",
    version="0.1.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
model: Optional[MarketLSTM] = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Pydantic models
class PredictionRequest(BaseModel):
    """Request model for price prediction"""

    type_id: int = Field(..., description="Item type ID", ge=1)
    region_id: int = Field(default=10000002, description="Region ID (default: The Forge)")
    horizon_hours: int = Field(default=24, ge=1, le=168, description="Prediction horizon in hours")


class PredictionResponse(BaseModel):
    """Response model for price prediction"""

    type_id: int
    region_id: int
    current_price: Optional[float] = None
    predicted_price: float
    horizon_hours: int
    confidence: Optional[float] = None
    timestamp: datetime
    model_version: str = "0.1.0"


class HealthResponse(BaseModel):
    """Health check response"""

    status: str
    model_loaded: bool
    timestamp: datetime
    device: str


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global model

    logger.info("Starting up EVE Price Prediction API...")

    # Initialize database
    try:
        await init_db()
        logger.info("Database initialized")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")

    # Load model
    model_path = settings.model_path
    if Path(model_path).exists():
        try:
            checkpoint = torch.load(model_path, map_location=device)

            model = MarketLSTM(
                input_size=checkpoint["input_size"],
                hidden_size=checkpoint["hidden_size"],
                num_layers=checkpoint["num_layers"],
                dropout=checkpoint["dropout"],
            ).to(device)

            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()

            logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            model = None
    else:
        logger.warning(f"Model not found at {model_path}")
        model = None


@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "message": "EVE Price Prediction API",
        "version": "0.1.0",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model is not None else "degraded",
        model_loaded=model is not None,
        timestamp=datetime.utcnow(),
        device=str(device),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_price(request: PredictionRequest):
    """
    Predict future price for an item

    Args:
        request: PredictionRequest with type_id, region_id, horizon_hours

    Returns:
        PredictionResponse with predicted price
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train a model first.",
        )

    try:
        # Get feature vector
        feature_engine = FeatureEngine()
        features = await feature_engine.get_feature_vector(
            type_id=request.type_id,
            region_id=request.region_id,
        )

        if features is None:
            raise HTTPException(
                status_code=404,
                detail=f"No features found for type_id={request.type_id}",
            )

        # Get current price
        current_price = features.get("average") or features.get("mid_price")

        # Prepare input tensor (simplified - needs proper sequence preparation)
        # In production, this would fetch historical sequences
        feature_values = list(features.values())
        input_tensor = torch.FloatTensor([feature_values]).unsqueeze(0).to(device)

        # Make prediction
        with torch.no_grad():
            prediction = model(input_tensor)
            predicted_price = float(prediction.cpu().numpy()[0][0])

        # Calculate confidence (simplified)
        confidence = 0.85  # Placeholder

        return PredictionResponse(
            type_id=request.type_id,
            region_id=request.region_id,
            current_price=current_price,
            predicted_price=predicted_price,
            horizon_hours=request.horizon_hours,
            confidence=confidence,
            timestamp=datetime.utcnow(),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/items")
async def list_tracked_items():
    """List all tracked items"""
    from sqlalchemy import select
    from src.database import get_session, Item

    async with get_session() as session:
        result = await session.execute(select(Item))
        items = result.scalars().all()

        return {
            "items": [
                {
                    "type_id": item.type_id,
                    "name": item.name,
                    "liquidity_tier": item.liquidity_tier,
                }
                for item in items
            ]
        }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
