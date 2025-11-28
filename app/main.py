"""
FastAPI application for clinical assertion classification
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import time

from app.schemas import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
)
from app.model import get_model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup"""
    logger.info("Loading model during startup...")
    try:
        get_model()  # Load model into cache
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model during startup: {e}")
        raise
    yield


# Initialize FastAPI app
app = FastAPI(
    title="Clinical Assertion Classification API",
    description="API for classifying assertion status (PRESENT, ABSENT, CONDITIONAL) in clinical text",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {"message": "Clinical Assertion Classification API", "docs": "/docs", "health": "/health"}


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    try:
        model = get_model()
        return HealthResponse(status="healthy", model_loaded=model.is_loaded, model_name=model.MODEL_NAME)
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(status="unhealthy", model_loaded=False, model_name="N/A")


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    """
    Predict assertion status for a single clinical sentence

    Args:
        request: PredictionRequest with a single sentence

    Returns:
        PredictionResponse with label and confidence score
    """
    try:
        start_time = time.time()

        # Get model and make prediction
        model = get_model()
        result = model.predict(request.sentence)

        elapsed_time = (time.time() - start_time) * 1000
        logger.info(f"Prediction completed in {elapsed_time:.2f}ms")

        return PredictionResponse(**result)

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict assertion status for multiple clinical sentences

    Args:
        request: BatchPredictionRequest with multiple sentences

    Returns:
        BatchPredictionResponse with list of predictions
    """
    try:
        start_time = time.time()

        # Get model and make batch prediction
        model = get_model()
        results = model.predict_batch(request.sentences)

        elapsed_time = (time.time() - start_time) * 1000
        logger.info(f"Batch prediction for {len(request.sentences)} sentences completed in {elapsed_time:.2f}ms")

        predictions = [PredictionResponse(**result) for result in results]
        return BatchPredictionResponse(predictions=predictions)

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
