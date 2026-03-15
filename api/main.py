import logging
from fastapi import FastAPI, HTTPException
from schemas import FeatureVector, PredictionResult, HealthResponse
from model_loader import AnomalyModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="CryptoAnom - Anomaly Detection API",
    description="Real-time anomaly detection on crypto markets",
    version="1.0"
)

model = AnomalyModel()


@app.get("/health", response_model=HealthResponse)
def health():
    # Try to load model on each health check if not yet loaded
    model.ensure_loaded()
    return {
        "status": "ok" if model.loaded else "waiting_for_model",
        "model_loaded": model.loaded
    }


@app.post("/predict", response_model=PredictionResult)
def predict(features: FeatureVector):
    # Try loading model if it wasn't available at startup
    model.ensure_loaded()

    if not model.loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not yet available. Training may still be in progress."
        )

    feature_array = [
        features.z_score_price,
        features.z_score_volume,
        features.rolling_price_std,
        features.rolling_volume_std
    ]

    try:
        score, is_anomaly = model.predict(feature_array)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    logger.info(
        f"Prediction for {features.symbol}: "
        f"score={score:.4f}, anomaly={is_anomaly}"
    )

    return {
        "symbol": features.symbol,
        "anomaly_score": float(score),
        "is_anomaly": bool(is_anomaly)
    }
