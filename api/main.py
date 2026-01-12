from fastapi import FastAPI
from schemas import FeatureVector, PredictionResult
from model_loader import AnomalyModel

app = FastAPI(
    title="CryptoAnom – Anomaly Detection API",
    description="Real-time anomaly detection on crypto markets",
    version="1.0"
)

model = AnomalyModel()

@app.post("/predict", response_model=PredictionResult)
def predict(features: FeatureVector):

    feature_array = [
        features.z_score_price,
        features.z_score_volume,
        features.rolling_price_std,
        features.rolling_volume_std
    ]

    score, is_anomaly = model.predict(feature_array)

    return {
        "symbol": features.symbol,
        "anomaly_score": float(score),
        "is_anomaly": bool(is_anomaly)
    }
