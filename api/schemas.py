from pydantic import BaseModel, Field


class FeatureVector(BaseModel):
    symbol: str = Field(..., description="Cryptocurrency pair (e.g. BTC-USDT)")
    z_score_price: float = Field(..., ge=-100, le=100, description="Price z-score")
    z_score_volume: float = Field(..., ge=-100, le=100, description="Volume z-score")
    rolling_price_std: float = Field(..., ge=0, description="Rolling price standard deviation")
    rolling_volume_std: float = Field(..., ge=0, description="Rolling volume standard deviation")


class PredictionResult(BaseModel):
    symbol: str
    anomaly_score: float
    is_anomaly: bool


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
