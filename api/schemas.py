from pydantic import BaseModel

class FeatureVector(BaseModel):
    symbol: str
    z_score_price: float
    z_score_volume: float
    rolling_price_std: float
    rolling_volume_std: float


class PredictionResult(BaseModel):
    symbol: str
    anomaly_score: float
    is_anomaly: bool
