from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class FeatureVector(BaseModel):
    symbol: str = Field(..., description="Cryptocurrency pair (e.g. BTC-USDT)")
    z_score_price: float = Field(..., ge=-100, le=100, description="Price z-score")
    z_score_log_return: float = Field(..., ge=-100, le=100, description="Log-return z-score")
    z_score_volume: float = Field(..., ge=-100, le=100, description="Volume z-score")
    rolling_price_std: float = Field(..., ge=0, description="Rolling price standard deviation")
    rolling_volume_std: float = Field(..., ge=0, description="Rolling volume standard deviation")


class PredictionResult(BaseModel):
    symbol: str
    anomaly_score: float
    is_anomaly: bool


class PredictionHistoryItem(BaseModel):
    symbol: str
    z_score_price: float
    z_score_log_return: float
    z_score_volume: float
    rolling_price_std: float
    rolling_volume_std: float
    anomaly_score: float
    is_anomaly: bool


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


# --- New models ---

class ServiceStatus(BaseModel):
    name: str
    status: str
    response_time_ms: float
    details: Optional[str] = None


class SystemStatusResponse(BaseModel):
    services: List[ServiceStatus]
    timestamp: str


class SymbolStats(BaseModel):
    count: int
    anomalies: int
    anomaly_rate: float
    avg_score: float


class ScorePercentiles(BaseModel):
    p25: float
    p50: float
    p75: float
    p95: float
    p99: float


class FeatureStat(BaseModel):
    mean: float
    std: float
    min: float
    max: float


class StatsResponse(BaseModel):
    total_predictions: int
    total_anomalies: int
    anomaly_rate: float
    avg_score: float
    per_symbol: Dict[str, SymbolStats]
    score_percentiles: Optional[ScorePercentiles] = None
    feature_stats: Optional[Dict[str, FeatureStat]] = None


class ModelInfoResponse(BaseModel):
    loaded: bool
    model_type: Optional[str] = None
    n_estimators: Optional[int] = None
    contamination: Optional[float] = None
    max_samples: Optional[str] = None
    feature_names: Optional[List[str]] = None
    scaler_means: Optional[List[float]] = None
    scaler_stds: Optional[List[float]] = None
    model_file_size_kb: Optional[float] = None
    model_file_modified: Optional[str] = None
