from pydantic import BaseModel, ConfigDict, Field


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
    id: int
    timestamp: str
    symbol: str
    z_score_price: float
    z_score_log_return: float
    z_score_volume: float
    rolling_price_std: float
    rolling_volume_std: float
    anomaly_score: float
    is_anomaly: bool


class HealthResponse(BaseModel):
    # `model_loaded` is about the ML model, not pydantic's. Same reason as below.
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: bool


# Payloads for /system-status, /stats and /model-info

class ServiceStatus(BaseModel):
    name: str
    status: str
    response_time_ms: float
    details: str | None = None


class SystemStatusResponse(BaseModel):
    services: list[ServiceStatus]
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
    per_symbol: dict[str, SymbolStats]
    score_percentiles: ScorePercentiles | None = None
    feature_stats: dict[str, FeatureStat] | None = None


class ModelInfoResponse(BaseModel):
    # These fields describe the ML model, not pydantic's own model API, and
    # pydantic reserves the `model_` prefix. Renaming them would change the
    # payload every client reads, so the namespace is opened instead.
    model_config = ConfigDict(protected_namespaces=())

    loaded: bool
    model_type: str | None = None
    n_estimators: int | None = None
    contamination: float | None = None
    max_samples: str | None = None
    feature_names: list[str] | None = None
    scaler_means: list[float] | None = None
    scaler_stds: list[float] | None = None
    model_file_size_kb: float | None = None
    model_file_modified: str | None = None
