from pydantic import BaseModel, ConfigDict, Field


class FeatureVector(BaseModel):
    """One window of one pair, as five dimensionless numbers.

    They are ratios rather than z-scores against the window's own deviation.
    That old shape hid the thing it was meant to measure: an anomaly inflates
    the deviation used to normalise it, so the score of a step depended on
    where in the window it landed and not on how big it was.

    Every field has a floor of zero because every one of them is a magnitude,
    and a ceiling loose enough to admit any real market and tight enough to
    catch a unit mix-up before it reaches the scaler.
    """

    symbol: str = Field(..., description="Cryptocurrency pair (e.g. BTC-USDT)")
    abs_return_max: float = Field(
        ..., ge=0, le=10,
        description="Largest absolute log return in the window")
    return_std: float = Field(
        ..., ge=0, le=10,
        description="Realised volatility: standard deviation of log returns")
    price_range_rel: float = Field(
        ..., ge=0, le=100,
        description="(high - low) / mean price over the window")
    volume_max_ratio: float = Field(
        ..., ge=0, le=10000,
        description="Largest single trade over the window's mean volume")
    volume_cv: float = Field(
        ..., ge=0, le=1000,
        description="Volume standard deviation over mean volume")


class FeatureBatch(BaseModel):
    vectors: list[FeatureVector] = Field(..., min_length=1, max_length=500)


class PredictionResult(BaseModel):
    symbol: str
    anomaly_score: float
    is_anomaly: bool


class PredictionHistoryItem(BaseModel):
    id: int
    timestamp: str
    symbol: str
    abs_return_max: float
    return_std: float
    price_range_rel: float
    volume_max_ratio: float
    volume_cv: float
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
    stored: int | None = None


class HoldoutScores(BaseModel):
    """What the training run measured on rows it had not fitted on."""

    precision: float
    recall: float
    f1: float
    support: int


class ModelMetrics(BaseModel):
    trained_at: str | None = None
    n_rows: int | None = None
    n_train: int | None = None
    n_test: int | None = None
    contamination: float | None = None
    labelled: bool | None = None
    label_rate: float | None = None
    holdout: HoldoutScores | None = None


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
    metrics: ModelMetrics | None = None
