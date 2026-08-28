import logging
import os
import socket
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from urllib.request import urlopen

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from model_loader import MODEL_PATH, AnomalyModel
from schemas import (
    FeatureBatch,
    FeatureStat,
    FeatureVector,
    HealthResponse,
    ModelInfoResponse,
    PredictionHistoryItem,
    PredictionResult,
    ScorePercentiles,
    ServiceStatus,
    StatsResponse,
    SymbolStats,
    SystemStatusResponse,
)
from store import FEATURE_COLUMNS, PredictionStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


app = FastAPI(
    title="Sentinel - Anomaly Detection API",
    description="Real-time anomaly detection on crypto markets",
    version="1.1.0",
)

model = AnomalyModel()

# Predictions live in SQLite rather than in a deque, so a restart no longer
# throws away everything the pipeline has flagged. With no PREDICTIONS_DB set
# the database is in memory and behaves the way the deque did.
store = PredictionStore()

# How many of the most recent rows /stats aggregates. The whole table would be
# an honest answer too, but a week of history makes a percentile that reacts to
# nothing, and this endpoint exists to describe the recent past.
STATS_ROWS = int(os.environ.get("STATS_ROWS", "5000"))


@app.get("/health", response_model=HealthResponse)
def health():
    # Try to load model on each health check if not yet loaded
    model.ensure_loaded()
    return {
        "status": "ok" if model.loaded else "waiting_for_model",
        "model_loaded": model.loaded
    }


def _vector(features: FeatureVector) -> list:
    return [getattr(features, name) for name in FEATURE_COLUMNS]


def _record(features: FeatureVector, score, is_anomaly) -> dict:
    return store.append(
        features.symbol,
        {name: getattr(features, name) for name in FEATURE_COLUMNS},
        score,
        is_anomaly,
    )


def _require_model():
    model.ensure_loaded()
    if not model.loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not yet available. Training may still be in progress."
        )


@app.post("/predict", response_model=PredictionResult)
def predict(features: FeatureVector):
    # Try loading model if it wasn't available at startup
    _require_model()

    try:
        score, is_anomaly = model.predict(_vector(features))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}") from e

    logger.info(
        f"Prediction for {features.symbol}: "
        f"score={score:.4f}, anomaly={is_anomaly}"
    )

    _record(features, score, is_anomaly)

    return {
        "symbol": features.symbol,
        "anomaly_score": float(score),
        "is_anomaly": bool(is_anomaly)
    }


@app.post("/predict/batch", response_model=list[PredictionResult])
def predict_batch(batch: FeatureBatch):
    """Score a whole Parquet file's worth of rows in one call.

    The scorer used to POST one row at a time, which is one HTTP round trip and
    one single-row scaler call per window. The forest is happy to take the lot
    at once and the results come back in the order they were sent.
    """
    _require_model()

    try:
        scored = model.predict_many([_vector(vector) for vector in batch.vectors])
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}") from e

    results = []
    flagged = 0
    for features, (score, is_anomaly) in zip(batch.vectors, scored, strict=True):
        _record(features, score, is_anomaly)
        flagged += bool(is_anomaly)
        results.append({
            "symbol": features.symbol,
            "anomaly_score": float(score),
            "is_anomaly": bool(is_anomaly),
        })

    logger.info(f"Scored {len(results)} vector(s), {flagged} flagged")
    return results


@app.get("/latest-predictions", response_model=list[PredictionHistoryItem])
def latest_predictions(
    limit: int = Query(default=100, ge=1, le=500),
    symbol: str | None = Query(default=None),
    after: int | None = Query(default=None, ge=0,
                                 description="only entries newer than this id")
):
    return store.latest(limit=limit, symbol=symbol, after=after)


# Service reachability helpers, used by /system-status

def _check_http(name: str, url: str, timeout: float = 3.0) -> ServiceStatus:
    start = time.time()
    try:
        resp = urlopen(url, timeout=timeout)
        elapsed = (time.time() - start) * 1000
        resp.read()
        return ServiceStatus(name=name, status="online", response_time_ms=round(elapsed, 1))
    except Exception as e:
        elapsed = (time.time() - start) * 1000
        return ServiceStatus(name=name, status="offline", response_time_ms=round(elapsed, 1),
                             details=str(e)[:200])


def _check_tcp(host: str, port: int, name: str, timeout: float = 2.0) -> ServiceStatus:
    start = time.time()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        s.connect((host, port))
        elapsed = (time.time() - start) * 1000
        s.close()
        return ServiceStatus(name=name, status="online", response_time_ms=round(elapsed, 1))
    except Exception as e:
        elapsed = (time.time() - start) * 1000
        return ServiceStatus(name=name, status="offline", response_time_ms=round(elapsed, 1),
                             details=str(e)[:200])


@app.get("/system-status", response_model=SystemStatusResponse)
def system_status():
    # Run the probes at the same time. One after another, a stack that is
    # entirely down took the sum of every timeout to say so, and this is the
    # endpoint you call precisely when things are not answering.
    probes = [
        # The Spark UI usually answers in under 100ms, but it is served off the
        # driver, so a micro-batch or a GC pause can hold the thread well past a
        # second. Three was tight enough to report a healthy Spark as offline.
        lambda: _check_http("Spark", "http://spark:4040/api/v1/applications", timeout=8.0),
        lambda: _check_tcp("kafka", 9092, "Kafka", timeout=2.0),
    ]

    with ThreadPoolExecutor(max_workers=len(probes)) as pool:
        checked = list(pool.map(lambda probe: probe(), probes))

    services = [ServiceStatus(name="API", status="online", response_time_ms=0.0), *checked]
    return SystemStatusResponse(
        services=services,
        timestamp=datetime.now(UTC).isoformat(),
    )


@app.get("/stats", response_model=StatsResponse)
def stats():
    items = store.recent(STATS_ROWS)
    total = len(items)
    if total == 0:
        return StatsResponse(
            total_predictions=0, total_anomalies=0, anomaly_rate=0.0,
            avg_score=0.0, per_symbol={}, stored=0,
        )

    scores = np.array([i["anomaly_score"] for i in items])
    anomalies_mask = np.array([i["is_anomaly"] for i in items])
    total_anomalies = int(anomalies_mask.sum())

    # Per-symbol breakdown
    per_symbol = {}
    symbol_groups: dict = {}
    for item in items:
        symbol_groups.setdefault(item["symbol"], []).append(item)
    for sym, group in symbol_groups.items():
        sym_scores = [g["anomaly_score"] for g in group]
        sym_anom = sum(1 for g in group if g["is_anomaly"])
        per_symbol[sym] = SymbolStats(
            count=len(group), anomalies=sym_anom,
            anomaly_rate=round(sym_anom / len(group) * 100, 2),
            avg_score=round(float(np.mean(sym_scores)), 6),
        )

    # Score percentiles
    percentiles = ScorePercentiles(
        p25=round(float(np.percentile(scores, 25)), 6),
        p50=round(float(np.percentile(scores, 50)), 6),
        p75=round(float(np.percentile(scores, 75)), 6),
        p95=round(float(np.percentile(scores, 95)), 6),
        p99=round(float(np.percentile(scores, 99)), 6),
    )

    # Feature stats
    feature_stats = {}
    for col in FEATURE_COLUMNS:
        vals = np.array([i[col] for i in items])
        feature_stats[col] = FeatureStat(
            mean=round(float(vals.mean()), 6),
            std=round(float(vals.std()), 6),
            min=round(float(vals.min()), 6),
            max=round(float(vals.max()), 6),
        )

    return StatsResponse(
        total_predictions=total,
        total_anomalies=total_anomalies,
        anomaly_rate=round(total_anomalies / total * 100, 2),
        avg_score=round(float(scores.mean()), 6),
        per_symbol=per_symbol,
        score_percentiles=percentiles,
        feature_stats=feature_stats,
        stored=store.count(),
    )


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    model.ensure_loaded()
    if not model.loaded:
        return ModelInfoResponse(loaded=False)

    m = model.model
    s = model.scaler

    info = ModelInfoResponse(
        loaded=True,
        model_type=type(m).__name__,
        n_estimators=getattr(m, "n_estimators", None),
        contamination=float(getattr(m, "contamination", 0)),
        max_samples=str(getattr(m, "max_samples", "auto")),
        # Read off the bundle rather than hardcoded here, so a model fitted on a
        # different feature set says so instead of mislabelling its own scaler.
        feature_names=list(model.features),
        scaler_means=[round(float(v), 6) for v in s.mean_] if hasattr(s, "mean_") else None,
        scaler_stds=[round(float(v), 6) for v in s.scale_] if hasattr(s, "scale_") else None,
        metrics=model.metrics,
    )

    if os.path.exists(MODEL_PATH):
        stat = os.stat(MODEL_PATH)
        info.model_file_size_kb = round(stat.st_size / 1024, 2)
        info.model_file_modified = datetime.fromtimestamp(
            stat.st_mtime, tz=UTC
        ).isoformat()

    return info
