import logging
import os
import socket
import time
from collections import deque
from datetime import datetime, timezone
from typing import Optional, List
from urllib.request import urlopen
from urllib.error import URLError

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from schemas import (
    FeatureVector, PredictionResult, PredictionHistoryItem, HealthResponse,
    ServiceStatus, SystemStatusResponse, StatsResponse, SymbolStats,
    ScorePercentiles, FeatureStat, ModelInfoResponse,
)
from model_loader import AnomalyModel, MODEL_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="CryptoSentinel - Anomaly Detection API",
    description="Real-time anomaly detection on crypto markets",
    version="3.0"
)

model = AnomalyModel()

# In-memory ring buffer for prediction history
prediction_history: deque = deque(maxlen=500)


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
        features.z_score_log_return,
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

    # Store in ring buffer
    prediction_history.append({
        "symbol": features.symbol,
        "z_score_price": features.z_score_price,
        "z_score_log_return": features.z_score_log_return,
        "z_score_volume": features.z_score_volume,
        "rolling_price_std": features.rolling_price_std,
        "rolling_volume_std": features.rolling_volume_std,
        "anomaly_score": float(score),
        "is_anomaly": bool(is_anomaly)
    })

    return {
        "symbol": features.symbol,
        "anomaly_score": float(score),
        "is_anomaly": bool(is_anomaly)
    }


@app.get("/latest-predictions", response_model=List[PredictionHistoryItem])
def latest_predictions(
    limit: int = Query(default=100, ge=1, le=500),
    symbol: Optional[str] = Query(default=None)
):
    items = list(prediction_history)
    if symbol:
        items = [i for i in items if i["symbol"] == symbol]
    return items[-limit:]


# ── Helper functions for service checks ──────────────────────────

def _check_http(url: str, timeout: float = 3.0) -> ServiceStatus:
    name = url.split("//")[1].split("/")[0].split(":")[0].capitalize()
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


def _check_zookeeper(host: str = "zookeeper", port: int = 2181,
                     timeout: float = 2.0) -> ServiceStatus:
    start = time.time()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        s.connect((host, port))
        s.sendall(b"ruok")
        resp = s.recv(4)
        elapsed = (time.time() - start) * 1000
        s.close()
        ok = resp == b"imok"
        return ServiceStatus(
            name="Zookeeper", status="online" if ok else "degraded",
            response_time_ms=round(elapsed, 1),
            details=None if ok else f"Unexpected response: {resp!r}",
        )
    except Exception as e:
        elapsed = (time.time() - start) * 1000
        return ServiceStatus(name="Zookeeper", status="offline",
                             response_time_ms=round(elapsed, 1),
                             details=str(e)[:200])


# ── New endpoints ────────────────────────────────────────────────

@app.get("/system-status", response_model=SystemStatusResponse)
def system_status():
    services = [
        ServiceStatus(name="API", status="online", response_time_ms=0.0),
        _check_http("http://spark:4040/api/v1/applications", timeout=3.0),
        _check_tcp("kafka", 9092, "Kafka", timeout=2.0),
        _check_zookeeper("zookeeper", 2181, timeout=2.0),
    ]
    # Fix Spark name (helper derives from host)
    services[1].name = "Spark"
    return SystemStatusResponse(
        services=services,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/stats", response_model=StatsResponse)
def stats():
    items = list(prediction_history)
    total = len(items)
    if total == 0:
        return StatsResponse(
            total_predictions=0, total_anomalies=0, anomaly_rate=0.0,
            avg_score=0.0, per_symbol={},
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
    feature_cols = [
        "z_score_price", "z_score_log_return", "z_score_volume",
        "rolling_price_std", "rolling_volume_std",
    ]
    feature_stats = {}
    for col in feature_cols:
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
    )


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    model.ensure_loaded()
    if not model.loaded:
        return ModelInfoResponse(loaded=False)

    m = model.model
    s = model.scaler
    feature_names = ["z_score_price", "z_score_log_return", "z_score_volume",
                     "rolling_price_std", "rolling_volume_std"]

    info = ModelInfoResponse(
        loaded=True,
        model_type=type(m).__name__,
        n_estimators=getattr(m, "n_estimators", None),
        contamination=float(getattr(m, "contamination", 0)),
        max_samples=str(getattr(m, "max_samples", "auto")),
        feature_names=feature_names,
        scaler_means=[round(float(v), 6) for v in s.mean_] if hasattr(s, "mean_") else None,
        scaler_stds=[round(float(v), 6) for v in s.scale_] if hasattr(s, "scale_") else None,
    )

    if os.path.exists(MODEL_PATH):
        stat = os.stat(MODEL_PATH)
        info.model_file_size_kb = round(stat.st_size / 1024, 2)
        info.model_file_modified = datetime.fromtimestamp(
            stat.st_mtime, tz=timezone.utc
        ).isoformat()

    return info
