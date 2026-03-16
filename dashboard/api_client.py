import os
import logging
from typing import Optional, List
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")


def check_health() -> dict:
    """Check API health status."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise


def get_prediction(features: dict) -> dict:
    """Send features to the ML API and return the prediction."""
    try:
        response = requests.post(f"{API_BASE_URL}/predict", json=features, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.ConnectionError:
        logger.error(f"Cannot connect to API at {API_BASE_URL}")
        raise ConnectionError(f"API unavailable at {API_BASE_URL}. Is the API service running?")
    except requests.Timeout:
        logger.error(f"API request timed out ({API_BASE_URL})")
        raise TimeoutError("API request timed out. Please try again.")
    except requests.HTTPError as e:
        logger.error(f"API returned error: {e.response.status_code} - {e.response.text}")
        raise


def get_latest_predictions(limit: int = 100, symbol: Optional[str] = None) -> List[dict]:
    """Fetch latest predictions from the API."""
    params = {"limit": limit}
    if symbol:
        params["symbol"] = symbol
    try:
        response = requests.get(
            f"{API_BASE_URL}/latest-predictions", params=params, timeout=10
        )
        response.raise_for_status()
        return response.json()
    except requests.ConnectionError:
        logger.error(f"Cannot connect to API at {API_BASE_URL}")
        raise ConnectionError(f"API unavailable at {API_BASE_URL}.")
    except requests.Timeout:
        logger.error("API request timed out")
        raise TimeoutError("API request timed out.")
    except requests.HTTPError as e:
        logger.error(f"API returned error: {e.response.status_code}")
        raise


def get_system_status() -> dict:
    """Fetch system-wide service status."""
    try:
        response = requests.get(f"{API_BASE_URL}/system-status", timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"System status check failed: {e}")
        return {
            "services": [
                {"name": "API", "status": "offline", "response_time_ms": 0, "details": str(e)}
            ],
            "timestamp": "",
        }


def get_stats() -> dict:
    """Fetch aggregated prediction stats."""
    try:
        response = requests.get(f"{API_BASE_URL}/stats", timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Stats fetch failed: {e}")
        return {
            "total_predictions": 0, "total_anomalies": 0,
            "anomaly_rate": 0.0, "avg_score": 0.0, "per_symbol": {},
        }


def get_model_info() -> dict:
    """Fetch loaded model information."""
    try:
        response = requests.get(f"{API_BASE_URL}/model-info", timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Model info fetch failed: {e}")
        return {"loaded": False}
