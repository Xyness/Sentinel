import os
import logging
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
API_URL = f"{API_BASE_URL}/predict"


def get_prediction(features: dict):
    """Send features to the ML API and return the prediction."""
    try:
        response = requests.post(API_URL, json=features, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.ConnectionError:
        logger.error(f"Cannot connect to API at {API_URL}")
        raise ConnectionError(f"API unavailable at {API_URL}. Is the API service running?")
    except requests.Timeout:
        logger.error(f"API request timed out ({API_URL})")
        raise TimeoutError("API request timed out. Please try again.")
    except requests.HTTPError as e:
        logger.error(f"API returned error: {e.response.status_code} - {e.response.text}")
        raise
