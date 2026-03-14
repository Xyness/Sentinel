import os
import logging
import joblib
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH = os.environ.get("MODEL_PATH", "models/isolation_forest.joblib")


class AnomalyModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.loaded = False
        self._try_load()

    def _try_load(self):
        if not os.path.exists(MODEL_PATH):
            logger.warning(f"Model file not found yet: {MODEL_PATH}")
            return False

        try:
            logger.info(f"Loading model from {MODEL_PATH}")
            bundle = joblib.load(MODEL_PATH)
            self.model = bundle["model"]
            self.scaler = bundle["scaler"]
            self.loaded = True
            logger.info("Model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def ensure_loaded(self):
        """Try to load the model if not already loaded."""
        if not self.loaded:
            return self._try_load()
        return True

    def predict(self, features: list) -> tuple:
        if not self.loaded:
            raise RuntimeError("Model not loaded")

        features_array = np.array(features).reshape(1, -1)

        if np.any(np.isnan(features_array)) or np.any(np.isinf(features_array)):
            raise ValueError("Input features contain NaN or infinite values")

        X_scaled = self.scaler.transform(features_array)
        score = self.model.decision_function(X_scaled)[0]
        prediction = self.model.predict(X_scaled)[0]

        logger.debug(f"Prediction: score={score:.4f}, anomaly={prediction == -1}")
        return score, prediction == -1
