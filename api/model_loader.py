import logging
import os

import joblib
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH = os.environ.get("MODEL_PATH", "models/isolation_forest.joblib")

# What a bundle written before the model carried its own feature list contains.
DEFAULT_FEATURES = [
    "abs_return_max",
    "return_std",
    "price_range_rel",
    "volume_max_ratio",
    "volume_cv",
]


class AnomalyModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.features = list(DEFAULT_FEATURES)
        self.metrics = None
        self.loaded = False
        self._mtime = None
        self._try_load()

    def _try_load(self):
        if not os.path.exists(MODEL_PATH):
            logger.warning(f"Model file not found yet: {MODEL_PATH}")
            return False

        try:
            # Read the timestamp first. Taking it afterwards would record the
            # state of a file that may have been replaced mid-read, and the
            # newer model would then never look new.
            mtime = os.path.getmtime(MODEL_PATH)
            logger.info(f"Loading model from {MODEL_PATH}")
            bundle = joblib.load(MODEL_PATH)
            self.model = bundle["model"]
            self.scaler = bundle["scaler"]
            self.features = list(bundle.get("features") or DEFAULT_FEATURES)
            self.metrics = bundle.get("metrics")
            self.loaded = True
            self._mtime = mtime
            logger.info("Model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def ensure_loaded(self):
        """Load the model, and pick up a newer one when training writes it.

        Training no longer runs once and exits, so the file under MODEL_PATH
        changes while the API is up. Returning early on `loaded` alone meant
        the first model ever written was the one served for the life of the
        process, and every retrain after it went nowhere.
        """
        if not self.loaded:
            return self._try_load()

        try:
            mtime = os.path.getmtime(MODEL_PATH)
        except OSError:
            # The file went away underneath us. The model already in memory is
            # still a model, and refusing to score would help nobody.
            return True

        if mtime != self._mtime:
            logger.info("Model file changed on disk, reloading")
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
        # Plain Python types out, the same as predict_many. A numpy bool goes
        # into SQLite as a blob and comes back out as something that is not a
        # bool any more.
        return float(score), bool(prediction == -1)

    def predict_many(self, rows: list) -> list:
        """Score a whole batch in one pass through the forest.

        The scorer sends a Parquet file at a time, and a round trip plus a
        single-row scaler call per row was most of the cost of scoring them.
        """
        if not self.loaded:
            raise RuntimeError("Model not loaded")
        if not rows:
            return []

        features_array = np.array(rows, dtype=float)

        if np.any(np.isnan(features_array)) or np.any(np.isinf(features_array)):
            raise ValueError("Input features contain NaN or infinite values")

        X_scaled = self.scaler.transform(features_array)
        scores = self.model.decision_function(X_scaled)
        predictions = self.model.predict(X_scaled)
        return [(float(score), bool(prediction == -1))
                for score, prediction in zip(scores, predictions, strict=True)]
