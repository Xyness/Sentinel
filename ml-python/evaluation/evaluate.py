import os
import logging
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import joblib

try:
    from training.load_dataset import load_features
    from training.preprocess import preprocess
except ImportError:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))
    from load_dataset import load_features
    from preprocess import preprocess

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH = os.environ.get("MODEL_PATH", "models/isolation_forest.joblib")


def evaluate():
    df = load_features()
    X, y_true = preprocess(df)

    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    scaler = bundle["scaler"]

    has_labels = y_true is not None

    if has_labels:
        # Use the same split as training for consistent evaluation
        _, X_test, _, y_test = train_test_split(
            X, y_true, test_size=0.2, random_state=42, stratify=y_true
        )
        X_scaled = scaler.transform(X_test)
        predictions = model.predict(X_scaled)

        # IsolationForest: -1 = anomaly, 1 = normal
        y_pred = (predictions == -1).astype(int)

        logger.info("Evaluation on held-out test set (20%):")
        logger.info("\n" + classification_report(y_test, y_pred))

        cm = confusion_matrix(y_test, y_pred)
        logger.info(f"Confusion matrix:\n{cm}")

        decision_scores = model.decision_function(X_scaled)
    else:
        # Unsupervised mode: no labels, just show score distribution
        X_scaled = scaler.transform(X)
        decision_scores = model.decision_function(X_scaled)
        predictions = model.predict(X_scaled)

        n_anomalies = (predictions == -1).sum()
        logger.info(
            f"Unsupervised evaluation on {len(X)} samples — "
            f"detected {n_anomalies} anomalies ({n_anomalies/len(X)*100:.2f}%)"
        )

    logger.info(
        f"Decision score stats — mean: {decision_scores.mean():.4f}, "
        f"std: {decision_scores.std():.4f}, "
        f"min: {decision_scores.min():.4f}, "
        f"max: {decision_scores.max():.4f}"
    )


if __name__ == "__main__":
    evaluate()
