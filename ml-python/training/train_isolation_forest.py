import os
import logging
import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

try:
    from training.load_dataset import load_features
    from training.preprocess import preprocess
except ImportError:
    from load_dataset import load_features
    from preprocess import preprocess

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH = os.environ.get("MODEL_PATH", "models/isolation_forest.joblib")


def train():
    logger.info("Loading features...")
    df = load_features()
    X, y = preprocess(df)

    logger.info(f"Dataset size: {len(X)} samples, {X.shape[1]} features")
    has_labels = y is not None

    if has_labels:
        # Supervised evaluation mode: split train/test
        logger.info(f"Anomaly ratio: {y.mean():.4f}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    else:
        # Unsupervised mode (real data): use all data for training
        logger.info("No labels available — training in fully unsupervised mode")
        X_train = X
        X_test = None

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = IsolationForest(
        n_estimators=200,
        max_samples="auto",
        contamination=0.01,
        random_state=42
    )

    model.fit(X_train_scaled)

    # Evaluate if labels are available
    if has_labels and X_test is not None:
        X_test_scaled = scaler.transform(X_test)
        y_pred_test = (model.predict(X_test_scaled) == -1).astype(int)
        logger.info("Test set evaluation:")
        logger.info("\n" + classification_report(y_test, y_pred_test))
    else:
        # Show score distribution for unsupervised mode
        scores = model.decision_function(X_train_scaled)
        n_detected = (model.predict(X_train_scaled) == -1).sum()
        logger.info(
            f"Unsupervised training complete — "
            f"detected {n_detected}/{len(X_train)} anomalies ({n_detected/len(X_train)*100:.2f}%)"
        )
        logger.info(
            f"Score distribution — mean: {scores.mean():.4f}, "
            f"std: {scores.std():.4f}, min: {scores.min():.4f}, max: {scores.max():.4f}"
        )

    os.makedirs(os.path.dirname(MODEL_PATH) or ".", exist_ok=True)

    joblib.dump(
        {"model": model, "scaler": scaler},
        MODEL_PATH
    )

    logger.info(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train()
