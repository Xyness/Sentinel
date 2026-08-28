import logging
import os

import joblib
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from training.load_dataset import load_features
    from training.preprocess import preprocess
except ImportError:
    from load_dataset import load_features
    from preprocess import preprocess

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH = os.environ.get("MODEL_PATH", "models/isolation_forest.joblib")

# Training starts as soon as MIN_PARQUET_FILES windows exist, which can be three
# rows. A stratified split needs two of each class to put one on either side, and
# a classification report over a handful of rows describes the split rather than
# the model. Below this, everything is trained on and nothing is held out.
MIN_ROWS_FOR_HOLDOUT = 40


def train():
    logger.info("Loading features...")
    df = load_features()
    X, y = preprocess(df)

    logger.info(f"Dataset size: {len(X)} samples, {X.shape[1]} features")
    has_labels = y is not None

    if has_labels:
        logger.info(f"Anomaly ratio: {y.mean():.4f}")

    rarest = int(y.value_counts().min()) if has_labels else 0
    can_hold_out = has_labels and len(X) >= MIN_ROWS_FOR_HOLDOUT and rarest >= 2

    if can_hold_out:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    else:
        # Say which of the three reasons it was. A run that quietly skipped its
        # own evaluation looks identical to one that passed it.
        if not has_labels:
            logger.info("No labels available, training in fully unsupervised mode")
        elif rarest < 2:
            logger.info(
                f"Only {rarest} sample(s) in the rarest class: nothing to hold out, "
                f"training on all {len(X)} rows"
            )
        else:
            logger.info(
                f"Only {len(X)} rows, under the {MIN_ROWS_FOR_HOLDOUT} a holdout needs "
                f"to mean anything: training on all of them"
            )
        X_train, X_test, y_test = X, None, None

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = IsolationForest(
        n_estimators=200,
        max_samples="auto",
        contamination=0.01,
        random_state=42
    )

    model.fit(X_train_scaled)

    # Evaluate only where something was actually held back
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
        y_pred_test = (model.predict(X_test_scaled) == -1).astype(int)
        logger.info("Test set evaluation:")
        # A forest that flagged nothing in the holdout has a precision of zero,
        # not an undefined one worth a warning halfway through the report.
        logger.info("\n" + classification_report(y_test, y_pred_test, zero_division=0))
    else:
        # Show score distribution for unsupervised mode
        scores = model.decision_function(X_train_scaled)
        n_detected = (model.predict(X_train_scaled) == -1).sum()
        logger.info(
            f"Unsupervised training complete: "
            f"detected {n_detected}/{len(X_train)} anomalies ({n_detected/len(X_train)*100:.2f}%)"
        )
        logger.info(
            f"Score distribution: mean: {scores.mean():.4f}, "
            f"std: {scores.std():.4f}, min: {scores.min():.4f}, max: {scores.max():.4f}"
        )

    os.makedirs(os.path.dirname(MODEL_PATH) or ".", exist_ok=True)

    joblib.dump(
        {"model": model, "scaler": scaler},
        MODEL_PATH
    )

    logger.info(f"Model saved to {MODEL_PATH}")

    if len(X_train) < MIN_ROWS_FOR_HOLDOUT:
        logger.warning(
            f"Fitted on {len(X_train)} rows. The model will score, but the scores are "
            f"not worth much until the pipeline has been running longer."
        )


if __name__ == "__main__":
    train()
