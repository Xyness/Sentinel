import logging
import os
from datetime import UTC, datetime

import joblib
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from training.load_dataset import load_features
    from training.preprocess import FEATURE_COLUMNS, preprocess
except ImportError:
    from load_dataset import load_features
    from preprocess import FEATURE_COLUMNS, preprocess

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH = os.environ.get("MODEL_PATH", "models/isolation_forest.joblib")

N_ESTIMATORS = int(os.environ.get("N_ESTIMATORS", "200"))

# What to flag when there are no labels to count. Unsupervised means nobody can
# tell you the rate, so this is a stated assumption rather than a measurement.
DEFAULT_CONTAMINATION = float(os.environ.get("CONTAMINATION", "0.05"))

# Where a measured rate stops being believable. Under the floor the forest has
# too few splits to isolate anything; over the ceiling the word anomaly has
# stopped meaning much and the generator wants turning down instead.
MIN_CONTAMINATION = 0.005
MAX_CONTAMINATION = 0.25

# Training starts as soon as MIN_PARQUET_FILES windows exist, which can be three
# rows. A stratified split needs two of each class to put one on either side, and
# a classification report over a handful of rows describes the split rather than
# the model. Below this, everything is trained on and nothing is held out.
MIN_ROWS_FOR_HOLDOUT = 40


def contamination_for(y):
    """What fraction of rows to flag.

    This has to track the rate the labels actually carry. It used to be pinned
    at 1 % while the generator, drawing once per event, put an anomaly in 45 %
    of one-minute windows: the forest flagged a fiftieth of what was there and
    the recall could not exceed 2 % however good the features were. Reading the
    rate off the labels means changing the generator cannot silently break the
    model.
    """
    if y is None:
        logger.info(
            f"No labels to count, using the stated contamination of {DEFAULT_CONTAMINATION:.4f}"
        )
        return DEFAULT_CONTAMINATION

    observed = float(y.mean())
    contamination = min(max(observed, MIN_CONTAMINATION), MAX_CONTAMINATION)

    if contamination != observed:
        logger.warning(
            f"Labels put the anomaly rate at {observed:.4f}, clamped to "
            f"{contamination:.4f}. Outside [{MIN_CONTAMINATION}, {MAX_CONTAMINATION}] "
            f"the generator's rate is the thing to change, not the model's."
        )
    else:
        logger.info(f"Contamination set from the labels: {contamination:.4f}")

    return contamination


def _atomic_dump(bundle, path):
    """Write beside the model, then rename over it.

    The API reloads the file when its timestamp moves, and joblib.dump on the
    live path would let it read a half-written bundle. A rename on the same
    filesystem cannot be seen half done.
    """
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    staged = f"{path}.tmp"
    joblib.dump(bundle, staged)
    os.replace(staged, path)


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

    contamination = contamination_for(y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        max_samples="auto",
        contamination=contamination,
        random_state=42
    )

    model.fit(X_train_scaled)

    metrics = {
        "trained_at": datetime.now(UTC).isoformat(),
        "n_rows": int(len(X)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)) if X_test is not None else None,
        "contamination": float(contamination),
        "labelled": bool(has_labels),
        "label_rate": float(y.mean()) if has_labels else None,
        "holdout": None,
    }

    # Evaluate only where something was actually held back
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
        y_pred_test = (model.predict(X_test_scaled) == -1).astype(int)
        logger.info("Test set evaluation:")
        # A forest that flagged nothing in the holdout has a precision of zero,
        # not an undefined one worth a warning halfway through the report.
        logger.info("\n" + classification_report(y_test, y_pred_test, zero_division=0))

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred_test, average="binary", pos_label=1, zero_division=0
        )
        # Carried in the bundle so the API can serve it and `sentinel status`
        # can print it. A model that cannot say how it scored is a model whose
        # numbers live in a log nobody kept.
        metrics["holdout"] = {
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
            "support": int(y_test.sum()),
        }
        logger.info(
            f"Holdout: precision {precision:.3f}, recall {recall:.3f}, f1 {f1:.3f} "
            f"over {int(y_test.sum())} labelled anomalies"
        )
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

    _atomic_dump(
        {"model": model, "scaler": scaler, "features": list(FEATURE_COLUMNS),
         "metrics": metrics},
        MODEL_PATH,
    )

    logger.info(f"Model saved to {MODEL_PATH}")

    if len(X_train) < MIN_ROWS_FOR_HOLDOUT:
        logger.warning(
            f"Fitted on {len(X_train)} rows. The model will score, but the scores are "
            f"not worth much until the pipeline has been running longer."
        )

    return metrics


if __name__ == "__main__":
    train()
