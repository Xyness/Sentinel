"""Score a trained model against the features currently on disk.

Two different numbers come out of here and they are not interchangeable.

The bundle carries the holdout the training run measured itself on, which is
the out-of-sample one and the only one worth quoting. What this script computes
on top is over every row in the feature store, including the rows the model was
fitted on, so it is in-sample and it flatters. It is still worth having: it is
how you see the score distribution move after a change without restarting the
stack.

This used to re-derive `train_test_split(random_state=42)` and call the result
"the same split as training". It was not. The feature store grows between the
training run and this one, so splitting it again draws a different line through
a different dataset, and the rows it called held out were mostly rows the model
had trained on.
"""

import logging
import os

import joblib
from sklearn.metrics import classification_report, confusion_matrix

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


def report_training_metrics(metrics):
    if not metrics:
        logger.info("This bundle predates carrying its own metrics, nothing to report")
        return

    logger.info(
        f"Trained {metrics.get('trained_at')} on {metrics.get('n_train')} rows "
        f"at contamination {metrics.get('contamination')}"
    )

    holdout = metrics.get("holdout")
    if holdout:
        logger.info(
            f"Held-out scores (out of sample): precision {holdout['precision']}, "
            f"recall {holdout['recall']}, f1 {holdout['f1']} over {holdout['support']} "
            f"labelled anomalies"
        )
    else:
        logger.info("No holdout was taken, so there are no out-of-sample scores")


def evaluate():
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    scaler = bundle["scaler"]

    logger.info("-- what the training run measured --")
    report_training_metrics(bundle.get("metrics"))

    df = load_features()
    X, y_true = preprocess(df)

    X_scaled = scaler.transform(X)
    decision_scores = model.decision_function(X_scaled)
    predictions = model.predict(X_scaled)
    y_pred = (predictions == -1).astype(int)

    logger.info(f"-- over all {len(X)} rows on disk, in sample --")

    if y_true is not None:
        logger.info("\n" + classification_report(y_true, y_pred, zero_division=0))
        logger.info(f"Confusion matrix:\n{confusion_matrix(y_true, y_pred)}")
    else:
        flagged = int(y_pred.sum())
        logger.info(
            f"No labels, so nothing to score against: flagged {flagged} of {len(X)} "
            f"({flagged/len(X)*100:.2f}%)"
        )

    logger.info(
        f"Decision score stats: mean: {decision_scores.mean():.4f}, "
        f"std: {decision_scores.std():.4f}, "
        f"min: {decision_scores.min():.4f}, "
        f"max: {decision_scores.max():.4f}"
    )


if __name__ == "__main__":
    evaluate()
