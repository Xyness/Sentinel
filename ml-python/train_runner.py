"""
Training runner. Waits for Spark to produce enough Parquet data, trains the
Isolation Forest, then keeps retraining as the feature store grows.
"""
import glob
import logging
import os
import sys
import time

# Ensure ml-python directory is on the path for imports
sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FEATURES_PATH = os.environ.get("FEATURES_PATH", "data/features")
MODEL_PATH = os.environ.get("MODEL_PATH", "models/isolation_forest.joblib")
MIN_FILES = int(os.environ.get("MIN_PARQUET_FILES", "3"))
CHECK_INTERVAL = int(os.environ.get("CHECK_INTERVAL_SECONDS", "15"))
MAX_WAIT = int(os.environ.get("MAX_WAIT_SECONDS", "600"))

# How long to leave between runs. The first model out of a cold stack is fitted
# on the two minutes of data that existed when it started, and a market it saw
# for two minutes is not a market it knows. Set it to 0 to train once and exit,
# which is what this did before.
RETRAIN_INTERVAL = int(os.environ.get("RETRAIN_INTERVAL_SECONDS", "300"))


def feature_files():
    """Every Parquet file with something in it. Empty batches leave zero-byte ones."""
    found = glob.glob(f"{FEATURES_PATH}/**/*.parquet", recursive=True)
    return [path for path in found if os.path.getsize(path) > 0]


def fingerprint():
    """Enough to tell whether the feature store moved since the last run."""
    files = feature_files()
    newest = max((os.path.getmtime(path) for path in files), default=0.0)
    return len(files), newest


def wait_for_data():
    """Wait until enough Parquet files with something in them are available.

    Counting every path matching *.parquet counted the zero-byte files Spark
    leaves behind on an empty batch, so this could release the training job on
    no usable rows at all and load_features would raise on the next line.
    """
    elapsed = 0
    while elapsed < MAX_WAIT:
        files = feature_files()
        if len(files) >= MIN_FILES:
            logger.info(f"Found {len(files)} parquet files, starting training")
            return True

        logger.info(
            f"Waiting for data... ({len(files)}/{MIN_FILES} files found, "
            f"{elapsed}s/{MAX_WAIT}s elapsed)"
        )
        time.sleep(CHECK_INTERVAL)
        elapsed += CHECK_INTERVAL

    logger.warning(f"Timeout: only found {len(feature_files())} files after {MAX_WAIT}s")
    return False


def retrain_forever(train, seen):
    """Retrain on a timer, and only when there is something new to train on.

    A failure here is logged and slept off rather than raised. The model on disk
    is still the last one that worked, the API is still serving it, and taking
    the container down would only replace a stale model with no model at all.
    """
    while True:
        time.sleep(RETRAIN_INTERVAL)

        current = fingerprint()
        if current == seen:
            logger.info("Feature store unchanged, skipping this round")
            continue

        files, _ = current
        logger.info(f"Retraining on {files} parquet file(s)")
        try:
            train()
        except Exception as error:
            logger.error(f"Retraining failed, keeping the model already on disk: {error}")
        else:
            seen = current


def main():
    os.makedirs(os.path.dirname(MODEL_PATH) or ".", exist_ok=True)

    if not wait_for_data():
        logger.error("Not enough data to train. Exiting.")
        sys.exit(1)

    from training.train_isolation_forest import train

    seen = fingerprint()
    train()
    logger.info("Training complete. Model is ready for the API.")

    if RETRAIN_INTERVAL <= 0:
        logger.info("RETRAIN_INTERVAL_SECONDS is 0, training once and exiting")
        return

    logger.info(f"Retraining every {RETRAIN_INTERVAL}s from here")
    retrain_forever(train, seen)


if __name__ == "__main__":
    main()
