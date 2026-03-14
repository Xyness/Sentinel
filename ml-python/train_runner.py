"""
Training runner that waits for Spark to produce enough Parquet data,
then trains the Isolation Forest model and saves it for the API.
"""
import os
import sys
import time
import glob
import logging

# Ensure ml-python directory is on the path for imports
sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FEATURES_PATH = os.environ.get("FEATURES_PATH", "data/features")
MODEL_PATH = os.environ.get("MODEL_PATH", "models/isolation_forest.joblib")
MIN_FILES = int(os.environ.get("MIN_PARQUET_FILES", "3"))
CHECK_INTERVAL = int(os.environ.get("CHECK_INTERVAL_SECONDS", "15"))
MAX_WAIT = int(os.environ.get("MAX_WAIT_SECONDS", "600"))


def wait_for_data():
    """Wait until enough Parquet files are available."""
    elapsed = 0
    while elapsed < MAX_WAIT:
        files = glob.glob(f"{FEATURES_PATH}/**/*.parquet", recursive=True)
        if len(files) >= MIN_FILES:
            logger.info(f"Found {len(files)} parquet files, starting training")
            return True

        logger.info(
            f"Waiting for data... ({len(files)}/{MIN_FILES} files found, "
            f"{elapsed}s/{MAX_WAIT}s elapsed)"
        )
        time.sleep(CHECK_INTERVAL)
        elapsed += CHECK_INTERVAL

    remaining = glob.glob(f"{FEATURES_PATH}/**/*.parquet", recursive=True)
    logger.warning(f"Timeout: only found {len(remaining)} files after {MAX_WAIT}s")
    return False


def main():
    os.makedirs(os.path.dirname(MODEL_PATH) or ".", exist_ok=True)

    if not wait_for_data():
        logger.error("Not enough data to train. Exiting.")
        sys.exit(1)

    from training.train_isolation_forest import train
    train()

    logger.info("Training complete. Model is ready for the API.")


if __name__ == "__main__":
    main()
