import os
import logging
import glob
import pandas as pd
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FEATURES_PATH = os.environ.get("FEATURES_PATH", "data/features")


def load_features(path=None):
    path = path or FEATURES_PATH
    all_files = glob.glob(f"{path}/**/*.parquet", recursive=True)
    files = [f for f in all_files if os.path.getsize(f) > 0]

    if not files:
        raise FileNotFoundError(f"No valid parquet files found in {path}")

    logger.info(f"Loading {len(files)} parquet file(s) from {path} (skipped {len(all_files) - len(files)} empty)")
    dfs = [pq.read_table(f).to_pandas() for f in files]
    df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


if __name__ == "__main__":
    df = load_features()
    print(df.head())
