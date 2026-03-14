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
    files = glob.glob(f"{path}/**/*.parquet", recursive=True)

    if not files:
        raise FileNotFoundError(f"No parquet files found in {path}")

    logger.info(f"Loading {len(files)} parquet file(s) from {path}")
    dfs = [pq.read_table(f).to_pandas() for f in files]
    df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


if __name__ == "__main__":
    df = load_features()
    print(df.head())
