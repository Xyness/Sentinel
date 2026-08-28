"""Scores what Spark wrote, through the API.

Spark writes feature rows to Parquet and the training job reads them, but until
now nothing put those rows back through the model: `/predict` only ever got
called by hand, so the prediction buffer stayed empty on a pipeline that was
otherwise running fine. This closes the loop.
"""

import glob
import logging
import os
import time
from pathlib import Path

import httpx
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
FEATURES_PATH = os.environ.get("FEATURES_PATH", "data/features")
INTERVAL_SECONDS = int(os.environ.get("SCORE_INTERVAL_SECONDS", "15"))

FEATURE_COLUMNS = [
    "z_score_price",
    "z_score_log_return",
    "z_score_volume",
    "rolling_price_std",
    "rolling_volume_std",
]


class ModelNotReady(Exception):
    """The API is up but has no model yet. Worth waiting for, not worth losing rows over."""


def symbol_from_path(path):
    """Spark partitions by symbol, so the pair is a directory name.

    It is not a column inside the file. pyarrow can infer it back from a hive
    path, but that depends on the version and on the layout being exactly the
    one it expects, and reading the directory name is the same answer with
    nothing behind it that can change underneath.
    """
    for part in Path(path).parts:
        if part.startswith("symbol="):
            return part.split("=", 1)[1]
    return None


def rows_in(path):
    """The feature vectors in one Parquet file, in the shape /predict wants."""
    symbol = symbol_from_path(path)
    if symbol is None:
        logger.warning(f"No symbol partition in {path}, skipping it")
        return []

    table = pq.read_table(path, columns=FEATURE_COLUMNS)

    vectors = []
    for row in table.to_pylist():
        # A window Spark could not complete leaves nulls behind. The API would
        # reject them, and a partial vector is not a prediction worth having.
        if any(row.get(column) is None for column in FEATURE_COLUMNS):
            continue
        vectors.append({
            "symbol": symbol,
            **{column: float(row[column]) for column in FEATURE_COLUMNS},
        })
    return vectors


def pending(seen):
    """New Parquet files, oldest first.

    Spark's parquet sink never rewrites a file it has closed, so one that has
    been read is done with and the path is enough to remember it by. Zero-byte
    files are the empty batches it leaves behind.
    """
    found = glob.glob(f"{FEATURES_PATH}/**/*.parquet", recursive=True)
    fresh = [path for path in found if path not in seen and os.path.getsize(path) > 0]
    return sorted(fresh, key=os.path.getmtime)


def score(client, vectors):
    """POST each vector, and count what came back flagged."""
    anomalies = 0
    for vector in vectors:
        response = client.post("/predict", json=vector)
        if response.status_code == 503:
            raise ModelNotReady(response.json().get("detail", "no model loaded"))
        response.raise_for_status()
        anomalies += bool(response.json().get("is_anomaly"))
    return len(vectors), anomalies


def main():
    logger.info(f"Scorer starting: {FEATURES_PATH} -> {API_BASE_URL}, every {INTERVAL_SECONDS}s")

    # Files already on disk are sent too. Remembering a cursor across restarts
    # would need a writable volume, and all it would protect is a 500-entry ring
    # buffer, so a restart replaying what is still there is the cheaper trade.
    seen = set()
    total = flagged = 0

    with httpx.Client(base_url=API_BASE_URL, timeout=30.0) as client:
        while True:
            for path in pending(seen):
                try:
                    sent, anomalies = score(client, rows_in(path))
                except ModelNotReady as error:
                    logger.info(f"Waiting for the model: {error}")
                    break
                except httpx.HTTPError as error:
                    logger.warning(f"API call failed, will retry: {error}")
                    break

                # Marked only once the whole file landed. One interrupted
                # halfway is sent again from the start, which can duplicate the
                # rows that did land; at three rows a minute into a ring buffer
                # that beats tracking an offset per file.
                seen.add(path)
                total += sent
                flagged += anomalies
                if sent:
                    logger.info(f"Scored {sent} row(s) from {os.path.basename(path)}, "
                                f"{anomalies} flagged (total {total}, {flagged} flagged)")

            time.sleep(INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
