import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "z_score_price",
    "z_score_log_return",
    "z_score_volume",
    "rolling_price_std",
    "rolling_volume_std"
]


def preprocess(df: pd.DataFrame):
    initial_len = len(df)
    drop_cols = FEATURE_COLUMNS + (["is_anomaly"] if "is_anomaly" in df.columns else [])
    df = df.dropna(subset=drop_cols)
    dropped = initial_len - len(df)

    if dropped > 0:
        logger.warning(f"Dropped {dropped} rows with NaN values ({dropped/initial_len*100:.1f}%)")

    X = df[FEATURE_COLUMNS]

    # Labels may not exist (real data) or may be all 0
    y = None
    if "is_anomaly" in df.columns:
        y = df["is_anomaly"].astype(int)
        n_anomalies = y.sum()
        if n_anomalies == 0:
            logger.info("No labeled anomalies found (real data mode — unsupervised training)")
            y = None
        else:
            logger.info(f"Found {n_anomalies} labeled anomalies ({n_anomalies/len(y)*100:.2f}%)")

    logger.info(f"Preprocessed: {len(X)} samples, features={FEATURE_COLUMNS}, labeled={y is not None}")
    return X, y
