import logging

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# The contract with FeatureAssembler.java, the scorer and the API schema. All
# five are dimensionless, so a row from BTC and a row from BNB are comparable
# and the model spends itself on anomalies rather than on telling pairs apart.
FEATURE_COLUMNS = [
    "abs_return_max",
    "return_std",
    "price_range_rel",
    "volume_max_ratio",
    "volume_cv",
]


def preprocess(df: pd.DataFrame):
    missing = [column for column in FEATURE_COLUMNS if column not in df.columns]
    if missing:
        # The feature set is written down in Java, here and in the API schema,
        # and they only meet through Parquet on a shared volume. Say which one
        # drifted rather than dying on a KeyError three frames down.
        raise KeyError(
            f"the feature store is missing {', '.join(missing)}. "
            f"Expected {FEATURE_COLUMNS}, found {list(df.columns)}. "
            f"FeatureAssembler.java and this list have to agree."
        )

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
            logger.info("No labeled anomalies found (real data mode, unsupervised training)")
            y = None
        else:
            logger.info(f"Found {n_anomalies} labeled anomalies ({n_anomalies/len(y)*100:.2f}%)")

    logger.info(f"Preprocessed: {len(X)} samples, features={FEATURE_COLUMNS}, labeled={y is not None}")
    return X, y
