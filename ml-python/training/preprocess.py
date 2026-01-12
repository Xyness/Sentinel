import pandas as pd

FEATURE_COLUMNS = [
    "z_score_price",
    "z_score_volume",
    "rolling_price_std",
    "rolling_volume_std"
]

def preprocess(df: pd.DataFrame):
    df = df.dropna()
    X = df[FEATURE_COLUMNS]
    y = df["is_anomaly"]
    return X, y
