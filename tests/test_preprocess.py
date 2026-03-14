import sys
import os
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ml-python", "training"))

from preprocess import preprocess, FEATURE_COLUMNS


class TestPreprocess:

    def _make_df(self, n=100, nan_count=0, anomaly_ratio=0.01):
        df = pd.DataFrame({
            "z_score_price": np.random.randn(n),
            "z_score_volume": np.random.randn(n),
            "rolling_price_std": np.abs(np.random.randn(n)) * 0.001,
            "rolling_volume_std": np.abs(np.random.randn(n)) * 10,
            "is_anomaly": np.random.choice(
                [0, 1], size=n, p=[1 - anomaly_ratio, anomaly_ratio]
            ),
        })
        if nan_count > 0:
            idx = np.random.choice(n, nan_count, replace=False)
            df.loc[idx, "z_score_price"] = np.nan
        return df

    def test_output_shape_with_labels(self):
        df = self._make_df(100, anomaly_ratio=0.05)
        X, y = preprocess(df)
        assert X.shape == (100, 4)
        assert y is not None
        assert len(y) == 100

    def test_feature_columns(self):
        df = self._make_df(50, anomaly_ratio=0.05)
        X, y = preprocess(df)
        assert list(X.columns) == FEATURE_COLUMNS

    def test_drops_nan_rows(self):
        df = self._make_df(100, nan_count=10, anomaly_ratio=0.05)
        X, y = preprocess(df)
        assert len(X) == 90
        assert not X.isnull().any().any()

    def test_no_labels_returns_none(self):
        """When is_anomaly is all 0, y should be None (unsupervised mode)."""
        df = self._make_df(100, anomaly_ratio=0.0)
        df["is_anomaly"] = 0
        X, y = preprocess(df)
        assert len(X) == 100
        assert y is None

    def test_missing_is_anomaly_column(self):
        """When is_anomaly column doesn't exist, y should be None."""
        df = pd.DataFrame({
            "z_score_price": np.random.randn(50),
            "z_score_volume": np.random.randn(50),
            "rolling_price_std": np.abs(np.random.randn(50)) * 0.001,
            "rolling_volume_std": np.abs(np.random.randn(50)) * 10,
        })
        X, y = preprocess(df)
        assert len(X) == 50
        assert y is None

    def test_empty_after_dropna(self):
        df = pd.DataFrame({
            "z_score_price": [np.nan, np.nan],
            "z_score_volume": [np.nan, np.nan],
            "rolling_price_std": [np.nan, np.nan],
            "rolling_volume_std": [np.nan, np.nan],
            "is_anomaly": [0, 1],
        })
        X, y = preprocess(df)
        assert len(X) == 0
