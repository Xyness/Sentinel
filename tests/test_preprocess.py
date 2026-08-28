import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ml-python", "training"))

from preprocess import FEATURE_COLUMNS, preprocess


def _features(n, rng):
    """A frame of plausible feature values. All five have a floor of zero."""
    return {
        "abs_return_max": np.abs(rng.randn(n)) * 0.002,
        "return_std": np.abs(rng.randn(n)) * 0.001,
        "price_range_rel": np.abs(rng.randn(n)) * 0.004,
        "volume_max_ratio": 1 + np.abs(rng.randn(n)),
        "volume_cv": np.abs(rng.randn(n)) * 0.4,
    }


class TestPreprocess:

    def _make_df(self, n=100, nan_count=0, anomaly_ratio=0.01):
        rng = np.random.RandomState(42)
        labels = rng.choice([0, 1], size=n, p=[1 - anomaly_ratio, anomaly_ratio])
        if anomaly_ratio > 0 and labels.sum() == 0:
            labels[0] = 1
        df = pd.DataFrame({**_features(n, rng), "is_anomaly": labels})
        if nan_count > 0:
            idx = np.random.choice(n, nan_count, replace=False)
            df.loc[idx, "abs_return_max"] = np.nan
        return df

    def test_output_shape_with_labels(self):
        df = self._make_df(100, anomaly_ratio=0.05)
        X, y = preprocess(df)
        assert X.shape == (100, 5)
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
        df = pd.DataFrame(_features(50, np.random.RandomState(0)))
        X, y = preprocess(df)
        assert len(X) == 50
        assert y is None

    def test_empty_after_dropna(self):
        df = pd.DataFrame({**{name: [np.nan, np.nan] for name in FEATURE_COLUMNS},
                           "is_anomaly": [0, 1]})
        X, y = preprocess(df)
        assert len(X) == 0

    def test_a_feature_store_from_an_older_job_says_so(self):
        """The feature set lives in Java, here and in the API schema, and they
        only meet through Parquet. A drift used to surface as a KeyError three
        frames down at inference time."""
        stale = pd.DataFrame({"z_score_price": [1.0], "rolling_volume_std": [2.0]})

        with pytest.raises(KeyError, match="FeatureAssembler.java"):
            preprocess(stale)
