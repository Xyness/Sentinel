"""Training, and the small datasets it has to survive.

The job starts as soon as MIN_PARQUET_FILES windows exist, so the first run of a
cold stack trains on a handful of rows. Everything here is about that end of the
range, because that is where it used to fall over.
"""

import logging
import os
import sys

import joblib
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ml-python", "training"))

import train_isolation_forest as training  # noqa: E402

FEATURES = [
    "z_score_price",
    "z_score_log_return",
    "z_score_volume",
    "rolling_price_std",
    "rolling_volume_std",
]


def _frame(rows, anomalies):
    """A feature frame with `anomalies` of its `rows` labelled."""
    rng = np.random.default_rng(0)
    data = {name: rng.normal(0, 1, rows) for name in FEATURES}
    data["rolling_price_std"] = abs(data["rolling_price_std"])
    data["rolling_volume_std"] = abs(data["rolling_volume_std"])
    data["is_anomaly"] = [1] * anomalies + [0] * (rows - anomalies)
    return pd.DataFrame(data)


@pytest.fixture
def trains_on(tmp_path, monkeypatch, caplog):
    """Point training at a frame and at a throwaway model path.

    caplog only keeps WARNING and above on its own, and most of what training
    says about the shape of its dataset it says at INFO.
    """
    caplog.set_level(logging.INFO)
    model_path = tmp_path / "isolation_forest.joblib"
    monkeypatch.setattr(training, "MODEL_PATH", str(model_path))

    def run(frame):
        monkeypatch.setattr(training, "load_features", lambda: frame)
        training.train()
        return joblib.load(model_path)

    return run


def test_a_single_anomaly_does_not_take_the_run_down(trains_on, caplog):
    """The regression. train_test_split cannot stratify one member of a class,
    and the whole job died on it, so a cold stack never produced a model."""
    bundle = trains_on(_frame(rows=3, anomalies=1))

    assert set(bundle) == {"model", "scaler"}
    assert "nothing to hold out" in caplog.text


def test_no_labels_at_all_still_trains(trains_on, caplog):
    frame = _frame(rows=60, anomalies=0)
    assert trains_on(frame)["model"] is not None
    assert "unsupervised" in caplog.text


def test_a_thin_run_says_the_scores_are_not_worth_much(trains_on, caplog):
    trains_on(_frame(rows=3, anomalies=1))
    assert "not worth much" in caplog.text


def test_enough_rows_and_both_classes_gets_a_real_holdout(trains_on, caplog):
    trains_on(_frame(rows=200, anomalies=20))
    assert "Test size: 40" in caplog.text
    assert "Test set evaluation" in caplog.text


def test_a_big_but_one_sided_dataset_still_skips_the_split(trains_on, caplog):
    """Enough rows, but stratifying on one anomaly is still impossible."""
    trains_on(_frame(rows=200, anomalies=1))
    assert "nothing to hold out" in caplog.text
    assert "Test set evaluation" not in caplog.text
