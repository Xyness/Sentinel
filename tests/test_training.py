"""Training, and the two things it has to get right.

The small datasets it has to survive: the job starts as soon as
MIN_PARQUET_FILES windows exist, so the first run of a cold stack trains on a
handful of rows.

And the contamination it fits with. That used to be pinned at 1 % while the
labels carried 45 %, which capped recall at a fiftieth of what was there no
matter how good the features were. It is read off the labels now, so changing
the generator cannot silently break the model.
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
    "abs_return_max",
    "return_std",
    "price_range_rel",
    "volume_max_ratio",
    "volume_cv",
]


def _frame(rows, anomalies):
    """A feature frame with `anomalies` of its `rows` labelled.

    Every feature has a floor of zero, so the noise is folded rather than
    centred: a negative price range is not a thing the pipeline can produce.
    """
    rng = np.random.default_rng(0)
    data = {name: abs(rng.normal(0, 1, rows)) for name in FEATURES}
    data["volume_max_ratio"] = 1 + data["volume_max_ratio"]
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

    run.path = model_path
    return run


# --- the small datasets -----------------------------------------------------


def test_a_single_anomaly_does_not_take_the_run_down(trains_on, caplog):
    """The regression. train_test_split cannot stratify one member of a class,
    and the whole job died on it, so a cold stack never produced a model."""
    bundle = trains_on(_frame(rows=3, anomalies=1))

    assert set(bundle) == {"model", "scaler", "features", "metrics"}
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


# --- contamination ----------------------------------------------------------


def test_contamination_follows_the_label_rate(trains_on):
    bundle = trains_on(_frame(rows=200, anomalies=20))

    assert bundle["metrics"]["contamination"] == pytest.approx(0.10)
    assert bundle["model"].contamination == pytest.approx(0.10)


def test_a_rate_nothing_could_call_anomalous_is_clamped(trains_on, caplog):
    """Half the rows labelled is a generator turned up too far, not a market
    that is half broken. Believing it would have the forest flag half of
    everything and the word stop meaning anything."""
    bundle = trains_on(_frame(rows=200, anomalies=100))

    assert bundle["metrics"]["contamination"] == pytest.approx(training.MAX_CONTAMINATION)
    assert "clamped" in caplog.text


def test_an_unlabelled_run_falls_back_to_the_stated_rate(trains_on):
    bundle = trains_on(_frame(rows=60, anomalies=0))

    assert bundle["metrics"]["contamination"] == pytest.approx(training.DEFAULT_CONTAMINATION)
    assert bundle["metrics"]["labelled"] is False
    assert bundle["metrics"]["holdout"] is None


# --- what the bundle carries ------------------------------------------------


def test_the_bundle_carries_its_own_scorecard(trains_on):
    """So /model-info can serve it and `sentinel status` can print it. Numbers
    that only ever existed in a training log are numbers nobody can check."""
    bundle = trains_on(_frame(rows=200, anomalies=20))
    holdout = bundle["metrics"]["holdout"]

    assert set(holdout) == {"precision", "recall", "f1", "support"}
    assert 0.0 <= holdout["precision"] <= 1.0
    assert 0.0 <= holdout["recall"] <= 1.0
    assert holdout["support"] == 4          # 20 % of 20 labelled anomalies
    assert bundle["metrics"]["n_train"] == 160
    assert bundle["metrics"]["n_test"] == 40


def test_the_bundle_names_the_features_it_was_fitted_on(trains_on):
    assert trains_on(_frame(rows=60, anomalies=6))["features"] == FEATURES


def test_the_model_is_renamed_into_place_never_written_over(trains_on):
    """The API reloads on the file's timestamp, so a dump straight onto the
    live path would let it read a half-written bundle."""
    trains_on(_frame(rows=60, anomalies=6))

    assert trains_on.path.exists()
    assert not trains_on.path.with_suffix(".joblib.tmp").exists()
    assert list(trains_on.path.parent.glob("*.tmp")) == []
