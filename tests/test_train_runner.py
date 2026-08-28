"""The retraining loop.

The first model out of a cold stack is fitted on the two minutes of data that
existed when training started, and it used to be the model served for the rest
of the run because the job trained once and exited.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ml-python"))

import train_runner as runner  # noqa: E402


class Stop(Exception):
    """Breaks out of a loop that is otherwise deliberately infinite."""


@pytest.fixture
def features(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "FEATURES_PATH", str(tmp_path))
    monkeypatch.setattr(runner, "RETRAIN_INTERVAL", 1)
    return tmp_path


def _write(directory, name, symbol="BTC-USDT"):
    partition = directory / f"symbol={symbol}"
    partition.mkdir(parents=True, exist_ok=True)
    path = partition / name
    path.write_bytes(b"not really parquet, but it has a size")
    return path


def _sleeps(limit):
    """A sleep that lets the loop go round `limit` times, then stops it."""
    calls = []

    def sleep(seconds):
        calls.append(seconds)
        if len(calls) > limit:
            raise Stop

    return sleep, calls


def test_the_empty_files_spark_leaves_behind_are_not_data(features):
    _write(features, "real.parquet")
    (features / "symbol=BTC-USDT" / "empty.parquet").write_bytes(b"")

    assert len(runner.feature_files()) == 1


def test_waiting_for_data_does_not_count_the_empty_ones(features, monkeypatch):
    """It used to glob every path ending in .parquet, so a directory of empty
    batches released the training job and load_features raised on the line
    after it."""
    monkeypatch.setattr(runner, "MIN_FILES", 2)
    monkeypatch.setattr(runner, "MAX_WAIT", 1)
    monkeypatch.setattr(runner, "CHECK_INTERVAL", 1)
    monkeypatch.setattr(runner.time, "sleep", lambda seconds: None)

    _write(features, "real.parquet")
    (features / "symbol=BTC-USDT" / "empty.parquet").write_bytes(b"")

    assert runner.wait_for_data() is False

    _write(features, "second.parquet")
    assert runner.wait_for_data() is True


def test_the_fingerprint_moves_when_a_window_lands(features):
    _write(features, "part-0.parquet")
    before = runner.fingerprint()

    _write(features, "part-1.parquet")
    assert runner.fingerprint() != before


def test_nothing_new_means_no_retraining(features, monkeypatch):
    _write(features, "part-0.parquet")
    sleep, _ = _sleeps(2)
    monkeypatch.setattr(runner.time, "sleep", sleep)

    trained = []
    with pytest.raises(Stop):
        runner.retrain_forever(lambda: trained.append(1), runner.fingerprint())

    assert trained == []


def test_a_new_window_gets_retrained_on(features, monkeypatch):
    _write(features, "part-0.parquet")
    seen = runner.fingerprint()

    trained = []

    def sleep(seconds):
        if trained:
            raise Stop
        _write(features, "part-1.parquet")

    monkeypatch.setattr(runner.time, "sleep", sleep)

    with pytest.raises(Stop):
        runner.retrain_forever(lambda: trained.append(1), seen)

    assert trained == [1]


def test_a_failed_retrain_keeps_the_model_already_on_disk(features, monkeypatch, caplog):
    """The model on disk is still the last one that worked and the API is
    still serving it. Dying here would trade a stale model for no model."""
    _write(features, "part-0.parquet")
    seen = runner.fingerprint()

    attempts = []

    def sleep(seconds):
        if len(attempts) >= 2:
            raise Stop
        _write(features, f"part-{len(attempts) + 1}.parquet")

    def explode():
        attempts.append(1)
        raise RuntimeError("no space left on device")

    monkeypatch.setattr(runner.time, "sleep", sleep)

    with pytest.raises(Stop):
        runner.retrain_forever(explode, seen)

    assert len(attempts) == 2
    assert "keeping the model already on disk" in caplog.text
