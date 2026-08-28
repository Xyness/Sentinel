"""Where predictions live now that they outlive the process."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))

from store import FEATURE_COLUMNS, PredictionStore  # noqa: E402

FEATURES = {
    "abs_return_max": 0.1150,
    "return_std": 0.0210,
    "price_range_rel": 0.1240,
    "volume_max_ratio": 4.2,
    "volume_cv": 1.30,
}


@pytest.fixture
def store():
    memory = PredictionStore(":memory:")
    yield memory
    memory.close()


def _fill(store, count, symbol="BTC-USDT"):
    return [store.append(symbol, FEATURES, -0.1, index % 2 == 0) for index in range(count)]


def test_an_appended_row_comes_back_with_an_id_and_a_timestamp(store):
    row = store.append("BTC-USDT", FEATURES, -0.31, True)

    assert row["id"] == 1
    assert row["symbol"] == "BTC-USDT"
    assert row["is_anomaly"] is True
    assert row["timestamp"].startswith("20")
    assert all(row[name] == pytest.approx(FEATURES[name]) for name in FEATURE_COLUMNS)


def test_the_feed_reads_oldest_first(store):
    _fill(store, 3)
    assert [row["id"] for row in store.latest()] == [1, 2, 3]


def test_a_limit_takes_the_most_recent_not_the_first(store):
    _fill(store, 5)
    assert [row["id"] for row in store.latest(limit=2)] == [4, 5]


def test_filtering_by_symbol(store):
    store.append("BTC-USDT", FEATURES, -0.1, True)
    store.append("ETH-USDT", FEATURES, -0.2, False)

    assert [row["symbol"] for row in store.latest(symbol="ETH-USDT")] == ["ETH-USDT"]


def test_after_an_id_is_what_a_follower_asks_for(store):
    _fill(store, 4)
    assert [row["id"] for row in store.latest(after=2)] == [3, 4]


def test_a_flag_survives_the_round_trip_as_a_bool(store):
    """SQLite has no boolean, so it goes in as an integer. Coming back out as
    one would put a 1 in the JSON where every client expects true."""
    store.append("BTC-USDT", FEATURES, -0.1, True)
    assert store.latest()[0]["is_anomaly"] is True


def test_the_table_is_capped_rather_than_growing_forever():
    small = PredictionStore(":memory:", max_rows=50)
    try:
        # Pruning runs on a counter rather than on every insert, so the cap is
        # a ceiling with one batch of slack, not an exact row count.
        _fill(small, 500)
        assert small.count() <= 50 + 200
        assert small.latest(limit=1)[0]["id"] == 500
    finally:
        small.close()


def test_a_file_backed_store_survives_the_process_that_wrote_it(tmp_path):
    path = str(tmp_path / "nested" / "predictions.db")

    first = PredictionStore(path)
    first.append("BTC-USDT", FEATURES, -0.42, True)
    first.close()

    second = PredictionStore(path)
    try:
        rows = second.latest()
        assert len(rows) == 1
        assert rows[0]["anomaly_score"] == pytest.approx(-0.42)
    finally:
        second.close()
