"""The scorer: what it reads off disk, and what it sends on."""

import json
import os
import sys

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scorer"))

import scorer  # noqa: E402

ROWS = {
    "abs_return_max": [0.1150, 0.0012],
    "return_std": [0.0210, 0.0008],
    "price_range_rel": [0.1240, 0.0035],
    "volume_max_ratio": [4.2, 1.8],
    "volume_cv": [1.30, 0.35],
}


def _parquet(directory, symbol="BTC-USDT", name="part-0.parquet", rows=None):
    """One Parquet file where Spark would have put it, partitioned by symbol."""
    partition = directory / f"symbol={symbol}"
    partition.mkdir(parents=True, exist_ok=True)
    path = partition / name
    pq.write_table(pa.table(rows or ROWS), path)
    return str(path)


@pytest.fixture
def features(tmp_path, monkeypatch):
    monkeypatch.setattr(scorer, "FEATURES_PATH", str(tmp_path))
    return tmp_path


def _client(handler):
    return httpx.Client(base_url="http://api:8000", transport=httpx.MockTransport(handler))


def _scores(request):
    """Answer a batch with one result per vector, in the order they came."""
    vectors = json.loads(request.content)["vectors"]
    return httpx.Response(200, json=[
        {"symbol": vector["symbol"], "anomaly_score": -0.2, "is_anomaly": True}
        for vector in vectors
    ])


# --- reading what Spark wrote ----------------------------------------------


def test_the_symbol_comes_off_the_path_because_it_is_not_in_the_file(features):
    path = _parquet(features, symbol="ETH-USDT")
    assert "symbol" not in pq.ParquetFile(path).schema_arrow.names
    assert all(vector["symbol"] == "ETH-USDT" for vector in scorer.rows_in(path))


def test_a_file_outside_a_symbol_partition_is_skipped(features):
    path = features / "loose.parquet"
    pq.write_table(pa.table(ROWS), path)
    assert scorer.rows_in(str(path)) == []


def test_rows_with_a_null_in_them_never_reach_the_api(features):
    holed = {**ROWS, "price_range_rel": [0.1240, None]}
    assert len(scorer.rows_in(_parquet(features, rows=holed))) == 1


def test_the_empty_batches_spark_leaves_behind_are_ignored(features):
    (features / "symbol=BTC-USDT").mkdir(parents=True)
    (features / "symbol=BTC-USDT" / "empty.parquet").write_bytes(b"")
    real = _parquet(features, name="part-1.parquet")
    assert scorer.pending(set()) == [real]


def test_a_file_already_read_is_not_offered_again(features):
    path = _parquet(features)
    assert scorer.pending({path}) == []


# --- sending it on ----------------------------------------------------------


def test_every_row_becomes_one_prediction(features):
    sent = []

    def handler(request):
        sent.append(request)
        return _scores(request)

    count, anomalies = scorer.score(_client(handler), scorer.rows_in(_parquet(features)))
    assert (count, anomalies) == (2, 2)
    assert all(r.url.path == "/predict/batch" for r in sent)


def test_a_whole_file_goes_out_in_one_call(features):
    """It used to be a round trip and a single-row scaler call per row."""
    sent = []

    def handler(request):
        sent.append(request)
        return _scores(request)

    scorer.score(_client(handler), scorer.rows_in(_parquet(features)))
    assert len(sent) == 1
    assert len(json.loads(sent[0].content)["vectors"]) == 2


def test_a_backlog_is_split_into_batches(features, monkeypatch):
    monkeypatch.setattr(scorer, "BATCH_SIZE", 1)
    sent = []

    def handler(request):
        sent.append(request)
        return _scores(request)

    count, _ = scorer.score(_client(handler), scorer.rows_in(_parquet(features)))
    assert (count, len(sent)) == (2, 2)


def test_a_cold_model_is_waited_for_rather_than_dropped(features):
    def handler(request):
        return httpx.Response(503, json={"detail": "Model not yet available."})

    with pytest.raises(scorer.ModelNotReady, match="Model not yet available"):
        scorer.score(_client(handler), scorer.rows_in(_parquet(features)))


def test_only_the_five_features_and_the_symbol_go_out(features):
    seen = {}

    def handler(request):
        for vector in json.loads(request.content)["vectors"]:
            seen.update(vector)
        return _scores(request)

    scorer.score(_client(handler), scorer.rows_in(_parquet(features)))
    assert set(seen) == {"symbol", *scorer.FEATURE_COLUMNS}


def test_nothing_to_send_is_not_a_call(features):
    def handler(request):
        raise AssertionError("should not have been called")

    assert scorer.score(_client(handler), []) == (0, 0)
