"""The terminal client: what it asks the API for, and what it prints back."""

import io
import json
import os
import sys

import httpx
import pytest
from rich.console import Console

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "cli"))

from sentinel import cli, render                      # noqa: E402
from sentinel.client import ApiError, Client          # noqa: E402


def _item(number=1, symbol="BTC-USDT", score=-0.18, anomaly=True):
    return {
        "id": number,
        "timestamp": f"2026-08-27T10:00:{number:02d}+00:00",
        "symbol": symbol,
        "abs_return_max": 0.1150,
        "return_std": 0.0210,
        "price_range_rel": 0.1240,
        "volume_max_ratio": 4.2,
        "volume_cv": 1.30,
        "anomaly_score": score,
        "is_anomaly": anomaly,
    }


def _text(renderable, width=96) -> str:
    console = Console(file=io.StringIO(), width=width, no_color=True, highlight=False)
    console.print(renderable)
    return console.file.getvalue()


@pytest.fixture
def api(monkeypatch):
    """The API, in process. Routes answer with whatever the test put in them."""
    routes = {}
    seen = []

    def handle(request):
        seen.append(request)
        answer = routes.get(request.url.path)
        if answer is None:
            return httpx.Response(404, json={"detail": "no route"})
        if callable(answer):
            return answer(request)
        return httpx.Response(200, json=answer)

    transport = httpx.MockTransport(handle)
    monkeypatch.setattr(cli, "Client",
                        lambda url, *a, **kw: Client(url, transport=transport))

    fake = type("Fake", (), {})()
    fake.routes, fake.requests, fake.transport = routes, seen, transport
    return fake


def _args(command, argv):
    return cli.build_parser().parse_args([command, *argv])


def _console():
    return Console(file=io.StringIO(), width=96, no_color=True, highlight=False)


# --- the client -------------------------------------------------------------


def test_client_drops_unset_query_params(api):
    api.routes["/latest-predictions"] = [_item()]
    Client("http://api", transport=api.transport).latest_predictions(limit=10, symbol=None)
    assert api.requests[-1].url.params.get("limit") == "10"
    assert "symbol" not in api.requests[-1].url.params


def test_client_reports_the_api_reason_not_the_status_code(api):
    api.routes["/predict"] = lambda request: httpx.Response(
        503, json={"detail": "Model not yet available. Training may still be in progress."})
    with pytest.raises(ApiError, match="Training may still be in progress"):
        Client("http://api", transport=api.transport).predict({})


def test_client_says_where_it_could_not_reach():
    def refuse(request):
        raise httpx.ConnectError("connection refused")

    with pytest.raises(ApiError, match="http://api:8000 is not answering"):
        Client("http://api:8000", transport=httpx.MockTransport(refuse)).health()


# --- building a feature vector ----------------------------------------------


def test_preset_fills_the_vector_and_flags_beat_it():
    args = _args("predict", ["--preset", "flash-crash", "--volume-peak", "0.25"])
    features = cli._features(args)
    assert features["abs_return_max"] == pytest.approx(0.1150)
    assert features["volume_max_ratio"] == pytest.approx(0.25)


def test_a_half_given_vector_is_an_error_rather_than_zeros():
    args = _args("predict", ["--max-return", "0.115"])
    with pytest.raises(ValueError, match="volume_cv"):
        cli._features(args)


def test_stdin_beats_the_preset():
    args = _args("predict", ["--preset", "normal"])
    features = cli._features(args, overrides={"abs_return_max": 0.99, "symbol": "ETH-USDT"})
    assert features["abs_return_max"] == pytest.approx(0.99)
    assert features["symbol"] == "ETH-USDT"


# --- status -----------------------------------------------------------------


def test_status_reports_an_outage_instead_of_crashing_on_it(monkeypatch):
    def refuse(request):
        raise httpx.ConnectError("connection refused")

    transport = httpx.MockTransport(refuse)
    monkeypatch.setattr(cli, "Client", lambda url, *a, **kw: Client(url, transport=transport))

    console = _console()
    exit_code = cli.cmd_status(_args("status", ["--api", "http://api:8000"]), console)
    output = console.file.getvalue()

    assert exit_code == cli.EXIT_ERROR
    assert "the API did not answer" in output
    assert "is not answering" in output


def test_status_says_a_cold_model_is_not_a_failure(api):
    api.routes["/health"] = {"status": "waiting_for_model", "model_loaded": False}
    api.routes["/system-status"] = {"services": [], "timestamp": ""}
    api.routes["/model-info"] = {"loaded": False}

    console = _console()
    assert cli.cmd_status(_args("status", []), console) == cli.EXIT_OK
    assert "training may still be running" in console.file.getvalue()


def test_status_prints_the_scorecard_the_model_carries(api):
    """The whole point of the bundle carrying its own metrics: the number is
    one command away instead of buried in a training log nobody kept."""
    api.routes["/health"] = {"status": "ok", "model_loaded": True}
    api.routes["/system-status"] = {"services": [], "timestamp": ""}
    api.routes["/model-info"] = {
        "loaded": True, "model_type": "IsolationForest", "n_estimators": 200,
        "contamination": 0.05, "max_samples": "auto",
        "metrics": {"n_train": 1600, "label_rate": 0.048,
                    "holdout": {"precision": 0.81, "recall": 0.74, "f1": 0.77,
                                "support": 19}},
    }

    console = _console()
    assert cli.cmd_status(_args("status", []), console) == cli.EXIT_OK
    output = console.file.getvalue()

    assert "1,600 rows" in output
    assert "precision 0.810" in output
    assert "recall 0.740" in output
    assert "19 labelled anomalies" in output


def test_status_says_so_when_there_was_nothing_to_score_against(api):
    api.routes["/health"] = {"status": "ok", "model_loaded": True}
    api.routes["/system-status"] = {"services": [], "timestamp": ""}
    api.routes["/model-info"] = {
        "loaded": True, "model_type": "IsolationForest", "n_estimators": 200,
        "contamination": 0.05, "max_samples": "auto",
        "metrics": {"n_train": 900, "labelled": False, "holdout": None},
    }

    console = _console()
    cli.cmd_status(_args("status", []), console)
    assert "nothing to score against" in console.file.getvalue()


def test_unreachable_stages_are_drawn_unchecked_not_green():
    line = _text(render.pipeline_line([
        {"name": "API", "status": "online"},
        {"name": "Kafka", "status": "online"},
        {"name": "Spark", "status": "offline"},
    ]))
    assert "? generator" in line
    assert "? scorer" in line
    assert "! spark" in line
    assert "+ kafka" in line


# --- the feed ---------------------------------------------------------------


def test_feed_asks_only_for_what_it_has_not_seen(api):
    api.routes["/latest-predictions"] = [_item(1), _item(2)]
    cli.cmd_feed(_args("feed", ["--once", "--tail", "5"]), _console())
    assert api.requests[-1].url.params.get("limit") == "5"


def test_feed_prints_each_prediction_once(api):
    api.routes["/latest-predictions"] = [_item(1, score=-0.2), _item(2, score=0.1, anomaly=False)]
    console = _console()
    cli.cmd_feed(_args("feed", ["--once"]), console)
    output = console.file.getvalue()

    assert output.count("BTC-USDT") == 2
    assert "anomaly" in output and "normal" in output


def test_feed_can_drop_the_normal_ones(api):
    api.routes["/latest-predictions"] = [_item(1), _item(2, anomaly=False)]
    console = _console()
    cli.cmd_feed(_args("feed", ["--once", "--anomalies"]), console)
    assert console.file.getvalue().count("BTC-USDT") == 1


def test_an_empty_buffer_says_where_predictions_come_from(api):
    api.routes["/latest-predictions"] = []
    console = _console()
    cli.cmd_feed(_args("feed", ["--once"]), console)
    assert "scorer" in console.file.getvalue()


def test_a_filter_that_matched_nothing_is_not_an_empty_pipeline(api):
    api.routes["/latest-predictions"] = []
    console = _console()
    cli.cmd_feed(_args("feed", ["--once", "--symbol", "DOGE-USDT"]), console)
    output = console.file.getvalue()

    assert "nothing scored for DOGE-USDT" in output
    assert "scorer" not in output


def test_feed_line_columns_line_up():
    lines = [_text(render.feed_line(_item(1, "BTC-USDT", -0.1832, True))).rstrip("\n"),
             _text(render.feed_line(_item(2, "ETH-USDT", 0.0741, False))).rstrip("\n")]
    assert lines[0].index("ret ") == lines[1].index("ret ")


# --- stats ------------------------------------------------------------------


def test_a_flat_score_window_draws_flat():
    drawn = _text(render.sparkline([0.5, 0.5, 0.5, 0.5])).strip()
    assert len(set(drawn)) == 1


def test_narrowing_to_one_symbol_drops_the_figures_computed_across_all(api):
    stats = cli._narrow({
        "total_predictions": 10, "total_anomalies": 4, "anomaly_rate": 40.0,
        "avg_score": -0.1,
        "per_symbol": {"BTC-USDT": {"count": 6, "anomalies": 1, "anomaly_rate": 16.67,
                                    "avg_score": -0.05}},
        "score_percentiles": {"p50": -0.1},
        "feature_stats": {"abs_return_max": {"mean": 0.001}},
    }, "BTC-USDT")

    assert stats["total_predictions"] == 6
    assert stats["score_percentiles"] is None
    assert stats["feature_stats"] is None


def test_narrowing_to_a_symbol_nobody_scored_is_an_error():
    with pytest.raises(ValueError, match="ETH-USDT"):
        cli._narrow({"per_symbol": {}}, "ETH-USDT")


# --- export -----------------------------------------------------------------


def test_csv_keeps_a_fixed_column_order(api, tmp_path):
    api.routes["/latest-predictions"] = [_item(1), _item(2, "ETH-USDT")]
    out = tmp_path / "export.csv"
    cli.cmd_export(_args("export", ["-o", str(out)]), _console())

    rows = out.read_text(encoding="utf-8").splitlines()
    assert rows[0] == ",".join(cli.CSV_COLUMNS)
    assert len(rows) == 3


def test_export_can_keep_only_the_flagged_rows(api, tmp_path):
    api.routes["/latest-predictions"] = [_item(1), _item(2, anomaly=False)]
    out = tmp_path / "export.json"
    cli.cmd_export(_args("export", ["-f", "json", "-o", str(out), "--anomalies"]), _console())
    assert len(json.loads(out.read_text(encoding="utf-8"))) == 1


# --- exit codes -------------------------------------------------------------


def test_predict_can_fail_a_pipeline_on_an_anomaly(api):
    api.routes["/predict"] = {"symbol": "BTC-USDT", "anomaly_score": -0.31, "is_anomaly": True}
    code = cli.main(["predict", "--preset", "flash-crash", "--fail-on-anomaly"])
    assert code == cli.EXIT_ANOMALY


def test_a_normal_vector_leaves_the_pipeline_green(api):
    api.routes["/predict"] = {"symbol": "BTC-USDT", "anomaly_score": 0.12, "is_anomaly": False}
    assert cli.main(["predict", "--preset", "normal", "--fail-on-anomaly"]) == cli.EXIT_OK


def test_an_unreachable_api_is_an_error_not_an_empty_report(api):
    assert cli.main(["stats"]) == cli.EXIT_ERROR


def test_no_command_prints_help_and_exits_clean(capsys):
    assert cli.main([]) == cli.EXIT_OK
    assert "status" in capsys.readouterr().out
