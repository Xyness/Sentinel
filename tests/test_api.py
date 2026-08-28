import os
import sys
from unittest.mock import MagicMock

import joblib
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))

VECTOR = {
    "symbol": "BTC-USDT",
    "abs_return_max": 0.1150,
    "return_std": 0.0210,
    "price_range_rel": 0.1240,
    "volume_max_ratio": 4.2,
    "volume_cv": 1.30,
}


class TestSchemas:

    def test_feature_vector_valid(self):
        from schemas import FeatureVector
        fv = FeatureVector(**VECTOR)
        assert fv.symbol == "BTC-USDT"
        assert fv.abs_return_max == 0.1150

    def test_feature_vector_rejects_a_negative_magnitude(self):
        """Every one of the five is a magnitude. A negative price range is not
        a market condition, it is a unit mix-up upstream."""
        from schemas import FeatureVector
        with pytest.raises(Exception):
            FeatureVector(**{**VECTOR, "price_range_rel": -0.001})

    def test_feature_vector_rejects_an_impossible_return(self):
        from schemas import FeatureVector
        # a log return of 50 is e^50, which is not a price move
        with pytest.raises(Exception):
            FeatureVector(**{**VECTOR, "abs_return_max": 50.0})


class TestModelLoader:

    def test_missing_model_not_loaded(self):
        """Model should not crash on init with missing file, just not load."""
        import model_loader
        from model_loader import AnomalyModel
        original_path = model_loader.MODEL_PATH
        model_loader.MODEL_PATH = "/nonexistent/model.joblib"
        try:
            am = AnomalyModel()
            assert am.loaded is False
        finally:
            model_loader.MODEL_PATH = original_path

    def test_predict_with_nan_raises(self):
        """Model should reject NaN inputs."""
        from model_loader import AnomalyModel

        am = AnomalyModel.__new__(AnomalyModel)
        am.model = MagicMock()
        am.scaler = MagicMock()
        am.loaded = True

        with pytest.raises(ValueError, match="NaN"):
            am.predict([1.0, float("nan"), 0.5, 0.002, 10.0])

    def test_predict_with_inf_raises(self):
        """Model should reject infinite inputs."""
        from model_loader import AnomalyModel

        am = AnomalyModel.__new__(AnomalyModel)
        am.model = MagicMock()
        am.scaler = MagicMock()
        am.loaded = True

        with pytest.raises(ValueError, match="infinite"):
            am.predict([1.0, float("inf"), 0.5, 0.002, 10.0])

    def test_predict_not_loaded_raises(self):
        """Model should raise if not loaded."""
        from model_loader import AnomalyModel

        am = AnomalyModel.__new__(AnomalyModel)
        am.model = None
        am.scaler = None
        am.loaded = False

        with pytest.raises(RuntimeError, match="not loaded"):
            am.predict([1.0, 1.0, 0.5, 0.002, 10.0])

    def test_predict_returns_tuple(self):
        """Model predict should return (score, is_anomaly)."""
        from model_loader import AnomalyModel

        mock_model = MagicMock()
        mock_model.decision_function.return_value = np.array([-0.5])
        mock_model.predict.return_value = np.array([-1])

        mock_scaler = MagicMock()
        mock_scaler.transform.return_value = np.array([[1.0, 1.0, 0.5, 0.002, 10.0]])

        am = AnomalyModel.__new__(AnomalyModel)
        am.model = mock_model
        am.scaler = mock_scaler
        am.loaded = True

        score, is_anomaly = am.predict([1.0, 1.0, 0.5, 0.002, 10.0])
        assert score == -0.5
        assert is_anomaly is True

    def test_predict_many_scores_the_whole_batch_in_one_pass(self):
        from model_loader import AnomalyModel

        mock_model = MagicMock()
        mock_model.decision_function.return_value = np.array([-0.5, 0.3])
        mock_model.predict.return_value = np.array([-1, 1])

        mock_scaler = MagicMock()
        mock_scaler.transform.return_value = np.zeros((2, 5))

        am = AnomalyModel.__new__(AnomalyModel)
        am.model = mock_model
        am.scaler = mock_scaler
        am.loaded = True

        assert am.predict_many([[0.0] * 5, [0.0] * 5]) == [(-0.5, True), (0.3, False)]
        assert mock_scaler.transform.call_count == 1

    def test_an_empty_batch_never_reaches_the_model(self):
        from model_loader import AnomalyModel

        am = AnomalyModel.__new__(AnomalyModel)
        am.model = MagicMock()
        am.scaler = MagicMock()
        am.loaded = True

        assert am.predict_many([]) == []
        assert am.scaler.transform.call_count == 0


class TestModelReload:
    """Training no longer runs once and exits, so the file under MODEL_PATH
    changes while the API is up. Returning early on `loaded` meant the first
    model ever written was served for the life of the process."""

    def _bundle(self, path, tag, when=None):
        joblib.dump({"model": tag, "scaler": tag, "features": ["a"], "metrics": {"tag": tag}},
                    path)
        if when is not None:
            os.utime(path, (when, when))

    def test_a_newer_model_on_disk_is_picked_up(self, tmp_path, monkeypatch):
        import model_loader
        path = tmp_path / "isolation_forest.joblib"
        monkeypatch.setattr(model_loader, "MODEL_PATH", str(path))

        self._bundle(path, "first", when=1_700_000_000)
        am = model_loader.AnomalyModel()
        assert am.model == "first"

        self._bundle(path, "second", when=1_700_000_060)
        am.ensure_loaded()
        assert am.model == "second"
        assert am.metrics == {"tag": "second"}

    def test_an_unchanged_file_is_not_reloaded(self, tmp_path, monkeypatch):
        import model_loader
        path = tmp_path / "isolation_forest.joblib"
        monkeypatch.setattr(model_loader, "MODEL_PATH", str(path))

        self._bundle(path, "first", when=1_700_000_000)
        am = model_loader.AnomalyModel()

        calls = []
        original = am._try_load
        monkeypatch.setattr(am, "_try_load", lambda: calls.append(1) or original())

        am.ensure_loaded()
        assert calls == []

    def test_a_model_that_vanished_is_still_served(self, tmp_path, monkeypatch):
        """Refusing to score because the file went missing would help nobody:
        the model in memory is still a model."""
        import model_loader
        path = tmp_path / "isolation_forest.joblib"
        monkeypatch.setattr(model_loader, "MODEL_PATH", str(path))

        self._bundle(path, "first")
        am = model_loader.AnomalyModel()
        path.unlink()

        assert am.ensure_loaded() is True
        assert am.loaded is True
        assert am.model == "first"


class TestAPIEndpoints:

    @pytest.fixture
    def client(self):
        """Create a test client with a mocked model and an empty store."""
        import main as api_main
        from store import PredictionStore

        mock_model = MagicMock()
        mock_model.loaded = True
        mock_model.ensure_loaded.return_value = True
        mock_model.predict.return_value = (-0.3, True)
        mock_model.predict_many.side_effect = lambda rows: [(-0.3, True)] * len(rows)
        mock_model.features = ["abs_return_max", "return_std", "price_range_rel",
                               "volume_max_ratio", "volume_cv"]
        mock_model.metrics = {"n_train": 160, "holdout": {"precision": 0.8, "recall": 0.7,
                                                          "f1": 0.75, "support": 4}}
        api_main.model = mock_model
        api_main.store = PredictionStore(":memory:")

        from fastapi.testclient import TestClient
        return TestClient(api_main.app)

    def test_health_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data

    def test_predict_endpoint(self, client):
        response = client.post("/predict", json=VECTOR)
        assert response.status_code == 200
        data = response.json()
        assert data["symbol"] == "BTC-USDT"
        assert "anomaly_score" in data
        assert "is_anomaly" in data

    def test_predict_invalid_input(self, client):
        response = client.post("/predict", json={**VECTOR, "abs_return_max": "not_a_number"})
        assert response.status_code == 422

    def test_predict_missing_fields(self, client):
        response = client.post("/predict", json={"symbol": "BTC-USDT"})
        assert response.status_code == 422

    def test_a_scored_vector_is_stored(self, client):
        client.post("/predict", json=VECTOR)
        stored = client.get("/latest-predictions").json()

        assert len(stored) == 1
        assert stored[0]["symbol"] == "BTC-USDT"
        assert stored[0]["abs_return_max"] == pytest.approx(0.1150)
        assert stored[0]["id"] == 1

    def test_batch_scores_every_vector_in_order(self, client):
        payload = {"vectors": [VECTOR, {**VECTOR, "symbol": "ETH-USDT"}]}
        response = client.post("/predict/batch", json=payload)

        assert response.status_code == 200
        assert [r["symbol"] for r in response.json()] == ["BTC-USDT", "ETH-USDT"]
        assert len(client.get("/latest-predictions").json()) == 2

    def test_an_empty_batch_is_refused(self, client):
        assert client.post("/predict/batch", json={"vectors": []}).status_code == 422

    def test_a_bad_vector_takes_the_whole_batch_down(self, client):
        """Half a batch scored and half rejected is a worse answer than none."""
        payload = {"vectors": [VECTOR, {**VECTOR, "volume_cv": -1.0}]}
        assert client.post("/predict/batch", json=payload).status_code == 422
        assert client.get("/latest-predictions").json() == []

    def test_latest_predictions_endpoint(self, client):
        response = client.get("/latest-predictions")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_latest_predictions_with_symbol_filter(self, client):
        client.post("/predict", json={**VECTOR, "symbol": "ETH-USDT"})
        client.post("/predict", json=VECTOR)

        response = client.get("/latest-predictions?symbol=ETH-USDT")
        assert response.status_code == 200
        assert [item["symbol"] for item in response.json()] == ["ETH-USDT"]

    def test_latest_predictions_after_an_id(self, client):
        for _ in range(3):
            client.post("/predict", json=VECTOR)

        assert [i["id"] for i in client.get("/latest-predictions?after=1").json()] == [2, 3]

    def test_stats_over_an_empty_store(self, client):
        stats = client.get("/stats").json()
        assert stats["total_predictions"] == 0
        assert stats["per_symbol"] == {}

    def test_stats_break_down_by_symbol(self, client):
        client.post("/predict", json=VECTOR)
        client.post("/predict", json={**VECTOR, "symbol": "ETH-USDT"})

        stats = client.get("/stats").json()
        assert stats["total_predictions"] == 2
        assert set(stats["per_symbol"]) == {"BTC-USDT", "ETH-USDT"}
        assert set(stats["feature_stats"]) == set(VECTOR) - {"symbol"}

    def test_model_info_serves_the_scorecard_the_bundle_carries(self, client):
        info = client.get("/model-info").json()

        assert info["loaded"] is True
        assert info["feature_names"][0] == "abs_return_max"
        assert info["metrics"]["holdout"]["precision"] == 0.8

    def test_system_status_no_longer_reports_a_zookeeper(self, client):
        names = {s["name"] for s in client.get("/system-status").json()["services"]}
        assert "Zookeeper" not in names
        assert {"API", "Spark", "Kafka"} == names


class TestColdApi:

    def test_predict_says_the_model_is_not_ready_yet(self):
        import main as api_main
        from store import PredictionStore

        cold = MagicMock()
        cold.loaded = False
        cold.ensure_loaded.return_value = False
        api_main.model = cold
        api_main.store = PredictionStore(":memory:")

        from fastapi.testclient import TestClient
        response = TestClient(api_main.app).post("/predict", json=VECTOR)

        assert response.status_code == 503
        assert "Training may still be in progress" in response.json()["detail"]
