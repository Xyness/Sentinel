import sys
import os
import pytest
from unittest.mock import patch, MagicMock
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))


class TestSchemas:

    def test_feature_vector_valid(self):
        from schemas import FeatureVector
        fv = FeatureVector(
            symbol="BTC-USDT",
            z_score_price=1.5,
            z_score_volume=-0.3,
            rolling_price_std=0.002,
            rolling_volume_std=10.0
        )
        assert fv.symbol == "BTC-USDT"
        assert fv.z_score_price == 1.5

    def test_feature_vector_rejects_negative_std(self):
        from schemas import FeatureVector
        with pytest.raises(Exception):
            FeatureVector(
                symbol="BTC-USDT",
                z_score_price=1.0,
                z_score_volume=1.0,
                rolling_price_std=-0.001,
                rolling_volume_std=10.0
            )

    def test_feature_vector_rejects_extreme_zscore(self):
        from schemas import FeatureVector
        with pytest.raises(Exception):
            FeatureVector(
                symbol="BTC-USDT",
                z_score_price=200.0,
                z_score_volume=1.0,
                rolling_price_std=0.001,
                rolling_volume_std=10.0
            )


class TestModelLoader:

    def test_missing_model_not_loaded(self):
        """Model should not crash on init with missing file, just not load."""
        from model_loader import AnomalyModel
        import model_loader
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
            am.predict([1.0, float("nan"), 0.002, 10.0])

    def test_predict_with_inf_raises(self):
        """Model should reject infinite inputs."""
        from model_loader import AnomalyModel

        am = AnomalyModel.__new__(AnomalyModel)
        am.model = MagicMock()
        am.scaler = MagicMock()
        am.loaded = True

        with pytest.raises(ValueError, match="infinite"):
            am.predict([1.0, float("inf"), 0.002, 10.0])

    def test_predict_not_loaded_raises(self):
        """Model should raise if not loaded."""
        from model_loader import AnomalyModel

        am = AnomalyModel.__new__(AnomalyModel)
        am.model = None
        am.scaler = None
        am.loaded = False

        with pytest.raises(RuntimeError, match="not loaded"):
            am.predict([1.0, 1.0, 0.002, 10.0])

    def test_predict_returns_tuple(self):
        """Model predict should return (score, is_anomaly)."""
        from model_loader import AnomalyModel

        mock_model = MagicMock()
        mock_model.decision_function.return_value = np.array([-0.5])
        mock_model.predict.return_value = np.array([-1])

        mock_scaler = MagicMock()
        mock_scaler.transform.return_value = np.array([[1.0, 1.0, 0.002, 10.0]])

        am = AnomalyModel.__new__(AnomalyModel)
        am.model = mock_model
        am.scaler = mock_scaler
        am.loaded = True

        score, is_anomaly = am.predict([1.0, 1.0, 0.002, 10.0])
        assert score == -0.5
        assert is_anomaly is True


class TestAPIEndpoints:

    @pytest.fixture
    def client(self):
        """Create a test client with mocked model."""
        import main as api_main

        mock_model = MagicMock()
        mock_model.loaded = True
        mock_model.ensure_loaded.return_value = True
        mock_model.predict.return_value = (-0.3, True)
        api_main.model = mock_model

        from fastapi.testclient import TestClient
        return TestClient(api_main.app)

    def test_health_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data

    def test_predict_endpoint(self, client):
        payload = {
            "symbol": "BTC-USDT",
            "z_score_price": 2.0,
            "z_score_volume": 1.5,
            "rolling_price_std": 0.003,
            "rolling_volume_std": 15.0
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["symbol"] == "BTC-USDT"
        assert "anomaly_score" in data
        assert "is_anomaly" in data

    def test_predict_invalid_input(self, client):
        payload = {
            "symbol": "BTC-USDT",
            "z_score_price": "not_a_number",
            "z_score_volume": 1.5,
            "rolling_price_std": 0.003,
            "rolling_volume_std": 15.0
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_missing_fields(self, client):
        payload = {"symbol": "BTC-USDT"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422
