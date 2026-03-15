import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data-generator"))


class TestConfig:

    def test_default_values(self):
        import importlib
        import config
        importlib.reload(config)

        assert config.DATA_SOURCE == "simulated"
        assert config.KAFKA_TOPIC == "crypto-market"
        assert config.EVENT_FREQUENCY_SECONDS >= 1
        assert 0 < config.ANOMALY_PROBABILITY <= 1.0

    def test_data_source_binance(self, monkeypatch):
        monkeypatch.setenv("DATA_SOURCE", "binance")
        import importlib
        import config
        importlib.reload(config)
        assert config.DATA_SOURCE == "binance"

    def test_symbols_structure(self):
        from config import SYMBOLS

        assert len(SYMBOLS) > 0
        for symbol, params in SYMBOLS.items():
            assert "initial_price" in params
            assert "volatility" in params
            assert params["initial_price"] > 0
            assert params["volatility"] > 0

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "custom-host:9093")
        monkeypatch.setenv("KAFKA_TOPIC", "test-topic")

        import importlib
        import config
        importlib.reload(config)

        assert config.KAFKA_BOOTSTRAP_SERVERS == "custom-host:9093"
        assert config.KAFKA_TOPIC == "test-topic"
