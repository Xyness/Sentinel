import sys
import os

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

    def test_the_per_event_rate_makes_a_sane_rate_per_window(self):
        """The model is fitted on windows, not on events, and the two rates are
        not the same number. A per-event 0.01 over a 60-event window puts an
        anomaly in 45 % of them, which is what pinned recall at 2 % however good
        the features were."""
        import importlib
        import config
        importlib.reload(config)

        events_per_window = 60 / config.EVENT_FREQUENCY_SECONDS
        per_window = 1 - (1 - config.ANOMALY_PROBABILITY) ** events_per_window

        assert 0.01 <= per_window <= 0.10, (
            f"{per_window:.3f} of windows would be labelled anomalous"
        )

    def test_the_event_rate_can_be_turned_up_past_one_a_second(self, monkeypatch):
        """CI runs twenty-second windows at five events a second, so a window
        closes inside a job instead of inside a coffee break."""
        monkeypatch.setenv("EVENT_FREQUENCY_SECONDS", "0.2")
        import importlib
        import config
        importlib.reload(config)

        assert config.EVENT_FREQUENCY_SECONDS == 0.2

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "custom-host:9093")
        monkeypatch.setenv("KAFKA_TOPIC", "test-topic")

        import importlib
        import config
        importlib.reload(config)

        assert config.KAFKA_BOOTSTRAP_SERVERS == "custom-host:9093"
        assert config.KAFKA_TOPIC == "test-topic"
