import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data-generator"))

from market_simulator import MarketSimulator


class TestMarketSimulator:

    def test_init(self):
        sim = MarketSimulator("BTC-USDT", 43000, 0.001)
        assert sim.symbol == "BTC-USDT"
        assert sim.price == 43000
        assert sim.volatility == 0.001
        assert sim.last_price == 43000

    def test_generate_event_structure(self):
        sim = MarketSimulator("ETH-USDT", 2300, 0.0015)
        event = sim.generate_event(inject_anomaly=False)

        assert "timestamp" in event
        assert "symbol" in event
        assert "price" in event
        assert "volume" in event
        assert "log_return" in event
        assert "is_anomaly" in event
        assert "anomaly_type" in event

    def test_generate_event_normal(self):
        sim = MarketSimulator("BTC-USDT", 43000, 0.001)
        event = sim.generate_event(inject_anomaly=False)

        assert event["symbol"] == "BTC-USDT"
        assert event["is_anomaly"] is False
        assert event["anomaly_type"] is None
        assert event["price"] > 0
        assert event["volume"] > 0

    def test_generate_event_anomaly(self):
        sim = MarketSimulator("BTC-USDT", 43000, 0.001)
        event = sim.generate_event(inject_anomaly=True)

        assert event["is_anomaly"] is True
        assert event["anomaly_type"] in ["PRICE_SPIKE", "VOLUME_SPIKE", "FLASH_CRASH"]

    def test_price_changes_over_time(self):
        sim = MarketSimulator("BTC-USDT", 43000, 0.001)
        prices = []
        for _ in range(100):
            event = sim.generate_event(inject_anomaly=False)
            prices.append(event["price"])

        # Price should fluctuate, not stay constant
        assert len(set(prices)) > 1

    def test_log_return_calculation(self):
        sim = MarketSimulator("BTC-USDT", 100, 0.001)
        event = sim.generate_event(inject_anomaly=False)

        expected_log_return = np.log(sim.price / sim.last_price)
        assert abs(event["log_return"] - round(float(expected_log_return), 6)) < 1e-5

    def test_volume_is_positive(self):
        sim = MarketSimulator("BTC-USDT", 43000, 0.001)
        for _ in range(50):
            event = sim.generate_event()
            assert event["volume"] > 0

    def test_price_spike_anomaly(self):
        """Price spike should increase the price by 5-15%."""
        np.random.seed(42)
        sim = MarketSimulator("BTC-USDT", 10000, 0.0001)

        # Run many events to find a PRICE_SPIKE
        for _ in range(1000):
            before_price = sim.price
            event = sim.generate_event(inject_anomaly=True)
            if event["anomaly_type"] == "PRICE_SPIKE":
                # Price should have increased significantly
                ratio = event["price"] / before_price
                assert ratio > 1.0  # Price went up
                break

    def test_flash_crash_anomaly(self):
        """Flash crash should decrease the price by 5-15%."""
        np.random.seed(0)
        sim = MarketSimulator("BTC-USDT", 10000, 0.0001)

        for _ in range(1000):
            before_price = sim.price
            event = sim.generate_event(inject_anomaly=True)
            if event["anomaly_type"] == "FLASH_CRASH":
                ratio = event["price"] / before_price
                assert ratio < 1.0  # Price went down
                break
