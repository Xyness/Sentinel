import sys
import os
import json
import pytest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data-generator"))

from binance_connector import BinanceConnector, SYMBOL_MAP


class TestBinanceConnector:

    def test_symbol_mapping(self):
        assert SYMBOL_MAP["BTC-USDT"] == "btcusdt"
        assert SYMBOL_MAP["ETH-USDT"] == "ethusdt"
        assert SYMBOL_MAP["BNB-USDT"] == "bnbusdt"

    def test_reverse_symbol_mapping(self):
        connector = BinanceConnector(symbols=["BTC-USDT"], on_event=lambda e: None)
        assert connector._binance_to_our_symbol("BTCUSDT") == "BTC-USDT"
        assert connector._binance_to_our_symbol("ETHUSDT") == "ETH-USDT"
        assert connector._binance_to_our_symbol("BNBUSDT") == "BNB-USDT"

    def test_stream_url_construction(self):
        connector = BinanceConnector(
            symbols=["BTC-USDT", "ETH-USDT"],
            on_event=lambda e: None
        )
        url = connector._build_stream_url()
        assert "btcusdt@trade" in url
        assert "ethusdt@trade" in url
        assert "wss://stream.binance.com" in url

    def test_message_processing(self):
        events = []
        connector = BinanceConnector(
            symbols=["BTC-USDT"],
            on_event=lambda e: events.append(e)
        )

        # Simulate a Binance trade message
        trade_message = json.dumps({
            "data": {
                "e": "trade",
                "s": "BTCUSDT",
                "p": "43150.50",
                "q": "0.125000",
                "T": 1710000000000
            }
        })

        connector._on_message(None, trade_message)

        assert len(events) == 1
        event = events[0]
        assert event["symbol"] == "BTC-USDT"
        assert event["price"] == 43150.50
        assert event["volume"] == 0.125
        assert event["timestamp"] == 1710000000
        assert event["is_anomaly"] is False
        assert event["anomaly_type"] is None

    def test_log_return_calculation(self):
        events = []
        connector = BinanceConnector(
            symbols=["BTC-USDT"],
            on_event=lambda e: events.append(e)
        )

        # First trade
        msg1 = json.dumps({
            "data": {"e": "trade", "s": "BTCUSDT", "p": "100.00", "q": "1.0", "T": 1000000}
        })
        connector._on_message(None, msg1)

        # Second trade at higher price
        msg2 = json.dumps({
            "data": {"e": "trade", "s": "BTCUSDT", "p": "105.00", "q": "1.0", "T": 1001000}
        })
        connector._on_message(None, msg2)

        assert len(events) == 2
        # First event: log_return = 0 (no previous price)
        assert events[0]["log_return"] == 0.0
        # Second event: log_return = ln(105/100) ≈ 0.04879
        assert abs(events[1]["log_return"] - 0.04879) < 0.001

    def test_ignores_non_trade_messages(self):
        events = []
        connector = BinanceConnector(
            symbols=["BTC-USDT"],
            on_event=lambda e: events.append(e)
        )

        non_trade = json.dumps({
            "data": {"e": "kline", "s": "BTCUSDT"}
        })
        connector._on_message(None, non_trade)

        assert len(events) == 0
