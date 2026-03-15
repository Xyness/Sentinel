"""
Real-time data connector for Binance WebSocket API.
Subscribes to trade streams and produces events in the same format
as the market simulator, so the rest of the pipeline works unchanged.

No API key required — uses public market data only.
"""

import json
import time
import logging
import numpy as np
from websocket import WebSocketApp

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Binance WebSocket base URL
WS_BASE_URL = "wss://stream.binance.com:9443/stream"

# Mapping from our symbol format to Binance stream names
SYMBOL_MAP = {
    "BTC-USDT": "btcusdt",
    "ETH-USDT": "ethusdt",
    "BNB-USDT": "bnbusdt",
}


class BinanceConnector:
    """
    Connects to Binance WebSocket trade streams and converts
    real market trades into the same event format as MarketSimulator.
    """

    def __init__(self, symbols: list, on_event: callable):
        """
        Args:
            symbols: List of symbols like ["BTC-USDT", "ETH-USDT"]
            on_event: Callback function called with each formatted event dict
        """
        self.symbols = symbols
        self.on_event = on_event
        self.last_prices = {}
        self.ws = None
        self._running = False

    def _build_stream_url(self):
        streams = []
        for symbol in self.symbols:
            binance_symbol = SYMBOL_MAP.get(symbol, symbol.replace("-", "").lower())
            streams.append(f"{binance_symbol}@trade")
        return f"{WS_BASE_URL}?streams={'/'.join(streams)}"

    def _binance_to_our_symbol(self, binance_symbol: str) -> str:
        """Convert 'BTCUSDT' back to 'BTC-USDT'."""
        binance_lower = binance_symbol.lower()
        for our_symbol, b_symbol in SYMBOL_MAP.items():
            if b_symbol == binance_lower:
                return our_symbol
        return binance_symbol

    def _on_message(self, ws, message):
        try:
            data = json.loads(message)
            trade = data.get("data", {})

            if trade.get("e") != "trade":
                return

            binance_symbol = trade["s"]
            symbol = self._binance_to_our_symbol(binance_symbol)
            price = float(trade["p"])
            volume = float(trade["q"])
            timestamp = int(trade["T"]) // 1000  # ms to seconds

            # Compute log return
            last_price = self.last_prices.get(symbol, price)
            log_return = float(np.log(price / last_price)) if last_price > 0 else 0.0
            self.last_prices[symbol] = price

            event = {
                "timestamp": timestamp,
                "symbol": symbol,
                "price": round(price, 2),
                "volume": round(volume, 6),
                "log_return": round(log_return, 6),
                "is_anomaly": False,
                "anomaly_type": None
            }

            self.on_event(event)

        except Exception as e:
            logger.error(f"Error processing message: {e}")

    def _on_error(self, ws, error):
        logger.error(f"WebSocket error: {error}")

    def _on_close(self, ws, close_status, close_msg):
        logger.warning(f"WebSocket closed: {close_status} - {close_msg}")

    def _on_open(self, ws):
        logger.info(f"Connected to Binance WebSocket — streaming {self.symbols}")

    def start(self):
        """Start the WebSocket connection with automatic reconnection (blocking)."""
        self._running = True
        while self._running:
            url = self._build_stream_url()
            logger.info(f"Connecting to {url}")

            self.ws = WebSocketApp(
                url,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
                on_open=self._on_open
            )
            self.ws.run_forever()

            if self._running:
                logger.info("Reconnecting in 5 seconds...")
                time.sleep(5)

    def stop(self):
        """Stop the WebSocket connection."""
        self._running = False
        if self.ws:
            self.ws.close()
