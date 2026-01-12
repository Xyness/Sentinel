# market_simulator.py

import numpy as np
import random
import time


class MarketSimulator:
    def __init__(self, symbol, initial_price, volatility):
        self.symbol = symbol
        self.price = initial_price
        self.volatility = volatility
        self.last_price = initial_price

    def _simulate_price(self):
        noise = np.random.normal(0, self.volatility)
        self.price *= (1 + noise)

    def _simulate_volume(self):
        return np.random.lognormal(mean=2.5, sigma=0.5)

    def _inject_anomaly(self):
        anomaly_type = random.choice([
            "PRICE_SPIKE",
            "VOLUME_SPIKE",
            "FLASH_CRASH"
        ])

        if anomaly_type == "PRICE_SPIKE":
            self.price *= random.uniform(1.05, 1.15)

        elif anomaly_type == "VOLUME_SPIKE":
            return anomaly_type, self._simulate_volume() * 8

        elif anomaly_type == "FLASH_CRASH":
            self.price *= random.uniform(0.85, 0.95)

        return anomaly_type, None

    def generate_event(self, inject_anomaly=False):
        self.last_price = self.price
        self._simulate_price()
        volume = self._simulate_volume()

        is_anomaly = False
        anomaly_type = None

        if inject_anomaly:
            anomaly_type, anomalous_volume = self._inject_anomaly()
            is_anomaly = True
            if anomalous_volume:
                volume = anomalous_volume

        log_return = np.log(self.price / self.last_price)

        return {
            "timestamp": int(time.time()),
            "symbol": self.symbol,
            "price": round(self.price, 2),
            "volume": round(volume, 4),
            "log_return": round(float(log_return), 6),
            "is_anomaly": is_anomaly,
            "anomaly_type": anomaly_type
        }
