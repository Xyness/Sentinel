import os

# Data source: "simulated" or "binance"
DATA_SOURCE = os.environ.get("DATA_SOURCE", "simulated")

KAFKA_BOOTSTRAP_SERVERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC = os.environ.get("KAFKA_TOPIC", "crypto-market")

SYMBOLS = {
    "BTC-USDT": {
        "initial_price": 43000,
        "volatility": 0.001
    },
    "ETH-USDT": {
        "initial_price": 2300,
        "volatility": 0.0015
    },
    "BNB-USDT": {
        "initial_price": 320,
        "volatility": 0.002
    }
}

EVENT_FREQUENCY_SECONDS = int(os.environ.get("EVENT_FREQUENCY_SECONDS", "1"))

ANOMALY_PROBABILITY = float(os.environ.get("ANOMALY_PROBABILITY", "0.01"))
