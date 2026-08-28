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

# A float, so a test or a CI run can turn the rate up and fill a window in
# seconds instead of a minute.
EVENT_FREQUENCY_SECONDS = float(os.environ.get("EVENT_FREQUENCY_SECONDS", "1"))

# Drawn once per event, but the model is fitted on windows, and the two rates
# are not the same number. At one event a second, a one-minute window is 60
# draws, so a per-event 0.01 puts an anomaly in 1 - 0.99^60 = 45 % of windows.
# Training would then be told to flag 5 % of rows while 45 % were labelled, and
# no feature set can recover from that.
#
# Solving 1 - (1-p)^60 = 0.05 the other way round gives this. It is one
# anomalous minute in twenty per pair, which is rare enough to be worth the
# name and frequent enough to see on a demo.
ANOMALY_PROBABILITY = float(os.environ.get("ANOMALY_PROBABILITY", "0.00085"))
