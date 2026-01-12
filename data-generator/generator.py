# generator.py

import json
import time
import random
from kafka import KafkaProducer

from config import (
    KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_TOPIC,
    SYMBOLS,
    EVENT_FREQUENCY_SECONDS,
    ANOMALY_PROBABILITY
)
from market_simulator import MarketSimulator


def create_producer():
    return KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8")
    )


def main():
    producer = create_producer()

    simulators = {
        symbol: MarketSimulator(
            symbol,
            params["initial_price"],
            params["volatility"]
        )
        for symbol, params in SYMBOLS.items()
    }

    print("🚀 Crypto data generator started")

    while True:
        for simulator in simulators.values():
            inject_anomaly = random.random() < ANOMALY_PROBABILITY
            event = simulator.generate_event(inject_anomaly)

            producer.send(KAFKA_TOPIC, event)
            print(event)

        time.sleep(EVENT_FREQUENCY_SECONDS)


if __name__ == "__main__":
    main()
