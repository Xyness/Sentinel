import json
import time
import random
import logging
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable

from config import (
    DATA_SOURCE,
    KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_TOPIC,
    SYMBOLS,
    EVENT_FREQUENCY_SECONDS,
    ANOMALY_PROBABILITY
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MAX_RETRIES = 10
RETRY_DELAY_SECONDS = 5


def create_producer():
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            producer = KafkaProducer(
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                retries=3,
                acks="all"
            )
            logger.info(f"Connected to Kafka at {KAFKA_BOOTSTRAP_SERVERS}")
            return producer
        except NoBrokersAvailable:
            logger.warning(
                f"Kafka not available (attempt {attempt}/{MAX_RETRIES}), "
                f"retrying in {RETRY_DELAY_SECONDS}s..."
            )
            time.sleep(RETRY_DELAY_SECONDS)

    raise ConnectionError(
        f"Could not connect to Kafka at {KAFKA_BOOTSTRAP_SERVERS} "
        f"after {MAX_RETRIES} attempts"
    )


def run_simulated(producer):
    """Run with simulated market data."""
    from market_simulator import MarketSimulator

    simulators = {
        symbol: MarketSimulator(
            symbol,
            params["initial_price"],
            params["volatility"]
        )
        for symbol, params in SYMBOLS.items()
    }

    logger.info(
        f"Simulated mode — symbols={list(SYMBOLS.keys())}, "
        f"frequency={EVENT_FREQUENCY_SECONDS}s, anomaly_rate={ANOMALY_PROBABILITY}"
    )

    events_sent = 0
    anomalies_sent = 0

    while True:
        for simulator in simulators.values():
            inject_anomaly = random.random() < ANOMALY_PROBABILITY
            event = simulator.generate_event(inject_anomaly)

            producer.send(KAFKA_TOPIC, event)
            events_sent += 1

            if event.get("is_anomaly"):
                anomalies_sent += 1
                logger.info(f"Anomaly injected: {event['symbol']} - {event['anomaly_type']}")

        if events_sent % 100 == 0:
            logger.info(f"Events sent: {events_sent} (anomalies: {anomalies_sent})")

        time.sleep(EVENT_FREQUENCY_SECONDS)


def run_binance(producer):
    """Run with real Binance market data via WebSocket."""
    from binance_connector import BinanceConnector

    events_sent = 0

    def on_event(event):
        nonlocal events_sent
        producer.send(KAFKA_TOPIC, event)
        events_sent += 1
        if events_sent % 500 == 0:
            logger.info(f"Real events sent: {events_sent} (latest: {event['symbol']} @ {event['price']})")

    symbols = list(SYMBOLS.keys())
    connector = BinanceConnector(symbols=symbols, on_event=on_event)

    logger.info(f"Binance mode — streaming real trades for {symbols}")
    connector.start()


def main():
    producer = create_producer()

    logger.info(f"Data generator starting — mode={DATA_SOURCE}")

    if DATA_SOURCE == "binance":
        run_binance(producer)
    elif DATA_SOURCE == "simulated":
        run_simulated(producer)
    else:
        raise ValueError(f"Unknown DATA_SOURCE: {DATA_SOURCE}. Use 'simulated' or 'binance'.")


if __name__ == "__main__":
    main()
