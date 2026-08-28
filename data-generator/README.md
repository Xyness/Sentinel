# Data generator

Produces market events and publishes them to Kafka. Two sources, one event shape,
picked with `DATA_SOURCE`.

**Simulated** walks each price with gaussian noise, draws volume from a lognormal,
and injects a labelled anomaly with probability `ANOMALY_PROBABILITY`: a price
spike, a volume spike or a flash crash. The label is what makes evaluation possible,
so it goes on the event and never into training.

**Binance** streams real trades over the public WebSocket, no API key. The connector
emits the same fields, with `is_anomaly` false and `anomaly_type` null throughout,
because nothing out there is labelled.

```json
{
  "timestamp": 1710000000,
  "symbol": "BTC-USDT",
  "price": 43150.50,
  "volume": 12.534210,
  "log_return": 0.003521,
  "is_anomaly": false,
  "anomaly_type": null
}
```

## Configuration

| variable | default | |
|---|---|---|
| `DATA_SOURCE` | `simulated` | or `binance` |
| `KAFKA_BOOTSTRAP_SERVERS` | `localhost:9092` | |
| `KAFKA_TOPIC` | `crypto-market` | |
| `EVENT_FREQUENCY_SECONDS` | `1` | simulated mode only |
| `ANOMALY_PROBABILITY` | `0.01` | simulated mode only |

The three pairs and their starting prices live in `config.py` rather than in the
environment, since changing them means changing the Binance symbol map too.

## Running it on its own

```bash
pip install -r requirements.txt
python generator.py
```

It retries the Kafka connection ten times, five seconds apart, before giving up. On
a cold `docker compose up` the broker is usually not listening yet, and a generator
that exited on the first refusal would need the whole stack restarting in order.
