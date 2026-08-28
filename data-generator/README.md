# Data generator

Produces market events and publishes them to Kafka. Two sources, one event shape,
picked with `DATA_SOURCE`.

**Simulated** walks each price with gaussian noise, draws volume from a lognormal,
and injects a labelled anomaly with probability `ANOMALY_PROBABILITY`: a price
spike, a volume spike or a flash crash. The label is what makes evaluation possible,
so it goes on the event and never into the fit.

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

## The probability is per event, and the model sees windows

Worth being careful with, because it is what broke the detection.

The draw happens once per event. Spark aggregates a minute of them into one row
and labels that row anomalous if any event in it was. At one event a second
that's 60 draws, so a per-event `p` gives a per-window rate of `1 - (1-p)^60`. A
per-event 0.01 looks small and marks 45 % of windows.

The default solves that the other way round: `1 - (1-p)^60 = 0.05` gives
0.00085, so about one minute in twenty per pair carries an anomaly. If you change
it, change it knowing which of the two rates you are setting. Training reads the
contamination off the labels, so the model follows either way, but a per-window
rate over 25 % is clamped and warned about.

## Configuration

| variable | default | |
|---|---|---|
| `DATA_SOURCE` | `simulated` | or `binance` |
| `KAFKA_BOOTSTRAP_SERVERS` | `localhost:9092` | |
| `KAFKA_TOPIC` | `crypto-market` | |
| `EVENT_FREQUENCY_SECONDS` | `1` | simulated mode only, takes a float |
| `ANOMALY_PROBABILITY` | `0.00085` | simulated mode only, per event |

`EVENT_FREQUENCY_SECONDS` is a float so a CI run can turn the rate up and fill a
short window in seconds. The three pairs and their starting prices live in
`config.py` rather than in the environment, since changing them means changing
the Binance symbol map too.

## Running it on its own

```bash
pip install -r requirements.txt
python generator.py
```

It retries the Kafka connection ten times, five seconds apart, before giving up. On
a cold `docker compose up` the broker is usually not listening yet, and a generator
that exited on the first refusal would need the whole stack restarting in order.
