# Spark streaming job

Reads the market events off Kafka and turns them into the features everything
downstream uses. Java, built with Maven, submitted with `spark-submit`.

## What it computes

Tumbling windows per symbol, one minute by default, with a watermark the same
length. `RollingFeatures` aggregates the events in a window, `TechnicalIndicators`
turns those aggregates into five dimensionless ratios, and `FeatureAssembler`
drops the partial windows and picks the columns that leave the job. Out comes
one row per symbol per window, written as Parquet partitioned by symbol.

| feature | |
|---|---|
| `abs_return_max` | largest absolute log return in the window |
| `return_std` | realised volatility over the window |
| `price_range_rel` | `(high - low) / mean price` |
| `volume_max_ratio` | largest single trade over the window's mean volume |
| `volume_cv` | volume deviation over mean volume |

Two properties, both on purpose.

They're dimensionless, so a row from a 43,000 dollar pair and a row from a 320
dollar one are comparable. With absolute deviations the model separates symbols
before it separates anomalies.

They're order independent, so no `last()` and no `first()`. An aggregation
shuffles, and `last(price)` returns whichever row the executor saw last rather
than the last one in time, which means the same input can give two answers.

## What was wrong before

The features used to be z-scores of a window's last price against that same
window's mean and deviation. For a step of size `J` at event `k` of an `n` event
window that comes out as `sqrt(k/(n-k))`: `J` cancels, so the size of the move
never reached the model, and a spike anywhere but the last second of the window
was invisible while the label still said the window was anomalous.

`FeatureAssemblerTest` builds a window with a jump in the middle of it and
asserts the features see it, at any position and at any size.

## Configuration

Environment variables, read in `CryptoStreamJob`:

| variable | default | |
|---|---|---|
| `KAFKA_BOOTSTRAP_SERVERS` | `localhost:9092` | |
| `KAFKA_TOPIC` | `crypto-market` | |
| `OUTPUT_PATH` | `data/features` | where the Parquet goes |
| `CHECKPOINT_PATH` | `data/checkpoints/sentinel` | Spark's own state |
| `WINDOW_DURATION` | `1 minute` | also the watermark |
| `MIN_EVENTS_PER_WINDOW` | `10` | below this the window is dropped |

It reads from `latest`, so it does not replay a topic it was not running for.
CI sets `WINDOW_DURATION` to twenty seconds so the whole pipeline fits inside a
job.

## Build and test

```bash
mvn test
mvn package -DskipTests
```

The tests stand a real local SparkSession up and run over batch DataFrames, so
they cover the aggregation and the ratios rather than mocking around them.

## Why Java

The rest of the ML work is Python, but this is the part that has to keep up with a
stream, and Java is the language the Spark API is native to. It also keeps a hard
line between the Big Data half of the project and the Machine Learning half.
