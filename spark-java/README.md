# Spark streaming job

Reads the market events off Kafka and turns them into the features everything
downstream uses. Java, built with Maven, submitted with `spark-submit`.

## What it computes

One-minute tumbling windows per symbol, with a one-minute watermark. Inside each
window: rolling mean and standard deviation for price, log return and volume, and
z-scores off those. Out comes seven columns, written as Parquet partitioned by
symbol.

Where a standard deviation is zero the z-score returns 0 rather than a NaN. A flat
minute is not an anomaly, and a NaN there propagates through the whole feature
vector and out the other side of training.

`TechnicalIndicators` holds the z-score arithmetic, `RollingFeatures` the window
aggregation, and `FeatureAssembler` picks the columns that leave the job.

## Configuration

Environment variables, read in `CryptoStreamJob`:

| variable | default | |
|---|---|---|
| `KAFKA_BOOTSTRAP_SERVERS` | `localhost:9092` | |
| `KAFKA_TOPIC` | `crypto-market` | |
| `OUTPUT_PATH` | `data/features` | where the Parquet goes |
| `CHECKPOINT_PATH` | `data/checkpoints/sentinel` | Spark's own state |

It reads from `latest`, so it does not replay a topic it was not running for.

## Build and test

```bash
mvn test
mvn package -DskipTests
```

The tests cover the z-score arithmetic, including the zero-deviation case, which
is the one that quietly poisons everything if it regresses.

## Why Java

The rest of the ML work is Python, but this is the part that has to keep up with a
stream, and Java is the language the Spark API is native to. It also keeps a hard
line between the Big Data half of the project and the Machine Learning half.
