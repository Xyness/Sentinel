# Sentinel

[![CI](https://github.com/Xyness/Sentinel/actions/workflows/ci.yml/badge.svg)](https://github.com/Xyness/Sentinel/actions/workflows/ci.yml)

Streaming anomaly detection on crypto markets. Kafka carries the trades, Spark turns them into
per-window features, an Isolation Forest decides what looks wrong, and a terminal client shows you
what it flagged.

It runs on either a simulator with anomalies injected on purpose (useful because you know the
ground truth) or on live Binance trades over their public WebSocket.

![sentinel status](docs/demo.svg)

## Getting it up

Docker and Docker Compose for the pipeline, Python 3.11+ for the client.

```bash
docker compose up --build
pip install -e ./cli
```

For real market data instead of the simulator:

```bash
DATA_SOURCE=binance docker compose up --build
```

Give it two or three minutes on first start. Nothing is broken during that time: Spark has to
accumulate enough windows to write features, the training job waits for those files to show up,
and the API only picks up a model once training finishes. `sentinel status` says where it has
got to, and `docker compose logs -f` if you want the detail.

Then: API on [8000](http://localhost:8000) (`/docs` for Swagger), Spark UI on
[4040](http://localhost:4040). `docker compose down -v` when you're done and want the volumes
gone too.

## The pipeline

```
   generator --> Kafka --> Spark Streaming --> Parquet --> training --> model
  (sim or                  per-window        partitioned  Isolation    .joblib
   Binance)                aggregates        by symbol    Forest          |
                                                  |         (refits)      |
                                                  +--> scorer --> API <---+
                                                                   |
                                                                   +--> sentinel
                                                                        terminal
```

**Generating.** The simulator produces BTC/ETH/BNB against USDT with configurable rates of price
spikes, volume spikes and flash crashes, labelling each event so you can measure yourself
afterwards. The Binance connector streams real trades and needs no API key. Both emit the same
shape:

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

**Spark.** Structured Streaming reads the topic and aggregates each symbol over one-minute
tumbling windows, then turns those aggregates into five dimensionless features. Output is Parquet
partitioned by symbol. Windows holding fewer than `MIN_EVENTS_PER_WINDOW` events are dropped:
they're the partial ones at the start and end of a run, and they describe the sampling rather
than the market.

**Training.** Isolation Forest, 200 estimators, over the five features with a StandardScaler in
front. The contamination is read off the labels rather than pinned, which matters more than it
sounds like it does, see below. On simulated data there's an 80/20 split and a classification
report, since the labels exist. On real data it's fully unsupervised, because there's nothing to
score against. The job keeps running and refits every `RETRAIN_INTERVAL_SECONDS` as the feature
store grows, writing the bundle by rename so the API never reads one half-written.

**Scoring.** Something has to put live rows through the model. The scorer reads each Parquet file
Spark closes, pulls the five features out and posts the whole file to `/predict/batch` in one
call. Three symbols on one-minute windows is three predictions a minute, which is why it polls
rather than watches.

**Serving.** FastAPI comes up immediately and loads the model lazily, so the API is reachable
while training is still running and reports `model: not loaded` rather than refusing to start.
While it says that, the scorer holds onto the files it could not send instead of dropping them.
Once a model is loaded the API watches the file's timestamp, so a refit gets picked up without a
restart.

```bash
curl http://localhost:8000/health
curl http://localhost:8000/stats
curl http://localhost:8000/model-info
curl "http://localhost:8000/latest-predictions?limit=50&symbol=BTC-USDT"

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"symbol":"BTC-USDT","abs_return_max":0.115,"return_std":0.021,
       "price_range_rel":0.124,"volume_max_ratio":4.2,"volume_cv":1.3}'
```

`/system-status` reports on the other services too, which is what `sentinel status` prints.

## The features, and why they're these five

This is the part that has to be right, and it's the part that was wrong.

| feature | what it is |
|---|---|
| `abs_return_max` | largest absolute log return in the window |
| `return_std` | realised volatility over the window |
| `price_range_rel` | `(high - low) / mean price` |
| `volume_max_ratio` | largest single trade over the window's mean volume |
| `volume_cv` | volume deviation over mean volume |

All five are dimensionless and none of them care what order the events arrived in. Both of those
are deliberate.

**Dimensionless, because otherwise the model spends itself telling BTC from BNB.** An absolute
price deviation puts a 43,000 dollar pair two orders of magnitude away from a 320 dollar one
before anything unusual has happened, and one model over three symbols then separates symbols
instead of anomalies.

**Order independent, because an aggregation shuffles.** The features used to be built on
`last(price)`, which returns whichever row the executor happened to see last and not the last one
in time. Same input, different answer, depending on the plan.

### The z-score that hid what it measured

The old feature set z-scored the last price of a window against that same window's mean and
standard deviation. Work through what that does to a step of size `J` landing at event `k` of an
`n` event window:

```
mean        = p + J*(n-k)/n
last        = p + J
numerator   = J*k/n
deviation   = J*sqrt(k*(n-k))/n
z-score     = sqrt(k / (n-k))
```

`J` cancels. The z-score of a jump depends only on where in the window it landed, never on how
big it was, so a 15 % flash crash and a 0.3 % drift score identically. At the middle of a window
both come out at 1.0, which is about where an ordinary quiet minute sits. The anomaly inflates
the very deviation used to normalise it and hides itself. Worse, a spike anywhere but the last
second was invisible: the label said the minute was anomalous and the features only described its
final tick.

`abs_return_max` is `log(1.15) = 0.14` for that crash and `log(1.003) = 0.003` for the drift,
wherever in the window they land. That's the whole fix. `FeatureAssemblerTest.java` pins both
halves of it down.

### The other half: the rate has to match

The model is fitted on windows, not on events, and those are not the same rate. The simulator
draws once per event. At one event a second, a one-minute window is 60 draws, so a per-event
probability of 0.01 puts an anomaly in `1 - 0.99^60 = 45 %` of windows. Training was then told to
flag 1 % of rows while 45 % were labelled, which caps recall at about 2 % no matter how good the
features are.

Two changes. `ANOMALY_PROBABILITY` defaults to 0.00085, which is the solution of
`1 - (1-p)^60 = 0.05`, so about one minute in twenty per pair is anomalous. And training reads
the contamination off the labels instead of assuming it, clamped to `[0.005, 0.25]` because a
measured rate outside that says the generator wants turning down, not the model.

### What it actually scores

The model carries its own scorecard. Training measures precision, recall and f1 on the holdout,
puts them in the bundle, and the API serves them:

```bash
sentinel status              # prints the holdout under the model block
curl http://localhost:8000/model-info | jq .metrics
```

Numbers that only ever existed in a training log are numbers nobody can check, which is why they
travel with the model instead.

## The client

`sentinel` talks to the API over HTTP and to nothing else, so it runs from your machine against
the compose stack, or against an API anywhere else with `--api`.

```bash
sentinel status                                # what is up, and what the model looks like
sentinel status --details                      # and the scaler the model was fitted with

sentinel feed                                  # follow predictions as they land
sentinel feed --symbol BTC-USDT --anomalies    # one pair, only the flagged ones
sentinel feed --once --tail 50                 # the last 50, then stop

sentinel stats                                 # what has been scored, aggregated
sentinel stats --symbol ETH-USDT

sentinel predict --preset flash-crash          # score a feature vector by hand
sentinel predict --preset normal --volume-peak 9.5

sentinel export -f csv -o predictions.csv
```

**status** draws the services with their latency, the pipeline with a mark on each stage, and the
model that is loaded with what it scored on the holdout. The generator and the scorer answer on
nothing, so they are drawn unchecked rather than green: a stage nobody looked at is not a stage
that is up.

**feed** is a tail. One prediction per line, columns wide enough to stay aligned, anomalies the
only thing coloured, so it goes through `grep` and `awk` like anything else. Ctrl-C stops it and
prints what went past.

**stats** is the retrospective view: the per-symbol breakdown, the score trend as a sparkline,
the percentile spread and the feature table.

**predict** is the manual test, with the same four presets the dashboard had (`normal`,
`price-spike`, `volume-spike`, `flash-crash`). Any flag overrides the preset it started from, and
a vector with holes in it is an error rather than a set of zeros the model would happily score.

`--json` on any of them prints the raw payload instead of the rendered view:

```bash
sentinel stats --json | jq '.per_symbol'
sentinel feed --json | jq -c 'select(.is_anomaly)'
```

### Where the predictions come from

The scorer, on its own, at roughly three a minute. You can also push vectors in by hand, one at a
time or as a stream on stdin, which is what the presets are for:

```bash
jq -c '.[]' vectors.json | sentinel predict --stdin
```

Predictions go into SQLite on a volume rather than a buffer in the API process, so restarting the
API no longer throws away everything it ever flagged. The table is capped at
`PREDICTIONS_MAX_ROWS` and `/stats` aggregates the most recent `STATS_ROWS` of it.

### In a pipeline

```bash
sentinel predict --preset flash-crash --fail-on-anomaly --json
```

Exit codes: `0` fine, `1` error, `2` something was flagged, `130` interrupted. Output is plain
and uncoloured whenever it is not going to a terminal, and `--plain` forces that anywhere.

## Configuration

Environment variables, all with defaults that work under Compose:

| Variable | Default | |
|---|---|---|
| `DATA_SOURCE` | `simulated` | or `binance` |
| `KAFKA_BOOTSTRAP_SERVERS` | `localhost:9092` | |
| `KAFKA_TOPIC` | `crypto-market` | |
| `EVENT_FREQUENCY_SECONDS` | `1` | simulator only, takes a float |
| `ANOMALY_PROBABILITY` | `0.00085` | simulator only, per event |
| `WINDOW_DURATION` | `1 minute` | what Spark aggregates over |
| `MIN_EVENTS_PER_WINDOW` | `10` | below this a window is dropped |
| `MIN_PARQUET_FILES` | `3` | how much data before training starts |
| `MAX_WAIT_SECONDS` | `600` | give up waiting for it |
| `RETRAIN_INTERVAL_SECONDS` | `300` | 0 trains once and exits |
| `CONTAMINATION` | `0.05` | only used when there are no labels |
| `MODEL_PATH` / `FEATURES_PATH` | | where the model and features live |
| `PREDICTIONS_DB` | in memory | SQLite path for the prediction history |
| `PREDICTIONS_MAX_ROWS` | `50000` | the cap on that table |
| `STATS_ROWS` | `5000` | how much of it `/stats` aggregates |
| `API_BASE_URL` | `http://localhost:8000` | where the client and the scorer look for the API |
| `SCORE_INTERVAL_SECONDS` | `15` | how often the scorer looks for new feature files |

Six services come up: Kafka, the generator, Spark, the training job, the API and the scorer.
Kafka runs in KRaft mode, so there's no Zookeeper to run, wait for or report on. If you're coming
from a checkout that still had Zookeeper, run `docker compose down -v` first: the broker keeps a
cluster id next to its log and refuses to start against one written under the old setup. They share four
volumes: features written by Spark and read by both training and the scorer, Spark's checkpoints,
the model written by training and read by the API, and the API's prediction database. The client
is not one of them: it is a `pip install` on your machine that talks to port 8000.

## Tests

```bash
pip install -r tests/requirements.txt
pytest
```

Covers the simulator's event structure and log returns, the Binance connector's symbol mapping
and message parsing, preprocessing, the contamination arithmetic and the shape of the bundle,
the model reloading itself when training writes a new one, the prediction store, the retraining
loop, the API schemas and endpoints, config defaults and overrides, the scorer and the terminal
client. All of it offline: the scorer's tests write real Parquet into a temp directory and the
client's stand the API up in-process behind an `httpx` mock transport, so nothing binds a port.

The Spark side is JUnit, and it runs a real local SparkSession over batch DataFrames:

```bash
cd spark-java && mvn test
```

`FeatureAssemblerTest` is the one worth reading. It builds a window of events with a jump in the
middle of it and asserts the features see it, whatever position it lands in and whatever size it
is, which is exactly what the old feature set could not do.

CI runs both of those, then stands the whole stack up on twenty-second windows and waits for a
prediction to come out the far end. `scripts/smoke_test.py` is what it runs, and it works against
any running stack:

```bash
docker compose up -d --build
python scripts/smoke_test.py
```

Building the images only proves the Dockerfiles parse. Everything this project claims happens
between the services.

## Layout

`data-generator/` produces events, `spark-java/` is the Maven-built streaming job, `ml-python/`
trains and evaluates, `api/` serves predictions and stores them, `scorer/` feeds it the rows Spark
wrote, `cli/` is the terminal client, `docker/` holds the Dockerfiles and `tests/` the pytest
suite. `scripts/demo.py` regenerates the capture at the top of this file and
`scripts/smoke_test.py` is the end to end check.

The awkward seam is between Spark and everything downstream: they only talk through Parquet on a
shared volume, so the feature set is written down in `FeatureAssembler.java`, `preprocess.py`,
`store.py` and the scorer. Change one and `preprocess` will now tell you which, instead of dying
on a KeyError three frames down.
