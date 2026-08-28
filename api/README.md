# API

Inference for the anomaly detector. It loads the Isolation Forest the training job
wrote and scores feature vectors over HTTP, which is how the terminal client and
anything else reaches the model without importing it.

## Endpoints

| endpoint | |
|---|---|
| `POST /predict` | score one feature vector |
| `POST /predict/batch` | score up to 500 of them in one call |
| `GET /health` | whether the API is up, and whether a model is loaded |
| `GET /latest-predictions` | the recent predictions, filtered by symbol or by id |
| `GET /stats` | aggregated: per symbol, percentiles, feature stats |
| `GET /model-info` | what was loaded, its scaler, and what it scored on the holdout |
| `GET /system-status` | whether Spark and Kafka answer |

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"symbol":"BTC-USDT","abs_return_max":0.115,"return_std":0.021,
       "price_range_rel":0.124,"volume_max_ratio":4.2,"volume_cv":1.3}'
```

```json
{"symbol": "BTC-USDT", "anomaly_score": -0.37, "is_anomaly": true}
```

All five features are required, in the shape Spark writes them. Every one of them
is a magnitude with a floor of zero: the largest absolute log return in the
window, the realised volatility, the relative price range, the largest trade over
the window's average, and the volume's coefficient of variation. `anomaly_score`
is the raw `decision_function` output, so lower is more atypical and the boundary
sits at zero.

`/predict/batch` takes `{"vectors": [...]}` and answers with one result per
vector in the order they were sent. A vector the schema refuses takes the whole
batch down with a 422: half a file scored and half rejected is a worse answer
than none.

## The model

Loaded lazily rather than at startup. Training takes a few minutes on a cold stack,
so the API comes up first and reports `model_loaded: false` until the file appears,
instead of refusing to start and taking its own health check down with it. While it
waits, `/predict` answers 503 saying exactly that.

Training refits as the feature store grows, so the file changes while the API is
up. Every call that touches the model checks its timestamp and reloads when it
moves. Without that the first model ever written would be the one served for the
life of the process and every refit after it would go nowhere.

The bundle also carries the feature names it was fitted on and the metrics the
training run measured, and `/model-info` serves both. The feature list comes off
the bundle rather than being hardcoded here, so a model fitted on something else
says so instead of mislabelling its own scaler.

The anomalies the simulator injects are only ever used to set the contamination
and to evaluate the model. They never reach the fit, which is unsupervised on
both data sources.

## Prediction history

Predictions go into SQLite. With `PREDICTIONS_DB` pointing at a volume they
outlive the process; with nothing set the database is in memory and behaves the
way the old ring buffer did.

Each row carries an id and a timestamp. `/latest-predictions` serves them oldest
first, and `after` takes an id and returns only what came after it, so a follower
never prints the same prediction twice.

The table is capped at `PREDICTIONS_MAX_ROWS` and pruned in batches rather than
on every insert. `/stats` aggregates the most recent `STATS_ROWS` of it: a week
of history makes a percentile that reacts to nothing, and this endpoint is about
the recent past.

What fills it is the scorer, which reads the feature rows Spark writes and posts
them here. `/predict` is otherwise open to anything that wants a score.

## Configuration

| variable | default | |
|---|---|---|
| `MODEL_PATH` | `models/isolation_forest.joblib` | the bundle to load and watch |
| `PREDICTIONS_DB` | `:memory:` | SQLite path for the history |
| `PREDICTIONS_MAX_ROWS` | `50000` | the cap on that table |
| `STATS_ROWS` | `5000` | how much of it `/stats` reads |

## Running it on its own

```bash
pip install -r requirements.txt
MODEL_PATH=../models/isolation_forest.joblib uvicorn main:app --port 8000
```

Swagger is at `/docs`.
