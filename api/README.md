# API

Inference for the anomaly detector. It loads the Isolation Forest the training job
wrote and scores feature vectors over HTTP, which is how the terminal client and
anything else reaches the model without importing it.

## Endpoints

| endpoint | |
|---|---|
| `POST /predict` | score one feature vector |
| `GET /health` | whether the API is up, and whether a model is loaded |
| `GET /latest-predictions` | the recent predictions, filtered by symbol or by id |
| `GET /stats` | the buffer aggregated: per symbol, percentiles, feature stats |
| `GET /model-info` | what was loaded, and the scaler it was fitted with |
| `GET /system-status` | whether Spark, Kafka and Zookeeper answer |

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"symbol":"BTC-USDT","z_score_price":2.41,"z_score_log_return":1.02,
       "z_score_volume":3.12,"rolling_price_std":0.0018,"rolling_volume_std":12.4}'
```

```json
{"symbol": "BTC-USDT", "anomaly_score": -0.37, "is_anomaly": true}
```

All five features are required, in the shape Spark writes them: the z-scores for
price, log return and volume, and the rolling standard deviation of price and of
volume. `anomaly_score` is the raw `decision_function` output, so lower is more
atypical and the boundary sits at zero.

## The model

Loaded lazily rather than at startup. Training takes a few minutes on a cold stack,
so the API comes up first and reports `model_loaded: false` until the file appears,
instead of refusing to start and taking its own health check down with it. While it
waits, `/predict` answers 503 saying exactly that.

The anomalies the simulator injects are only ever used to evaluate the model. They
never reach training, which is unsupervised on both data sources.

## Prediction history

`/predict` keeps its last 500 results in memory, each with an id and a timestamp.
That is what `/latest-predictions` serves and what `sentinel feed` follows: `after`
takes an id and returns only what came after it, so a follower never prints the same
prediction twice.

It is a ring buffer in one process, so restarting the API empties it and starts
the ids over. A follower has to cope with that, which is why `sentinel feed`
filters on the id it last printed rather than trusting `after` alone.

What fills it is the scorer, which reads the feature rows Spark writes and posts
them here. `/predict` is otherwise open to anything that wants a score.

## Running it on its own

```bash
pip install -r requirements.txt
MODEL_PATH=../models/isolation_forest.joblib uvicorn main:app --port 8000
```

Swagger is at `/docs`.
