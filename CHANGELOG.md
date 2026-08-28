# Changelog

Follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `POST /predict/batch`, which takes up to 500 vectors and answers with one result per vector in
  the order they were sent. The scorer now sends a whole Parquet file in one call instead of a
  round trip and a single-row scaler call per row.
- Predictions are stored in SQLite (`api/store.py`) on its own volume, so restarting the API no
  longer throws away everything it ever flagged. With no `PREDICTIONS_DB` set the database is in
  memory and behaves the way the ring buffer did. The table is capped at `PREDICTIONS_MAX_ROWS`
  and `/stats` aggregates the most recent `STATS_ROWS` of it.
- The model bundle carries the feature names it was fitted on and the metrics the training run
  measured. `/model-info` serves both, and `sentinel status` prints the holdout precision, recall
  and f1 under the model block. Numbers that only existed in a training log were numbers nobody
  could check.
- `scripts/smoke_test.py` and an end to end CI job: the whole stack comes up on twenty-second
  windows and the job waits for a prediction to come out the far end, then compares the feature
  list the model was fitted on against the one the pipeline expects. Building the images only
  proved the Dockerfiles parse.
- `WINDOW_DURATION` and `MIN_EVENTS_PER_WINDOW` on the Spark job. Windows holding fewer events
  than the floor are dropped: they're the partial ones at the ends of a run and they describe the
  sampling rather than the market.
- `FeatureAssemblerTest`, which builds a window of events with a jump in the middle of it and
  asserts the features see it at any position and any size, plus tests for the contamination
  arithmetic, the model reloading itself, the prediction store and the retraining loop.
- `scorer/`, which reads the feature rows Spark writes and posts them to the API. Nothing was
  doing that, so the prediction buffer stayed empty on a pipeline that was otherwise running:
  Spark wrote Parquet, training read it, and the model never saw a live row. The symbol comes
  off the partition directory, rows with nulls in them are dropped, and a file the API refused
  while training was still running is retried rather than lost.
- `cli/`, a terminal client: `sentinel status`, `feed`, `stats`, `predict` and `export`, drawn
  with Rich over httpx. `--json` on any of them prints the raw payload instead, and `--plain`
  drops the colour. Exit codes are `0` fine, `1` error, `2` something was flagged.
- `id` and `timestamp` on prediction history entries, and an `after` parameter on
  `/latest-predictions`. A client following the feed needs to know what it has already printed,
  and position in a ring buffer is not that.
- `scripts/demo.py`, which regenerates the terminal capture at the top of the README.
- READMEs for `ml-python/` and `spark-java/`, which had none, and a `.gitattributes` so the tree
  stops mixing CRLF and LF.

### Changed

- Kafka runs in KRaft mode. Zookeeper is gone, along with its container, its 512 MB, its probe in
  `/system-status` and the row it drew in `sentinel status`. Six services instead of seven.
  **Upgrading needs `docker compose down -v` once.** A broker's stored cluster id has to match the
  one it starts with, and the image declares `/var/lib/kafka/data` as a volume, so Compose was
  quietly carrying the Zookeeper-era one across recreations and Kafka refused to start on it. That
  volume is named now rather than anonymous, so it shows up in `docker volume ls` and `down -v`
  actually clears it.
- Training keeps running and refits every `RETRAIN_INTERVAL_SECONDS` rather than training once
  and exiting. It skips a round when the feature store hasn't moved, and a refit that throws is
  logged and slept off rather than taking the container down: the model on disk is still the last
  one that worked. `RETRAIN_INTERVAL_SECONDS=0` restores the old behaviour.
- The model bundle is written beside the live path and renamed over it, so the API cannot read
  one half-written.
- `ANOMALY_PROBABILITY` defaults to 0.00085 rather than 0.01, which is the per-event rate that
  puts an anomaly in 5 % of one-minute windows rather than 45 %. `EVENT_FREQUENCY_SECONDS` takes
  a float now, so a window can be filled in seconds.
- `/system-status` probes Spark and Kafka at the same time instead of one after another, so a
  stack that is entirely down answers in the longest single timeout rather than the sum of them.
- Dockerfiles copy `requirements.txt` and install before copying the source, so editing a file
  no longer reinstalls scikit-learn. CI runs the Python suite on 3.11 and 3.12, which is both
  ends of the version the README claims.
- Line endings normalised to LF across the tree, and `api/` moved to the modern typing syntax.
- `api/README.md` and `data-generator/README.md` rewritten in English like the rest of the docs.

### Fixed

- **The features could not see the anomalies they were built to find.** They were z-scores of a
  window's last price against that same window's mean and standard deviation. For a step of size
  `J` landing at event `k` of an `n` event window that works out to `sqrt(k/(n-k))`: `J` cancels,
  so a 15 % flash crash and a 0.3 % drift scored identically, and both landed near 1.0, which is
  where a quiet window sits. An anomaly inflated the very deviation used to normalise it. A spike
  anywhere but the last second of a window was invisible while the label still said the window
  was anomalous. Replaced with five dimensionless magnitudes over the whole window:
  `abs_return_max`, `return_std`, `price_range_rel`, `volume_max_ratio` and `volume_cv`.
- **The contamination did not match the label rate.** It was pinned at 1 % while the generator,
  drawing once per event, marked 45 % of one-minute windows anomalous, which capped recall at
  about 2 % whatever the features did. It is read off the labels now, clamped to
  `[0.005, 0.25]` with a warning when it clamps.
- `last()` in a windowed aggregation returns whichever row the executor saw last, not the last
  one in time. Every feature is an extremum, a sum or a deviation now, so the same input gives
  the same answer whatever the plan does.
- Absolute price and volume deviations put a 43,000 dollar pair two orders of magnitude away
  from a 320 dollar one before anything unusual happened, so one model over three symbols spent
  itself separating symbols. Every feature is a ratio now.
- The API never reloaded the model. `ensure_loaded` returned early once anything was loaded, so
  the first model ever written was served for the life of the process. It watches the file's
  timestamp now, which is what makes retraining worth doing.
- `evaluation/evaluate.py` re-derived `train_test_split(random_state=42)` and called the result
  "the same split as training". It wasn't: the feature store grows between the two runs, so the
  rows it called held out were mostly rows the model had trained on. It reports the holdout the
  bundle carries, and labels its own pass over the current data as in-sample.
- `preprocess` raises naming the columns it expected when the feature store drifts, instead of a
  KeyError three frames down.

### Removed

- Zookeeper, and `_check_zookeeper` with it.
- `docs/report.md` and `docs/choices-en.md`.
- The Streamlit dashboard: `dashboard/`, `docker/Dockerfile-dashboard` and the compose service
  behind it. Port 8501 is free, and its four pages are the four client commands.
- `kafka/`, a directory holding one README that promised topic scripts and held nothing.
- The Solana pre-confirmation flow pipeline: `solana-flow/`, `shred-ingest/`, `reports/`,
  `deploy/`, the `/flow/*` endpoints, the dashboard's Solana Flow page and the four flow test
  suites. CI drops the Rust job and the report step.

## [1.0.0] - 2026-03-18

First release. The whole pipeline runs end to end under Docker Compose.

### Added

- Market data generator with two modes: a simulator that injects labelled anomalies
  (price spikes, volume spikes, flash crashes) and a Binance WebSocket connector for live
  trades. Both emit the same event shape.
- Spark Structured Streaming job reading Kafka and computing rolling stats and z-scores over
  one-minute tumbling windows, written out as Parquet partitioned by symbol.
- Isolation Forest training with StandardScaler, 200 estimators and 1% contamination.
  Labelled train/test split on simulated data, fully unsupervised on real data.
- FastAPI service with `/predict`, `/health`, `/latest-predictions`, `/stats`, `/model-info`
  and `/system-status`. The model loads lazily so the API is up before training finishes.
- Streamlit dashboard: system status, live feed, analytics with CSV export, and a manual test
  page with presets.
- Docker Compose covering all seven services, with shared volumes for features, checkpoints
  and the trained model. Spark image is a multi-stage Maven build.
- pytest suite over the simulator, the Binance connector, preprocessing, the API and config.
- `docs/report.md` and `docs/choices-en.md`.
