# Changelog

Follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `scorer/`, which reads the feature rows Spark writes and posts them to `/predict`. Nothing was
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
- `tests/test_cli.py` and `tests/test_scorer.py`, offline like the rest: the API is stood up
  in-process behind an httpx mock transport and the scorer reads real Parquet out of a temp
  directory, so nothing binds a port.
- READMEs for `ml-python/` and `spark-java/`, which had none, and a `.gitattributes` so the tree
  stops mixing CRLF and LF.

### Changed

- Line endings normalised to LF across the tree, and `api/` moved to the modern typing syntax.
- `api/README.md` and `data-generator/README.md` rewritten in English like the rest of the docs.
  The API one listed four features where the model takes five, and its example payload would
  have been rejected.

### Removed

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

### Notes

Z-scores return 0 rather than NaN when the standard deviation is zero. A flat minute isn't an
anomaly, and a NaN there propagates through the whole feature vector.
