# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] - 2026-03-18

### Added

#### Data Pipeline
- Simulated market data generator with configurable anomaly injection (price spikes, volume spikes, flash crashes)
- Binance WebSocket connector for real-time trade streaming (no API key required)
- Unified event format across simulated and real data sources
- Configurable event frequency and anomaly probability via environment variables

#### Stream Processing
- Apache Spark Structured Streaming job consuming from Kafka
- Rolling window feature engineering (1-minute tumbling windows)
- Z-score computation for price, log-return, and volume
- Division-by-zero protection for standard deviation calculations
- Parquet output partitioned by symbol

#### Machine Learning
- Isolation Forest model training with scikit-learn
- StandardScaler normalization pipeline
- Train/test split with classification report (simulated data)
- Fully unsupervised training mode (real data)
- Automatic wait-for-data orchestration before training

#### API
- FastAPI service with Uvicorn
- `POST /predict` endpoint for real-time anomaly detection
- `GET /health` endpoint with model loading status
- `GET /latest-predictions` with symbol filtering and pagination
- `GET /system-status` for pipeline service health monitoring
- `GET /stats` for aggregated prediction statistics and per-symbol breakdown
- `GET /model-info` for model parameters and scaler values
- Lazy model loading (API starts before training completes)

#### Dashboard
- Streamlit 4-page dashboard with futuristic dark/neon theme
- System Status page with service health cards and pipeline flow diagram
- Live Feed page with auto-refresh, KPI cards, anomaly score timeline, and gauge
- Analytics page with per-symbol charts, correlation matrix, and CSV export
- Manual Test page with presets (Normal, Price Spike, Volume Spike, Flash Crash)

#### Infrastructure
- Full Docker Compose orchestration (7 services)
- Multi-stage Maven build for Spark
- Shared Docker volumes for features, checkpoints, and model data
- Environment-based configuration for all services

#### Tests
- Unit tests for market simulator, Binance connector, preprocessing, API, and configuration
- pytest setup with dedicated test requirements

#### Documentation
- Technical report (`docs/report.md`)
- Design decisions document (`docs/choices-en.md`)
- Comprehensive README with architecture diagram, usage guide, and API examples
