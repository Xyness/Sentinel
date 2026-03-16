# CryptoSentinel

**Real-time anomaly detection on cryptocurrency markets**

Big Data | Streaming | Machine Learning | Finance

---

## Overview

CryptoSentinel is a Big Data platform that analyzes cryptocurrency market data streams in real time to automatically detect anomalous or suspicious behavior using unsupervised Machine Learning.

The system supports both **simulated data** (with injected anomalies for evaluation) and **real market data** from Binance via WebSocket.

---

## Architecture

```
                          +------------------+
                          |  Data Generator   |
                          |  (Simulated or    |
                          |   Binance Live)   |
                          +--------+---------+
                                   |
                                   v
                          +------------------+
                          |   Apache Kafka    |
                          |  (crypto-market)  |
                          +--------+---------+
                                   |
                                   v
                          +------------------+
                          |  Spark Streaming  |
                          |  (Java)           |
                          |  - Rolling stats  |
                          |  - Z-scores       |
                          +--------+---------+
                                   |
                                   v
                          +------------------+
                          |  Parquet Files    |
                          |  (partitioned by  |
                          |   symbol)         |
                          +--------+---------+
                                   |
                                   v
                          +------------------+
                          |  ML Training      |
                          |  Isolation Forest  |
                          |  (scikit-learn)   |
                          +--------+---------+
                                   |
                                   v
                          +------------------+
                          |  FastAPI          |
                          |  /predict /health |
                          |  /stats /model-info|
                          +--------+---------+
                                   |
                                   v
                          +------------------+
                          |  Streamlit        |
                          |  Dashboard        |
                          +------------------+
```

---

## Quick Start

### Prerequisites

- Docker & Docker Compose

### Run with simulated data (default)

```bash
docker compose up --build
```

### Run with real Binance market data

```bash
DATA_SOURCE=binance docker compose up --build
```

The full pipeline takes **2-3 minutes** to start (Spark needs to produce enough data, ML trains the model, then the API loads it).

Follow the startup progress with:

```bash
docker compose logs -f
```

### Access the services

| Service       | URL                          |
|---------------|------------------------------|
| Dashboard     | http://localhost:8501         |
| API           | http://localhost:8000         |
| API docs      | http://localhost:8000/docs    |
| Spark UI      | http://localhost:4040         |

### Stop

```bash
docker compose down          # stop all services
docker compose down -v       # stop + delete volumes (data)
```

---

## Usage

### Dashboard

Open http://localhost:8501. The sidebar on the left lets you navigate between 4 pages:

**◉ System Status** — Shows the health of each service (API, Spark, Kafka, Zookeeper) with online/offline cards and latency. Displays the pipeline flow diagram and current model parameters.

**◎ Live Feed** — Auto-refreshing anomaly monitor. Configure the symbol filter, refresh interval (3-30s) and history depth in the sidebar. Displays KPI cards, an anomaly score timeline, a gauge, and recent anomaly alerts.

**■ Analytics** — Per-symbol breakdown chart, feature correlation matrix, score trend line, filterable data table and CSV export. Toggle "Anomalies only" in the sidebar to focus on detected anomalies.

**▷ Manual Test** — Test anomaly detection manually. Pick a quick preset (Normal, Price Spike, Volume Spike, Flash Crash) to pre-fill the sliders, then click "Detect Anomaly" to see the result with a gauge and score.

### API

The API exposes these endpoints (Swagger docs at http://localhost:8000/docs):

```bash
# Health check
curl http://localhost:8000/health

# Anomaly prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"symbol":"BTC-USDT","z_score_price":4.5,"z_score_log_return":3.8,"z_score_volume":1.5,"rolling_price_std":0.008,"rolling_volume_std":25}'

# Recent predictions (with optional symbol filter)
curl "http://localhost:8000/latest-predictions?limit=50&symbol=BTC-USDT"

# Service health overview
curl http://localhost:8000/system-status

# Aggregated stats
curl http://localhost:8000/stats

# Model parameters
curl http://localhost:8000/model-info
```

---

## Data Pipeline

### 1. Data Generation

**Simulated mode** generates realistic crypto market events (BTC-USDT, ETH-USDT, BNB-USDT) with configurable anomaly injection (price spikes, volume spikes, flash crashes).

**Binance mode** streams real trades via WebSocket from Binance's public API (no API key required).

Both modes produce events in the same format:

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

### 2. Stream Processing (Spark)

Apache Spark Structured Streaming consumes from Kafka and computes rolling window features:

- **Rolling statistics**: price mean/std, log-return mean/std, volume mean/std (1-minute tumbling windows)
- **Technical indicators**: price z-score, log-return z-score, volume z-score
- **Division-by-zero protection**: returns 0 when standard deviation is 0

Output is written as Parquet files, partitioned by symbol.

### 3. ML Training

The Isolation Forest model trains on the computed features:

- **Simulated data**: train/test split (80/20) with classification report
- **Real data**: fully unsupervised training (no labels available)
- Features: `z_score_price`, `z_score_log_return`, `z_score_volume`, `rolling_price_std`, `rolling_volume_std`
- StandardScaler normalization
- 200 estimators, 1% contamination rate

### 4. API

FastAPI service exposing:

- `POST /predict` — anomaly detection on a feature vector
- `GET /health` — health check with model status
- `GET /latest-predictions` — recent prediction history with optional symbol filter
- `GET /system-status` — health of all pipeline services
- `GET /stats` — aggregated prediction statistics
- `GET /model-info` — loaded model parameters and scaler values

The API starts immediately and loads the model lazily once training completes.

### 5. Dashboard

Streamlit interface with 4 pages: System Status, Live Feed, Analytics, and Manual Test (see [Usage](#usage) above).

---

## Project Structure

```
CryptoSentinel/
├── api/                        # FastAPI anomaly detection service
│   ├── main.py                 #   API routes
│   ├── model_loader.py         #   Model loading with lazy init
│   ├── schemas.py              #   Pydantic request/response models
│   └── requirements.txt
├── dashboard/                  # Streamlit visualization
│   ├── app.py                  #   Main app with 4-page navigation
│   ├── api_client.py           #   HTTP client for API
│   ├── theme.py                #   Futuristic dark/neon theme
│   ├── components/             #   UI components
│   │   ├── header.py           #     Header with multi-service status
│   │   ├── kpi_cards.py        #     Glassmorphism KPI cards
│   │   ├── charts.py           #     Plotly charts
│   │   ├── status_cards.py     #     Service status cards
│   │   ├── pipeline_flow.py    #     Pipeline flow diagram
│   │   └── data_table.py       #     Filterable data table
│   ├── views/                  #   Page modules
│   │   ├── status.py           #     System Status page
│   │   ├── live_feed.py        #     Live Feed page
│   │   ├── analytics.py        #     Analytics page
│   │   └── manual_test.py      #     Manual Test page
│   └── requirements.txt
├── data-generator/             # Market data producers
│   ├── generator.py            #   Main entry (dispatches sim/real)
│   ├── market_simulator.py     #   Simulated price dynamics (GBM)
│   ├── binance_connector.py    #   Binance WebSocket connector
│   ├── config.py               #   Configuration (env vars)
│   └── requirements.txt
├── ml-python/                  # ML training & evaluation
│   ├── training/
│   │   ├── train_isolation_forest.py
│   │   ├── load_dataset.py
│   │   └── preprocess.py
│   ├── evaluation/
│   │   └── evaluate.py
│   ├── train_runner.py         #   Wait-for-data + train orchestrator
│   └── requirements.txt
├── spark-java/                 # Spark Structured Streaming
│   ├── pom.xml                 #   Maven config (Spark 3.5, Kafka)
│   └── src/main/java/com/cryptosentinel/
│       ├── streaming/
│       │   └── CryptoStreamJob.java
│       └── features/
│           ├── FeatureAssembler.java
│           ├── RollingFeatures.java
│           └── TechnicalIndicators.java
├── docker/                     # Dockerfiles
│   ├── Dockerfile-api
│   ├── Dockerfile-dashboard
│   ├── Dockerfile-generator
│   ├── Dockerfile-ml
│   └── Dockerfile-spark        #   Multi-stage Maven build
├── tests/                      # Unit tests (pytest)
│   ├── requirements.txt
│   ├── test_api.py
│   ├── test_binance_connector.py
│   ├── test_config.py
│   ├── test_market_simulator.py
│   └── test_preprocess.py
├── docs/
│   ├── report.md
│   └── choices-en.md
├── docker-compose.yml          # Full orchestration (7 services)
└── pytest.ini
```

---

## Tech Stack

| Layer              | Technology                                    |
|--------------------|-----------------------------------------------|
| Message Broker     | Apache Kafka 3.7 (Confluent 7.7.1)             |
| Stream Processing  | Apache Spark Structured Streaming 3.5 (Java)  |
| Storage            | Parquet (columnar, partitioned)                |
| ML                 | scikit-learn (Isolation Forest)                |
| API                | FastAPI + Uvicorn                              |
| Dashboard          | Streamlit + Plotly                             |
| Real Data          | Binance WebSocket API                         |
| Infrastructure     | Docker, Docker Compose                        |
| Build              | Maven (Java), pip (Python)                    |

---

## Configuration

All services are configurable via environment variables:

| Variable                  | Default           | Description                        |
|---------------------------|-------------------|------------------------------------|
| `DATA_SOURCE`             | `simulated`       | `simulated` or `binance`           |
| `KAFKA_BOOTSTRAP_SERVERS` | `localhost:9092`  | Kafka broker address               |
| `KAFKA_TOPIC`             | `crypto-market`   | Kafka topic name                   |
| `EVENT_FREQUENCY_SECONDS` | `1`               | Simulated event interval           |
| `ANOMALY_PROBABILITY`     | `0.01`            | Simulated anomaly rate             |
| `MODEL_PATH`              | `models/...`      | Path to saved model bundle         |
| `FEATURES_PATH`           | `data/features`   | Parquet features directory         |
| `API_BASE_URL`            | `http://localhost:8000` | API URL for dashboard        |
| `MIN_PARQUET_FILES`       | `3`               | Min files before training starts   |
| `MAX_WAIT_SECONDS`        | `600`             | Training timeout                   |

---

## Docker Services

| Service          | Port  | Description                                |
|------------------|-------|--------------------------------------------|
| zookeeper        | 2181  | Kafka coordinator                          |
| kafka            | 9092  | Message broker                             |
| data-generator   | —     | Produces market events to Kafka            |
| spark            | 4040  | Stream processing, feature engineering     |
| ml-training      | —     | Waits for data, trains model, exits        |
| api              | 8000  | ML inference API                           |
| dashboard        | 8501  | Web UI                                     |

Shared Docker volumes:
- `features-data`: Spark writes, ML training reads
- `checkpoint-data`: Spark streaming checkpoints
- `model-data`: ML training writes, API reads

---

## Testing

```bash
pip install -r tests/requirements.txt
pytest
```

Tests cover:
- Market simulator (event structure, anomalies, log returns)
- Binance connector (symbol mapping, message parsing, log returns)
- Preprocessing (NaN handling, labeled/unlabeled modes, 5 features)
- API (schemas validation, endpoints, error handling, prediction history)
- Configuration (defaults, env overrides)

---

## Documentation

- [`docs/report.md`](docs/report.md) — Technical report
- [`docs/choices-en.md`](docs/choices-en.md) — Design decisions and justifications (English)
