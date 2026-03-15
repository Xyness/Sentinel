# CryptoAnom

**Real-time anomaly detection on cryptocurrency markets**

Big Data | Streaming | Machine Learning | Finance

---

## Overview

CryptoAnom is a Big Data platform that analyzes cryptocurrency market data streams in real time to automatically detect anomalous or suspicious behavior using unsupervised Machine Learning.

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
docker-compose up --build
```

### Run with real Binance market data

```bash
DATA_SOURCE=binance docker-compose up --build
```

### Access the services

| Service       | URL                          |
|---------------|------------------------------|
| Dashboard     | http://localhost:8501         |
| API           | http://localhost:8000         |
| API docs      | http://localhost:8000/docs    |
| Spark UI      | http://localhost:4040         |

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

- **Rolling statistics**: price mean/std, volume mean/std (1-minute tumbling windows)
- **Technical indicators**: price z-score, volume z-score
- **Division-by-zero protection**: returns 0 when standard deviation is 0

Output is written as Parquet files, partitioned by symbol.

### 3. ML Training

The Isolation Forest model trains on the computed features:

- **Simulated data**: train/test split (80/20) with classification report
- **Real data**: fully unsupervised training (no labels available)
- Features: `z_score_price`, `z_score_volume`, `rolling_price_std`, `rolling_volume_std`
- StandardScaler normalization
- 200 estimators, 1% contamination rate

### 4. API

FastAPI service exposing:

- `POST /predict` — anomaly detection on a feature vector
- `GET /health` — health check with model status
- Auto-generated Swagger docs at `/docs`

The API starts immediately and loads the model lazily once training completes.

### 5. Dashboard

Streamlit interface with:

- Feature input sliders for manual testing
- Anomaly score display with alerts
- Prediction history with metrics and charts

---

## Project Structure

```
CryptoAnom/
├── api/                        # FastAPI anomaly detection service
│   ├── main.py                 #   API routes (/predict, /health)
│   ├── model_loader.py         #   Model loading with lazy init
│   ├── schemas.py              #   Pydantic request/response models
│   └── requirements.txt
├── dashboard/                  # Streamlit visualization
│   ├── app.py                  #   Dashboard UI
│   ├── api_client.py           #   HTTP client for API
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
│   └── src/main/java/com/cryptoanom/
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
│   ├── choices-en.md
│   └── architecture.png
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
| Dashboard          | Streamlit                                     |
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
- Preprocessing (NaN handling, labeled/unlabeled modes)
- API (schemas validation, endpoints, error handling)
- Configuration (defaults, env overrides)

---

## Documentation

- [`docs/report.md`](docs/report.md) — Technical report
- [`docs/choices-en.md`](docs/choices-en.md) — Design decisions and justifications (English)
