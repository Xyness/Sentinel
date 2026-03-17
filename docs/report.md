# Sentinel
## Big Data Platform for Real-Time Anomaly Detection on Crypto Markets

**Author:** Xyness

---

## Abstract

Cryptocurrency markets generate massive, continuous data streams and exhibit
high volatility. This combination makes manual anomaly detection particularly
challenging.

This project proposes a Big Data platform capable of processing financial data
in real time and automatically detecting abnormal behavior using unsupervised
Machine Learning models.

The architecture relies on Apache Kafka for ingestion, Spark Structured
Streaming for real-time processing, an Isolation Forest model for anomaly
detection, and an interactive dashboard for visualization.

---

## 1. Introduction

Anomaly detection in financial markets is a major challenge, especially in
the context of algorithmic trading and market manipulation surveillance.

Cryptocurrencies are a particularly interesting field of study due to their
high volatility, lack of centralized regulation, and the large volume of
data generated continuously.

The goal of this project is to design a system capable of:
- processing data streams in real time,
- extracting relevant financial indicators,
- automatically detecting anomalies without supervision,
- providing a scalable and reproducible architecture.

---

## 2. General Architecture

The system architecture is streaming-oriented and relies on a clear
separation of concerns.

1. Simulated (or real) crypto data generation
2. Real-time ingestion with Apache Kafka
3. Processing and feature engineering with Spark Structured Streaming
4. Inference via a Machine Learning model exposed through an API
5. Anomaly visualization through a dashboard

This architecture enables strong decoupling between components, facilitating
maintenance and future evolution.

---

## 3. Data and Simulation

The platform supports two data modes:

**Simulated mode** generates realistic market events with controlled anomaly
injection (price spikes, volume spikes, flash crashes), enabling precise
evaluation of system performance without relying on external APIs.

**Binance mode** streams real trades via WebSocket from Binance's public API
(no API key required), providing genuine market conditions.

Both modes produce events in the same format including: price, volume,
log return, and anomaly labels (when available).

---

## 4. Big Data Stream Processing

Apache Kafka is used as the ingestion system to handle continuous data
streams.

Spark Structured Streaming is employed to:
- consume data from Kafka,
- apply temporal windows,
- compute financial features in real time.

Using Spark provides a robust, distributed engine widely adopted in
industry.

---

## 5. Financial Feature Engineering

The extracted features include:
- price z-score
- log-return z-score
- volume z-score
- rolling price standard deviation
- rolling volume standard deviation

These features are chosen for their interpretability and relevance in an
anomaly detection context.

---

## 6. Anomaly Detection with Machine Learning

The project adopts an unsupervised approach, which is more realistic in a
financial context where anomalies are rare and poorly defined.

The primary model used is the Isolation Forest, chosen for:
- its robustness,
- its speed,
- its suitability for general anomaly detection.

Training is performed in batch, while inference is carried out in real time.

---

## 7. API and Visualization

The Machine Learning model is exposed via a FastAPI service, enabling simple
integration with the other system components.

The API provides endpoints for:
- anomaly prediction on feature vectors,
- prediction history retrieval,
- system-wide service health monitoring,
- aggregated prediction statistics,
- loaded model parameters and metadata.

A Streamlit dashboard provides a command center with four pages: system
status monitoring, live anomaly feed, analytics with data export, and
manual testing with presets.

---

## 8. Deployment and Reproducibility

The entire platform is containerized with Docker and orchestrated via
Docker Compose.

A single command launches the complete system, ensuring full
reproducibility.

---

## 9. Results and Evaluation

Evaluation relies on injecting known anomalies into simulated data.

Results show that the system is capable of effectively identifying atypical
behavior while maintaining a reasonable number of false positives.

---

## 10. Limitations and Future Work

Current limitations include:
- limited number of models tested,
- no automatic retraining,
- evaluation primarily on simulated data.

Future directions include:
- adding new models (autoencoders, DBSCAN),
- automatic model retraining on drift detection,
- cloud deployment.

---

## Conclusion

This project demonstrates the design of a complete Big Data & AI system,
from streaming data ingestion to anomaly detection and visualization.
