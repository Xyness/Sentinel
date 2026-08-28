# Sentinel: technical and methodological choices

This document details all the technical, methodological and architectural
choices made in the Sentinel project.

The goal is to demonstrate a deep understanding of Big Data, Machine Learning
and distributed systems challenges, as well as the ability to design
a coherent and well-justified architecture.

---

## 1. Domain choice: cryptocurrency markets

Cryptocurrency markets present several characteristics that are particularly
interesting for a Big Data & AI project:

- continuous, high-frequency data streams
- high volatility
- frequent atypical behaviors
- growing academic and industrial interest

This context makes anomaly detection particularly relevant,
while remaining credible and realistic for a real-world application.

---

## 2. Simulated data: why not use real data?

The project initially uses simulated crypto data.

### Justifications:

- guarantee full reproducibility of experiments
- precisely control the occurrence of anomalies
- avoid latency, quota and external API dependency issues
- facilitate quantitative evaluation of models

The injected anomalies (volume spikes, trend breaks, abnormal volatility)
provide a pseudo "ground truth" for evaluating system performance.

A real-data mode using the Binance WebSocket API has been implemented,
allowing the system to switch between simulated and live market data
via the `DATA_SOURCE` environment variable.

---

## 3. Streaming-oriented architecture

The core of the project relies on a real-time data processing architecture.

### Why streaming?

- financial markets produce data continuously
- anomalies must be detected as early as possible
- pure batch processing is not suited for real-time use cases

Streaming allows processing data on the fly while maintaining
scalability guarantees.

---

## 4. Apache Kafka as ingestion system

Apache Kafka is used as a message broker for data ingestion.

### Justifications:

- high fault tolerance
- decoupling between producers and consumers
- native real-time stream management
- widely used industry standard

Kafka provides a clear separation between data generation
and analytical processing, which improves the overall system robustness.

---

## 5. Spark Structured Streaming

Data processing is performed with Spark Structured Streaming.

### Why Spark?

- mature and proven distributed engine
- native integration with Kafka
- unified batch / streaming API
- automatic state and time window management

### Why Structured Streaming?

- declarative model based on DataFrames
- better readability than Spark Streaming (DStreams)
- processing guarantees (at-least-once, exactly-once depending on configuration)

Spark enables continuous computation of financial features
(moving averages, volatility, z-score) directly on the streams.

---

## 6. Choice of Java for Spark

Spark Structured Streaming is implemented in Java.

### Justifications:

- alignment with Master's in Computer Science curriculum
- high performance
- static typing and software rigor
- strong proximity to the native Spark ecosystem

Python remains used for Machine Learning, where it is most relevant.

---

## 7. Financial feature engineering

The extracted features are intentionally simple but robust:

- price z-score
- volume z-score
- rolling price volatility
- rolling volume volatility

### Why these features?

- interpretable
- well-known in finance
- suited for general anomaly detection
- efficiently computable in streaming

The goal is not to predict a price, but to characterize abnormal behavior
compared to a "normal" regime.

---

## 8. Unsupervised anomaly detection

The project adopts an unsupervised approach.

### Justifications:

- absence of reliable labels in finance
- anomalies are rare and poorly defined
- more realistic approach in production

The notion of anomaly is defined as a statistically atypical observation
compared to historical behavior.

---

## 9. Isolation Forest as the primary model

Isolation Forest is used as the primary anomaly detection model.

### Justifications:

- algorithm specifically designed for anomaly detection
- no labels required
- robust in high dimensions
- fast to train and infer

Isolation Forest relies on the idea that anomalies are easier
to isolate than normal data points.

An autoencoder is considered as an optional comparative model.

---

## 10. Batch training and streaming inference

The system adopts a clear separation:

- model training in batch (offline)
- inference in real-time (online)

### Why this choice?

- model stability in production
- controlled computational cost
- classic industrial architecture
- ease of monitoring and model updates

The model is loaded at API startup and used solely for scoring.

---

## 11. FastAPI for Machine Learning API

FastAPI is used to expose the Machine Learning model.

### Justifications:

- high performance
- simple implementation
- automatic documentation (OpenAPI)
- clear decoupling between Big Data and ML

This API allows Spark (or other services) to query the model
without direct dependency on the Python code.

---

## 12. A terminal client rather than a web dashboard

Detected anomalies are read through a command line client built on Rich.

### Justifications:

- nothing to serve, nothing to host, nothing left running
- the output composes: a feed that is one line per prediction goes through
  `grep`, `awk` and `jq` like any other stream
- a command that cannot answer exits non-zero, which a page cannot do
- the same client works against a local stack or a remote API

An earlier version used a Streamlit dashboard. It was quicker to build and it
demonstrated well, but it added a seventh service to the compose file, it
turned an outage into a page of zeros, and none of what it printed could be
piped into anything. The client is a `pip install` against port 8000 instead.

---

## 13. Batch training, streaming scoring

Training happens once, offline. Scoring happens continuously, on the rows the
streaming job produces.

### Justifications:

- the two have different failure modes and different resource profiles
- a scorer that only speaks HTTP does not need the model in its own process
- the feature store is the contract between them, so either side can be
  replaced without touching the other

The scorer is deliberately dumb: poll for files the streaming job has closed,
post each row, remember what it has sent. It holds nothing the pipeline could
not rebuild, which is why losing it costs a restart and nothing else.

---

## 14. Docker and Docker Compose

The entire system is containerized with Docker and orchestrated with Docker Compose.

### Justifications:

- full reproducibility
- component isolation
- ease of deployment
- industry standard

A single command launches the entire pipeline.

---

## 15. Current project limitations

- single primary model
- no automatic retraining
- no advanced long-term storage
- predictions live in a 500-entry in-memory buffer, so a restart loses them

These limitations are acknowledged and clearly identified.

---

## 16. Future perspectives

- comparison with other models
- advanced alerting
- analytical storage (Delta Lake, Elasticsearch)
- cloud deployment

---

## Conclusion

Sentinel illustrates the design of a complete, coherent and realistic
Big Data & AI system, spanning from streaming data ingestion
to anomaly detection and reading the results back.

Each choice was guided by a trade-off between academic rigor,
industrial realism and time constraints.
