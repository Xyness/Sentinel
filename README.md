# CryptoAnom
### Détection d’Anomalies sur Marchés Crypto en Temps Réel
**Big Data • Streaming • Machine Learning • Finance**

---

## 📌 Présentation
CryptoAnom est une plateforme Big Data conçue pour analyser en temps réel des flux
de données financières issues des marchés de crypto-monnaies, dans le but de détecter
automatiquement des comportements anormaux ou potentiellement frauduleux à l’aide
de techniques de Machine Learning non supervisées.

Ce projet a été développé dans le cadre **d'étude des marchés financiers et d'analyses d'actifs numériques**.

---

## 🎯 Problématique
Les marchés de crypto-monnaies génèrent des volumes massifs de données en continu
et présentent une forte volatilité. Cette combinaison rend la détection manuelle
d’anomalies (manipulations de marché, comportements atypiques) particulièrement
complexe.

L’objectif est donc de concevoir un système capable de :
- traiter des données financières en **temps réel**,
- extraire des **indicateurs pertinents**,
- détecter automatiquement des anomalies **sans supervision**,
- tout en étant **scalable** et **reproductible**.

---

## 🧱 Architecture Générale
Le système repose sur une architecture distribuée orientée streaming :

1. Génération ou ingestion de données crypto
2. Ingestion temps réel avec Apache Kafka
3. Traitement avec Spark Structured Streaming (Java)
4. Feature engineering financier
5. Détection d’anomalies par modèles ML
6. Stockage des résultats
7. Exposition via API et visualisation

*(Un schéma détaillé est disponible dans le dossier `docs/`.)*

---

## 📊 Données
Dans la version actuelle :
- Données crypto **simulées** réalistes (prix, volumes, volatilité)
- Génération d’anomalies contrôlées (pics de volume, ruptures de tendance)

Cette approche permet :
- une évaluation rigoureuse,
- une reproductibilité complète,
- une future extension vers des données réelles (API Binance).

---

## 🧠 Feature Engineering
Les caractéristiques extraites incluent notamment :
- variations de prix (Δp)
- volumes anormaux
- volatilité glissante
- moyennes et écarts-types mobiles
- indicateurs financiers (z-score, RSI, MACD)

Ces features sont calculées en streaming à l’aide de Spark.

---

## 🤖 Modèles de Machine Learning
Le projet utilise des modèles **non supervisés**, adaptés à la détection d’anomalies :

- **Isolation Forest** (modèle principal)
- Autoencoder (comparaison optionnelle)

L’entraînement est effectué en mode batch, tandis que l’inférence est réalisée
en temps réel sur les flux de données.

---

## 📈 Évaluation
L’évaluation repose sur :
- l’injection d’anomalies connues,
- des métriques quantitatives (precision, recall),
- une analyse qualitative des faux positifs / faux négatifs.

Cette méthodologie permet de mesurer la robustesse du système face à la volatilité
des marchés.

---

## 🛠️ Stack Technique
**Big Data**
- Apache Kafka
- Apache Spark Structured Streaming (Java)
- Parquet / Delta Lake

**Machine Learning**
- Python
- scikit-learn / PyTorch

**Backend & Visualisation**
- FastAPI
- PostgreSQL / Elasticsearch
- Dashboard interactif

**Infrastructure**
- Docker & Docker Compose

---

## 📁 Structure du Projet
```text
crypto-anomaly-detection/
├── data-generator/
├── kafka/
├── spark-java/
│ ├── streaming/
│ └── features/
├── ml-python/
│ ├── training/
│ └── evaluation/
├── api/
├── dashboard/
├── docs/
│ └── report.pdf
└── docker-compose.yml