# CryptoAnom – Anomaly Detection API

Cette API fournit un service d’inférence en temps réel pour la détection
d’anomalies sur les marchés de crypto-monnaies.

Elle s’appuie sur un modèle de Machine Learning non supervisé
(Isolation Forest), entraîné hors ligne, et expose un endpoint REST permettant
de scorer des observations issues d’un pipeline Big Data en streaming.

## Objectifs

- Industrialiser l’inférence d’un modèle de détection d’anomalies
- Découpler le traitement Big Data (Spark) du Machine Learning
- Fournir une API REST simple, réutilisable et scalable
- Faciliter l’intégration avec un dashboard ou un système d’alerting

## Modèle de Machine Learning

Le modèle utilisé est un Isolation Forest, un algorithme de détection
d’anomalies non supervisé.

Il est entraîné en mode batch sur des données historiques et chargé
au démarrage de l’API pour effectuer des prédictions en temps réel.

Les anomalies injectées dans les données simulées servent uniquement
à l’évaluation du modèle et ne sont jamais utilisées lors de l’entraînement.

Features utilisées :
- z-score du prix
- z-score du volume
- volatilité du prix (écart-type glissant)
- volatilité du volume (écart-type glissant)

## Endpoint d’inférence

POST /predict

Exemple de requête JSON :

{
  "symbol": "BTC-USDT",
  "z_score_price": 2.41,
  "z_score_volume": 3.12,
  "rolling_price_std": 0.0018,
  "rolling_volume_std": 12.4
}

Exemple de réponse JSON :

{
  "symbol": "BTC-USDT",
  "anomaly_score": -0.37,
  "is_anomaly": true
}

Le champ anomaly_score correspond au score continu retourné par
Isolation Forest. Une valeur plus faible indique un comportement plus atypique.

## Architecture

Kafka → Spark Structured Streaming (Java)
           |
           | Feature engineering
           v
        FastAPI (API ML)
           |
           | Score d’anomalie
           v
        Dashboard / Stockage / Alertes

## Stack technique

- FastAPI
- scikit-learn
- joblib
- Pydantic

## Lancement de l’API

Installation des dépendances :

pip install -r requirements.txt

Démarrage du serveur :

uvicorn main:app --host 0.0.0.0 --port 8000

Documentation interactive :

http://localhost:8000/docs

## Remarques académiques

Ce module illustre :
- la séparation stricte entre entraînement et inférence
- une architecture modulaire et extensible
- l’industrialisation d’un modèle de Machine Learning
- une conception conforme aux standards des systèmes temps réel
