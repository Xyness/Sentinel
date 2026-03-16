# CryptoSentinel  
## Plateforme Big Data de Détection d’Anomalies sur Marchés Crypto en Temps Réel

**Auteur :** Xyness 

---

## Résumé

Les marchés de crypto-monnaies génèrent des flux massifs de données en continu
et présentent une forte volatilité. Cette combinaison rend la détection manuelle
d’anomalies particulièrement complexe.

Ce projet propose une plateforme Big Data capable de traiter des données
financières en temps réel et de détecter automatiquement des comportements
anormaux à l’aide de modèles de Machine Learning non supervisés.

L’architecture repose sur Apache Kafka pour l’ingestion, Spark Structured
Streaming pour le traitement temps réel, un modèle Isolation Forest pour
la détection d’anomalies, et un dashboard interactif pour la visualisation.

---

## 1. Introduction

La détection d’anomalies sur les marchés financiers est un enjeu majeur,
notamment dans un contexte de trading algorithmique et de surveillance
des manipulations de marché.

Les crypto-monnaies constituent un terrain d’étude particulièrement
intéressant en raison de leur forte volatilité, de l’absence de régulation
centralisée et du volume important de données générées en continu.

L’objectif de ce projet est de concevoir un système capable de :
- traiter des flux de données en temps réel,
- extraire des indicateurs financiers pertinents,
- détecter automatiquement des anomalies sans supervision,
- proposer une architecture scalable et reproductible.

---

## 2. Architecture générale

L’architecture du système est orientée streaming et repose sur une
séparation claire des responsabilités.

1. Génération de données crypto simulées
2. Ingestion temps réel avec Apache Kafka
3. Traitement et feature engineering avec Spark Structured Streaming
4. Inférence via un modèle de Machine Learning exposé par une API
5. Visualisation des anomalies via un dashboard

Cette architecture permet un découplage fort entre les composants,
facilitant la maintenance et les évolutions futures.

---

## 3. Données et simulation

Dans la version actuelle, les données sont simulées afin de garantir
la reproductibilité des expériences.

Les données générées incluent :
- prix
- volume
- volatilité
- anomalies contrôlées (pics, ruptures)

Cette approche permet d’évaluer précisément les performances du système
sans dépendre d’API externes.

---

## 4. Traitement Big Data en streaming

Apache Kafka est utilisé comme système d’ingestion afin de gérer les flux
de données en continu.

Spark Structured Streaming est employé pour :
- consommer les données depuis Kafka,
- appliquer des fenêtres temporelles,
- calculer des features financières en temps réel.

L’utilisation de Spark permet de bénéficier d’un moteur distribué robuste
et largement utilisé dans l’industrie.

---

## 5. Feature engineering financier

Les caractéristiques extraites incluent :
- z-score du prix
- z-score du volume
- volatilité glissante du prix
- volatilité glissante du volume

Ces features sont choisies pour leur interprétabilité et leur pertinence
dans un contexte de détection d’anomalies.

---

## 6. Détection d’anomalies par Machine Learning

Le projet adopte une approche non supervisée, plus réaliste dans un
contexte financier où les anomalies sont rares et mal définies.

Le modèle principal utilisé est l’Isolation Forest, choisi pour :
- sa robustesse,
- sa rapidité,
- son adéquation à la détection d’anomalies générales.

L’entraînement est réalisé en batch, tandis que l’inférence est effectuée
en temps réel.

---

## 7. API et visualisation

Le modèle de Machine Learning est exposé via une API FastAPI, permettant
une intégration simple avec les autres composants du système.

Un dashboard Streamlit permet de visualiser les scores d’anomalie et
de démontrer le fonctionnement du pipeline de bout en bout.

---

## 8. Déploiement et reproductibilité

L’ensemble de la plateforme est containerisé avec Docker et orchestré
via Docker Compose.

Une seule commande permet de lancer l’intégralité du système, garantissant
une reproductibilité complète.

---

## 9. Résultats et évaluation

L’évaluation repose sur l’injection d’anomalies connues dans les données
simulées.

Les résultats montrent que le système est capable d’identifier efficacement
des comportements atypiques, tout en maintenant un nombre raisonnable
de faux positifs.

---

## 10. Limites et perspectives

Les principales limites actuelles sont :
- l’utilisation de données simulées,
- l’absence de ré-entraînement automatique,
- un nombre limité de modèles testés.

Les perspectives incluent :
- l’ingestion de données réelles,
- l’ajout de nouveaux modèles,
- le déploiement dans un environnement cloud.

---

## Conclusion

Ce projet illustre la conception d’un système Big Data & IA complet,
allant de l’ingestion de données en streaming à la détection d’anomalies
et à leur visualisation.
