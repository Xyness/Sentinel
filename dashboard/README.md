# Sentinel Dashboard

Ce dashboard permet de visualiser les anomalies détectées par le modèle
Isolation Forest en temps réel.

Il consomme l'API FastAPI (module `api/`) et affiche pour chaque crypto-monnaie
le score d’anomalie ainsi qu’une alerte visuelle.

---

## Lancement du dashboard

Installer les dépendances :
```bash
pip install -r requirements.txt
```

Lancer l'application :
```bash
streamlit run app.py
```

L’interface sera accessible dans le navigateur à l’adresse
```bash
http://localhost:8501
```