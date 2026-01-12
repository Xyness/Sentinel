import requests

API_URL = "http://localhost:8000/predict"

def get_prediction(features: dict):
    """
    Envoie les features à l'API ML et récupère la prédiction.
    """
    response = requests.post(API_URL, json=features)
    response.raise_for_status()
    return response.json()
