import numpy as np
from sklearn.metrics import classification_report
import joblib

from training.load_dataset import load_features
from training.preprocess import preprocess

MODEL_PATH = "models/isolation_forest.joblib"

def evaluate():
    df = load_features()
    X, y_true = preprocess(df)

    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    scaler = bundle["scaler"]

    X_scaled = scaler.transform(X)
    scores = model.predict(X_scaled)

    # IsolationForest : -1 = anomalie, 1 = normal
    y_pred = (scores == -1).astype(int)

    print(classification_report(y_true, y_pred))

if __name__ == "__main__":
    evaluate()
