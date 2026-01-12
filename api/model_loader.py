import joblib

MODEL_PATH = "models/isolation_forest.joblib"

class AnomalyModel:
    def __init__(self):
        bundle = joblib.load(MODEL_PATH)
        self.model = bundle["model"]
        self.scaler = bundle["scaler"]

    def predict(self, features):
        X_scaled = self.scaler.transform([features])
        score = self.model.decision_function(X_scaled)[0]
        prediction = self.model.predict(X_scaled)[0]

        return score, prediction == -1
