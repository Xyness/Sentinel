import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from load_dataset import load_features
from preprocess import preprocess

MODEL_PATH = "models/isolation_forest.joblib"

def train():
    df = load_features()
    X, y = preprocess(df)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=200,
        max_samples="auto",
        contamination=0.01,
        random_state=42
    )

    model.fit(X_scaled)

    joblib.dump(
        {"model": model, "scaler": scaler},
        MODEL_PATH
    )

    print("✅ Isolation Forest trained and saved")

if __name__ == "__main__":
    train()
