import streamlit as st
import random
from api_client import get_prediction

st.set_page_config(page_title="CryptoAnom Dashboard", layout="wide")

st.title("📈 CryptoAnom — Détection d’anomalies sur les cryptos")

# Sélection de la crypto
symbol = st.selectbox("Crypto-monnaie", ["BTC-USDT", "ETH-USDT", "BNB-USDT"])

st.subheader("Simulation de features (démo)")

# Les sliders génèrent des valeurs de features pour la démo
features = {
    "symbol": symbol,
    "z_score_price": st.slider(
        "Z-score prix", -5.0, 5.0, random.uniform(-1, 1)
    ),
    "z_score_volume": st.slider(
        "Z-score volume", -5.0, 5.0, random.uniform(-1, 1)
    ),
    "rolling_price_std": st.slider(
        "Volatilité prix", 0.0, 0.01, 0.002
    ),
    "rolling_volume_std": st.slider(
        "Volatilité volume", 0.0, 50.0, 10.0
    ),
}

# Bouton pour déclencher l'appel API
if st.button("🔍 Détecter anomalie"):
    result = get_prediction(features)

    st.metric(
        label="Score d’anomalie",
        value=round(result["anomaly_score"], 4)
    )

    if result["is_anomaly"]:
        st.error("🚨 Anomalie détectée !")
    else:
        st.success("✅ Comportement normal")
