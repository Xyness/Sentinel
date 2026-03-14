import streamlit as st
import pandas as pd
from api_client import get_prediction

st.set_page_config(page_title="CryptoAnom Dashboard", layout="wide")

st.title("CryptoAnom - Crypto Anomaly Detection")

# Initialize session state for prediction history
if "history" not in st.session_state:
    st.session_state.history = []

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    symbol = st.selectbox("Cryptocurrency", ["BTC-USDT", "ETH-USDT", "BNB-USDT"])
    st.divider()
    st.caption("Adjust feature values and click Detect to run anomaly detection.")

# Main layout: two columns
col_input, col_result = st.columns([1, 1])

with col_input:
    st.subheader("Feature Input")

    z_score_price = st.slider("Z-score price", -5.0, 5.0, 0.0, step=0.01)
    z_score_volume = st.slider("Z-score volume", -5.0, 5.0, 0.0, step=0.01)
    rolling_price_std = st.slider("Price volatility (std)", 0.0, 0.01, 0.002, step=0.0001, format="%.4f")
    rolling_volume_std = st.slider("Volume volatility (std)", 0.0, 50.0, 10.0, step=0.5)

    features = {
        "symbol": symbol,
        "z_score_price": z_score_price,
        "z_score_volume": z_score_volume,
        "rolling_price_std": rolling_price_std,
        "rolling_volume_std": rolling_volume_std,
    }

    detect_clicked = st.button("Detect Anomaly", type="primary", use_container_width=True)

with col_result:
    st.subheader("Detection Result")

    if detect_clicked:
        try:
            result = get_prediction(features)

            st.session_state.history.append({
                "symbol": symbol,
                "z_price": z_score_price,
                "z_volume": z_score_volume,
                "price_std": rolling_price_std,
                "volume_std": rolling_volume_std,
                "score": result["anomaly_score"],
                "anomaly": result["is_anomaly"],
            })

            score_col, status_col = st.columns(2)

            with score_col:
                st.metric(label="Anomaly Score", value=f"{result['anomaly_score']:.4f}")

            with status_col:
                st.metric(label="Symbol", value=symbol)

            if result["is_anomaly"]:
                st.error("ANOMALY DETECTED - Suspicious market behavior")
            else:
                st.success("Normal behavior - No anomaly detected")

        except ConnectionError as e:
            st.error(f"Connection error: {e}")
        except TimeoutError as e:
            st.error(f"Timeout: {e}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")
    else:
        st.info("Adjust the features and click Detect to analyze.")

# Prediction history
st.divider()
st.subheader("Prediction History")

if st.session_state.history:
    df_history = pd.DataFrame(st.session_state.history)

    # Summary metrics
    total = len(df_history)
    anomalies = df_history["anomaly"].sum()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Predictions", total)
    m2.metric("Anomalies Detected", int(anomalies))
    m3.metric("Anomaly Rate", f"{anomalies / total * 100:.1f}%")
    m4.metric("Avg Score", f"{df_history['score'].mean():.4f}")

    # Score chart
    st.line_chart(df_history["score"], use_container_width=True)

    # History table
    with st.expander("Show detailed history"):
        st.dataframe(df_history, use_container_width=True)

    if st.button("Clear History"):
        st.session_state.history = []
        st.rerun()
else:
    st.caption("No predictions yet. Run a detection to see history.")
