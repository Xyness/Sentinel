import streamlit as st
from api_client import get_prediction
from components.charts import anomaly_gauge


PRESETS = {
    "Normal": {
        "z_score_price": 0.1,
        "z_score_log_return": 0.05,
        "z_score_volume": 0.2,
        "rolling_price_std": 0.002,
        "rolling_volume_std": 10.0,
    },
    "Price Spike": {
        "z_score_price": 4.5,
        "z_score_log_return": 3.8,
        "z_score_volume": 1.5,
        "rolling_price_std": 0.008,
        "rolling_volume_std": 25.0,
    },
    "Volume Spike": {
        "z_score_price": 0.5,
        "z_score_log_return": 0.3,
        "z_score_volume": 4.8,
        "rolling_price_std": 0.003,
        "rolling_volume_std": 45.0,
    },
    "Flash Crash": {
        "z_score_price": -4.2,
        "z_score_log_return": -4.5,
        "z_score_volume": 3.5,
        "rolling_price_std": 0.009,
        "rolling_volume_std": 40.0,
    },
}


def render():
    st.markdown('<p class="section-title">Manual Test</p>', unsafe_allow_html=True)

    col_input, col_result = st.columns([1, 1])

    with col_input:
        st.markdown("#### Feature Input")

        preset_name = st.selectbox("Quick Preset", ["Custom"] + list(PRESETS.keys()))
        preset = PRESETS.get(preset_name, {})

        symbol = st.selectbox("Cryptocurrency", ["BTC-USDT", "ETH-USDT", "BNB-USDT"])
        z_score_price = st.slider(
            "Z-score price", -5.0, 5.0,
            preset.get("z_score_price", 0.0), step=0.01,
        )
        z_score_log_return = st.slider(
            "Z-score log return", -5.0, 5.0,
            preset.get("z_score_log_return", 0.0), step=0.01,
        )
        z_score_volume = st.slider(
            "Z-score volume", -5.0, 5.0,
            preset.get("z_score_volume", 0.0), step=0.01,
        )
        rolling_price_std = st.slider(
            "Price volatility (std)", 0.0, 0.01,
            preset.get("rolling_price_std", 0.002), step=0.0001, format="%.4f",
        )
        rolling_volume_std = st.slider(
            "Volume volatility (std)", 0.0, 50.0,
            preset.get("rolling_volume_std", 10.0), step=0.5,
        )

        features = {
            "symbol": symbol,
            "z_score_price": z_score_price,
            "z_score_log_return": z_score_log_return,
            "z_score_volume": z_score_volume,
            "rolling_price_std": rolling_price_std,
            "rolling_volume_std": rolling_volume_std,
        }

        detect_clicked = st.button("Detect Anomaly", type="primary", use_container_width=True)

    with col_result:
        st.markdown("#### Detection Result")

        if detect_clicked:
            try:
                result = get_prediction(features)
                score = result["anomaly_score"]
                is_anom = result["is_anomaly"]

                gauge_rate = 100.0 if is_anom else max(0, min(100, abs(score) * 20))
                st.plotly_chart(anomaly_gauge(gauge_rate), use_container_width=True)

                m1, m2 = st.columns(2)
                m1.metric("Anomaly Score", f"{score:.4f}")
                m2.metric("Symbol", symbol)

                if is_anom:
                    st.error("ANOMALY DETECTED - Suspicious market behavior")
                else:
                    st.success("Normal behavior - No anomaly detected")

                st.json(features)

            except ConnectionError as e:
                st.error(f"Connection error: {e}")
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.info("Adjust the features and click Detect to analyze.")
