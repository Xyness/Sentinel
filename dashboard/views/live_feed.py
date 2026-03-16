import streamlit as st
import pandas as pd
from api_client import get_latest_predictions
from components.kpi_cards import render_kpi_cards
from components.charts import anomaly_score_timeline, anomaly_gauge


def render():
    st.markdown('<p class="section-title">Live Anomaly Feed</p>', unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### Feed Settings")
        symbol_filter = st.selectbox(
            "Symbol filter", ["All", "BTC-USDT", "ETH-USDT", "BNB-USDT"],
            key="live_symbol",
        )
        refresh_interval = st.slider("Refresh interval (s)", 3, 30, 5, key="live_refresh")
        prediction_limit = st.slider("History depth", 50, 500, 100, step=50, key="live_depth")

    effective_symbol = None if symbol_filter == "All" else symbol_filter

    @st.fragment(run_every=f"{refresh_interval}s")
    def live_dashboard():
        try:
            data = get_latest_predictions(
                limit=prediction_limit, symbol=effective_symbol,
            )
        except Exception as e:
            st.error(f"Failed to fetch predictions: {e}")
            return

        if not data:
            st.info("No predictions yet. Waiting for data...")
            return

        df = pd.DataFrame(data)

        total = len(df)
        anomalies = int(df["is_anomaly"].sum())
        rate = anomalies / total * 100 if total > 0 else 0
        avg_score = df["anomaly_score"].mean()

        render_kpi_cards(total, anomalies, rate, avg_score)
        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

        # Timeline + Gauge
        col_timeline, col_gauge = st.columns([3, 1])
        with col_timeline:
            st.plotly_chart(anomaly_score_timeline(df), use_container_width=True)
        with col_gauge:
            st.plotly_chart(anomaly_gauge(rate), use_container_width=True)

        # Recent anomaly alerts
        recent_anomalies = df[df["is_anomaly"]].tail(5)
        if len(recent_anomalies) > 0:
            st.markdown("#### Recent Anomaly Alerts")
            for _, row in recent_anomalies.iterrows():
                st.markdown(
                    f'<div class="anomaly-alert">'
                    f'<strong>{row["symbol"]}</strong> &mdash; '
                    f'score: {row["anomaly_score"]:.4f} &nbsp;|&nbsp; '
                    f'z_price: {row["z_score_price"]:.2f} &nbsp;|&nbsp; '
                    f'z_log_ret: {row["z_score_log_return"]:.2f} &nbsp;|&nbsp; '
                    f'z_vol: {row["z_score_volume"]:.2f}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    live_dashboard()
