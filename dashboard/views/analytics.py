import streamlit as st
import pandas as pd
from api_client import get_latest_predictions, get_stats
from components.charts import (
    per_symbol_bar_chart,
    feature_correlation_matrix,
    score_trend_line,
)
from components.data_table import render_data_table


def render():
    st.markdown('<p class="section-title">Analytics</p>', unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### Analytics Settings")
        symbol_filter = st.selectbox(
            "Symbol filter", ["All", "BTC-USDT", "ETH-USDT", "BNB-USDT"],
            key="analytics_symbol",
        )
        data_depth = st.slider("Data depth", 50, 500, 200, step=50, key="analytics_depth")
        anomalies_only = st.toggle("Anomalies only", value=False, key="analytics_anom")

    effective_symbol = None if symbol_filter == "All" else symbol_filter

    try:
        data = get_latest_predictions(limit=data_depth, symbol=effective_symbol)
    except Exception as e:
        st.error(f"Failed to fetch data: {e}")
        return

    if not data:
        st.info("No prediction data available yet.")
        return

    df = pd.DataFrame(data)
    stats = get_stats()

    # Per-symbol chart + correlation
    per_symbol = stats.get("per_symbol", {})
    col1, col2 = st.columns(2)
    with col1:
        if per_symbol:
            st.plotly_chart(per_symbol_bar_chart(per_symbol), use_container_width=True)
    with col2:
        st.plotly_chart(feature_correlation_matrix(df), use_container_width=True)

    # Score trend
    st.plotly_chart(score_trend_line(df), use_container_width=True)

    # Data table
    st.markdown("#### Data Table")
    display_df = df[df["is_anomaly"]].copy() if anomalies_only else df.copy()
    st.markdown(f"**{len(display_df)}** records")
    render_data_table(display_df)

    # CSV export
    csv = display_df.to_csv(index=False)
    st.download_button(
        label="Export CSV",
        data=csv,
        file_name="cryptosentinel_export.csv",
        mime="text/csv",
    )
