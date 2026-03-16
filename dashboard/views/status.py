import streamlit as st
import plotly.graph_objects as go
from api_client import get_system_status, get_stats, get_model_info
from components.status_cards import render_service_cards
from components.pipeline_flow import render_pipeline_flow
from components.kpi_cards import render_kpi_cards


def render():
    st.markdown('<p class="section-title">System Status</p>', unsafe_allow_html=True)

    status = get_system_status()
    services = status.get("services", [])

    # Service cards
    render_service_cards(services)

    # Pipeline flow
    st.markdown('<p class="section-title">Pipeline</p>', unsafe_allow_html=True)
    render_pipeline_flow(services)

    # KPIs from stats
    stats = get_stats()
    if stats.get("total_predictions", 0) > 0:
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        render_kpi_cards(
            stats["total_predictions"],
            stats["total_anomalies"],
            stats["anomaly_rate"],
            stats["avg_score"],
        )

    # Model info (compact)
    st.markdown('<p class="section-title">Model</p>', unsafe_allow_html=True)
    info = get_model_info()
    if info.get("loaded"):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Type", info.get("model_type", "N/A"))
        c2.metric("Estimators", str(info.get("n_estimators", "N/A")))
        c3.metric("Contamination", f"{info.get('contamination', 0):.3f}")
        c4.metric("File Size", f"{info.get('model_file_size_kb', 0):.1f} KB")
    else:
        st.warning("Model not loaded yet.")
