import streamlit as st


def render_kpi_cards(total: int, anomalies: int, rate: float, avg_score: float):
    cols = st.columns(4)

    cards = [
        ("Total Predictions", f"{total:,}", "#00ffff"),
        ("Anomalies Detected", f"{anomalies:,}", "#ff3366"),
        ("Anomaly Rate", f"{rate:.1f}%", "#8b5cf6"),
        ("Avg Score", f"{avg_score:.4f}", "#00ff88"),
    ]

    for col, (label, value, color) in zip(cols, cards):
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <p class="kpi-label">{label}</p>
                <p class="kpi-value" style="color: {color}; text-shadow: 0 0 10px {color}40;">
                    {value}
                </p>
            </div>
            """, unsafe_allow_html=True)
