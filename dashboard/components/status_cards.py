import streamlit as st


def render_service_cards(services: list):
    cols = st.columns(len(services))
    for col, svc in zip(cols, services):
        name = svc.get("name", "?")
        status = svc.get("status", "offline")
        latency = svc.get("response_time_ms", 0)
        details = svc.get("details")

        css_class = "online" if status == "online" else "offline"
        status_color = "#00ff88" if status == "online" else "#ff3366"
        dot = "●"

        with col:
            st.markdown(f"""
            <div class="service-card {css_class}">
                <p class="svc-name">{name}</p>
                <p class="svc-status" style="color: {status_color};">
                    {dot} {status.upper()}
                </p>
                <p class="svc-latency">{latency:.0f} ms</p>
                {f'<p class="svc-latency">{details[:60]}</p>' if details else ''}
            </div>
            """, unsafe_allow_html=True)
