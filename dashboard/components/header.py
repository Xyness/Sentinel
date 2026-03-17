import streamlit as st


def render_header(api_online: bool, services: list = None):
    # Build status dots for all services (or just API)
    if services:
        dots_html = ""
        for svc in services:
            name = svc.get("name", "?")
            status = svc.get("status", "offline")
            css = "online" if status == "online" else "offline"
            color = "#00ff88" if status == "online" else "#ff3366"
            dots_html += (
                f'<span style="margin-left:12px; display:inline-flex; align-items:center;">'
                f'<span class="status-dot {css}"></span>'
                f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:0.7rem;color:{color};">'
                f'{name}</span></span>'
            )
    else:
        status_class = "online" if api_online else "offline"
        status_text = "API Online" if api_online else "API Offline"
        color = "#00ff88" if api_online else "#ff3366"
        dots_html = (
            f'<span class="status-dot {status_class}"></span>'
            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:0.8rem;color:{color};">'
            f'{status_text}</span>'
        )

    st.markdown(f"""
    <div class="sentinel-header">
        <div>
            <p class="sentinel-title">Sentinel</p>
            <p class="sentinel-subtitle">Real-time Anomaly Detection Command Center</p>
        </div>
        <div style="margin-left: auto; display: flex; align-items: center; flex-wrap: wrap;">
            {dots_html}
        </div>
    </div>
    """, unsafe_allow_html=True)
