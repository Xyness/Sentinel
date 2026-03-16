import streamlit as st


def render_pipeline_flow(services: list):
    svc_map = {s["name"]: s["status"] for s in services}

    nodes = [
        ("Generator", "online"),  # always assumed online if pipeline runs
        ("Kafka", svc_map.get("Kafka", "offline")),
        ("Spark", svc_map.get("Spark", "offline")),
        ("ML Model", svc_map.get("API", "offline")),
        ("API", svc_map.get("API", "offline")),
        ("Dashboard", "online"),
    ]

    html_parts = []
    for i, (label, status) in enumerate(nodes):
        css_class = "online" if status == "online" else "offline"
        dot_color = "#00ff88" if status == "online" else "#ff3366"
        html_parts.append(
            f'<div class="pipeline-node {css_class}">'
            f'<span style="color:{dot_color};">●</span> {label}</div>'
        )
        if i < len(nodes) - 1:
            html_parts.append('<span class="pipeline-arrow">→</span>')

    st.markdown(
        f'<div class="pipeline-container">{"".join(html_parts)}</div>',
        unsafe_allow_html=True,
    )
