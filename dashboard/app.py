import streamlit as st
from theme import inject_custom_css
from api_client import get_system_status
from components.header import render_header
from views import status, live_feed, analytics, manual_test

st.set_page_config(
    page_title="CryptoSentinel Dashboard",
    page_icon="⬡",
    layout="wide",
)
inject_custom_css()

# --- Health check (cached in session_state) ---
if "services" not in st.session_state:
    try:
        data = get_system_status()
        st.session_state["services"] = data.get("services", [])
    except Exception:
        st.session_state["services"] = []

api_online = any(
    s.get("name") == "API" and s.get("status") == "online"
    for s in st.session_state["services"]
)

render_header(api_online, services=st.session_state["services"])

# --- Navigation ---
PAGES = {
    "◉ System Status": status,
    "◎ Live Feed": live_feed,
    "■ Analytics": analytics,
    "▷ Manual Test": manual_test,
}

with st.sidebar:
    st.markdown("### Navigation")
    page_name = st.radio("Go to", list(PAGES.keys()), label_visibility="collapsed")

    st.divider()
    if st.button("Refresh Services", use_container_width=True):
        try:
            data = get_system_status()
            st.session_state["services"] = data.get("services", [])
        except Exception:
            pass
        st.rerun()

    st.divider()
    st.markdown(
        '<p style="font-size:0.7rem; color:#6b7280;">CryptoSentinel v3.0</p>',
        unsafe_allow_html=True,
    )

# --- Page dispatch ---
PAGES[page_name].render()
