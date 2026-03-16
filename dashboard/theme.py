import streamlit as st

# Color palette
BG_PRIMARY = "#0a0a0f"
BG_CARD = "rgba(15, 15, 25, 0.8)"
ACCENT_CYAN = "#00ffff"
ACCENT_VIOLET = "#8b5cf6"
ACCENT_GREEN = "#00ff88"
ACCENT_RED = "#ff3366"
TEXT_PRIMARY = "#e0e0e0"
TEXT_MUTED = "#6b7280"
BORDER_COLOR = "rgba(255,255,255,0.08)"


def inject_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* Global */
    .stApp {
        background: linear-gradient(135deg, #0a0a0f 0%, #0d0d1a 50%, #0a0a0f 100%);
        color: #e0e0e0;
        font-family: 'Inter', sans-serif;
    }

    /* Hide default Streamlit header/footer */
    header[data-testid="stHeader"] {
        background: transparent;
    }
    .stDeployButton { display: none; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d0d1a 0%, #0a0a12 100%);
        border-right: 1px solid rgba(255,255,255,0.05);
    }
    section[data-testid="stSidebar"] .stRadio label,
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stSlider label {
        color: #e0e0e0 !important;
    }

    /* Glass card */
    .glass-card {
        background: rgba(15, 15, 25, 0.8);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    .glass-card:hover {
        border-color: rgba(0,255,255,0.2);
        box-shadow: 0 0 20px rgba(0,255,255,0.05);
    }

    /* KPI card */
    .kpi-card {
        background: rgba(15, 15, 25, 0.8);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 1.2rem 1.5rem;
        text-align: center;
    }
    .kpi-label {
        font-family: 'Inter', sans-serif;
        font-size: 0.75rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #6b7280;
        margin-bottom: 0.5rem;
    }
    .kpi-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.8rem;
        font-weight: 600;
        margin: 0;
    }

    /* Neon glow text */
    .neon-cyan { color: #00ffff; text-shadow: 0 0 10px rgba(0,255,255,0.5); }
    .neon-violet { color: #8b5cf6; text-shadow: 0 0 10px rgba(139,92,246,0.5); }
    .neon-green { color: #00ff88; text-shadow: 0 0 10px rgba(0,255,136,0.5); }
    .neon-red { color: #ff3366; text-shadow: 0 0 10px rgba(255,51,102,0.5); }

    /* Status dot */
    .status-dot {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .status-dot.online {
        background: #00ff88;
        box-shadow: 0 0 8px rgba(0,255,136,0.6);
        animation: pulse 2s infinite;
    }
    .status-dot.offline {
        background: #ff3366;
        box-shadow: 0 0 8px rgba(255,51,102,0.6);
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #8b5cf6, #00ffff) !important;
        color: #0a0a0f !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        font-family: 'Inter', sans-serif !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        box-shadow: 0 0 20px rgba(139,92,246,0.4) !important;
        transform: translateY(-1px) !important;
    }

    /* Metrics override */
    [data-testid="stMetric"] {
        background: rgba(15, 15, 25, 0.8);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 1rem 1.2rem;
    }
    [data-testid="stMetricLabel"] {
        color: #6b7280 !important;
        font-size: 0.75rem !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace !important;
        color: #00ffff !important;
    }

    /* Slider */
    .stSlider [data-testid="stThumbValue"] {
        font-family: 'JetBrains Mono', monospace;
    }

    /* Alert overrides */
    .stAlert [data-testid="stAlertContentError"] {
        background: rgba(255,51,102,0.1);
        border-left-color: #ff3366;
    }
    .stAlert [data-testid="stAlertContentSuccess"] {
        background: rgba(0,255,136,0.1);
        border-left-color: #00ff88;
    }

    /* Anomaly alert card */
    .anomaly-alert {
        background: rgba(255,51,102,0.1);
        border: 1px solid rgba(255,51,102,0.3);
        border-radius: 12px;
        padding: 0.8rem 1.2rem;
        margin: 0.3rem 0;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        color: #ff3366;
    }

    /* Header */
    .sentinel-header {
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 0.5rem 0 1.5rem 0;
    }
    .sentinel-title {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00ffff, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
    }
    .sentinel-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 0.85rem;
        color: #6b7280;
        margin: 0;
    }

    /* Service cards */
    .service-card {
        background: rgba(15, 15, 25, 0.8);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    .service-card.online {
        border-color: rgba(0,255,136,0.3);
        box-shadow: 0 0 15px rgba(0,255,136,0.08);
    }
    .service-card.offline {
        border-color: rgba(255,51,102,0.3);
        box-shadow: 0 0 15px rgba(255,51,102,0.08);
    }
    .service-card .svc-name {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1rem;
        font-weight: 600;
        color: #e0e0e0;
        margin: 0 0 0.5rem 0;
    }
    .service-card .svc-status {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        margin: 0;
    }
    .service-card .svc-latency {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        color: #6b7280;
        margin: 0.3rem 0 0 0;
    }

    /* Pipeline flow */
    .pipeline-container {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0;
        flex-wrap: wrap;
        padding: 1.5rem 0;
    }
    .pipeline-node {
        background: rgba(15, 15, 25, 0.8);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 0.8rem 1.2rem;
        text-align: center;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        color: #e0e0e0;
        min-width: 90px;
    }
    .pipeline-node.online {
        border-color: rgba(0,255,136,0.4);
        box-shadow: 0 0 12px rgba(0,255,136,0.1);
    }
    .pipeline-node.offline {
        border-color: rgba(255,51,102,0.4);
        box-shadow: 0 0 12px rgba(255,51,102,0.1);
    }
    .pipeline-arrow {
        font-size: 1.2rem;
        color: #6b7280;
        padding: 0 0.4rem;
    }

    /* Section title */
    .section-title {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.3rem;
        font-weight: 600;
        background: linear-gradient(135deg, #00ffff, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 1rem 0 0.8rem 0;
    }

    /* Data table */
    .data-table-container {
        background: rgba(15, 15, 25, 0.8);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 1rem;
        overflow-x: auto;
    }
    .data-table-container table {
        width: 100%;
        border-collapse: collapse;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
    }
    .data-table-container th {
        color: #00ffff;
        text-transform: uppercase;
        font-size: 0.7rem;
        letter-spacing: 0.05em;
        padding: 0.6rem 0.8rem;
        border-bottom: 1px solid rgba(255,255,255,0.1);
        text-align: left;
    }
    .data-table-container td {
        color: #e0e0e0;
        padding: 0.5rem 0.8rem;
        border-bottom: 1px solid rgba(255,255,255,0.04);
    }
    .data-table-container tr.anomaly-row {
        background: rgba(255,51,102,0.08);
    }
    </style>
    """, unsafe_allow_html=True)
