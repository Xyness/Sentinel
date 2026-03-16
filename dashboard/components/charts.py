import plotly.graph_objects as go
import numpy as np
import pandas as pd


PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="JetBrains Mono, monospace", color="#e0e0e0", size=11),
    margin=dict(l=40, r=20, t=40, b=40),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.05)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.05)"),
)


def anomaly_score_timeline(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    normal = df[~df["is_anomaly"]]
    anomalies = df[df["is_anomaly"]]

    fig.add_trace(go.Scatter(
        x=list(range(len(df))),
        y=df["anomaly_score"],
        mode="lines",
        line=dict(color="#00ffff", width=1.5),
        name="Score",
        hovertemplate="Score: %{y:.4f}<extra></extra>",
    ))

    if len(anomalies) > 0:
        fig.add_trace(go.Scatter(
            x=anomalies.index.tolist(),
            y=anomalies["anomaly_score"],
            mode="markers",
            marker=dict(color="#ff3366", size=8, symbol="circle",
                        line=dict(width=1, color="#ff3366")),
            name="Anomaly",
            hovertemplate="ANOMALY<br>Score: %{y:.4f}<extra></extra>",
        ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Anomaly Score Timeline", font=dict(size=14)),
        showlegend=True,
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        height=350,
    )
    return fig


def score_distribution(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=df["anomaly_score"],
        nbinsx=30,
        marker=dict(color="#8b5cf6", line=dict(width=1, color="#a78bfa")),
        opacity=0.8,
        hovertemplate="Score: %{x:.3f}<br>Count: %{y}<extra></extra>",
    ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Score Distribution", font=dict(size=14)),
        xaxis_title="Anomaly Score",
        yaxis_title="Count",
        height=300,
    )
    return fig


def anomaly_gauge(rate: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=rate,
        number=dict(suffix="%", font=dict(size=36, color="#e0e0e0")),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor="#6b7280",
                      tickfont=dict(color="#6b7280")),
            bar=dict(color="#ff3366" if rate > 10 else "#00ffff"),
            bgcolor="rgba(15,15,25,0.5)",
            borderwidth=0,
            steps=[
                dict(range=[0, 5], color="rgba(0,255,136,0.15)"),
                dict(range=[5, 15], color="rgba(139,92,246,0.15)"),
                dict(range=[15, 100], color="rgba(255,51,102,0.15)"),
            ],
        ),
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="JetBrains Mono, monospace", color="#e0e0e0"),
        margin=dict(l=30, r=30, t=50, b=20),
        title=dict(text="Anomaly Rate", font=dict(size=14, color="#e0e0e0")),
        height=280,
    )
    return fig


def symbol_heatmap(df: pd.DataFrame) -> go.Figure:
    if df.empty or "symbol" not in df.columns:
        fig = go.Figure()
        fig.update_layout(**PLOTLY_LAYOUT, height=300,
                          title=dict(text="Score Heatmap by Symbol", font=dict(size=14)))
        return fig

    # Create bins of 10 predictions each
    df = df.copy()
    df["bin"] = df.groupby("symbol").cumcount() // 10

    pivot = df.pivot_table(values="anomaly_score", index="symbol",
                           columns="bin", aggfunc="mean")

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[f"T{i}" for i in pivot.columns],
        y=pivot.index.tolist(),
        colorscale=[
            [0, "#0a0a0f"],
            [0.3, "#1a0a2e"],
            [0.6, "#8b5cf6"],
            [1.0, "#ff3366"],
        ],
        hovertemplate="Symbol: %{y}<br>Period: %{x}<br>Avg Score: %{z:.4f}<extra></extra>",
    ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Score Heatmap by Symbol", font=dict(size=14)),
        height=300,
    )
    return fig


# ── New charts ───────────────────────────────────────────────────

FEATURE_COLS = [
    "z_score_price", "z_score_log_return", "z_score_volume",
    "rolling_price_std", "rolling_volume_std",
]


def feature_correlation_matrix(df: pd.DataFrame) -> go.Figure:
    cols = [c for c in FEATURE_COLS if c in df.columns]
    if not cols:
        fig = go.Figure()
        fig.update_layout(**PLOTLY_LAYOUT, height=350,
                          title=dict(text="Feature Correlation", font=dict(size=14)))
        return fig

    corr = df[cols].corr()
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=cols, y=cols,
        colorscale=[[0, "#0a0a0f"], [0.5, "#8b5cf6"], [1, "#00ffff"]],
        zmin=-1, zmax=1,
        hovertemplate="%{x} vs %{y}: %{z:.2f}<extra></extra>",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Feature Correlation Matrix", font=dict(size=14)),
        height=400,
    )
    return fig


def per_symbol_bar_chart(per_symbol_stats: dict) -> go.Figure:
    if not per_symbol_stats:
        fig = go.Figure()
        fig.update_layout(**PLOTLY_LAYOUT, height=350)
        return fig

    symbols = list(per_symbol_stats.keys())
    counts = [per_symbol_stats[s]["count"] for s in symbols]
    anomalies = [per_symbol_stats[s]["anomalies"] for s in symbols]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=symbols, y=counts, name="Total",
        marker_color="#00ffff", opacity=0.8,
    ))
    fig.add_trace(go.Bar(
        x=symbols, y=anomalies, name="Anomalies",
        marker_color="#ff3366", opacity=0.8,
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT, barmode="group",
        title=dict(text="Predictions per Symbol", font=dict(size=14)),
        height=350,
    )
    return fig


def score_trend_line(df: pd.DataFrame, window: int = 20) -> go.Figure:
    fig = go.Figure()
    if "anomaly_score" not in df.columns or df.empty:
        fig.update_layout(**PLOTLY_LAYOUT, height=300)
        return fig

    rolling = df["anomaly_score"].rolling(window=window, min_periods=1).mean()
    fig.add_trace(go.Scatter(
        x=list(range(len(df))), y=df["anomaly_score"],
        mode="lines", line=dict(color="rgba(0,255,255,0.2)", width=1),
        name="Raw",
    ))
    fig.add_trace(go.Scatter(
        x=list(range(len(rolling))), y=rolling,
        mode="lines", line=dict(color="#8b5cf6", width=2.5),
        name=f"MA({window})",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Score Trend (Moving Average)", font=dict(size=14)),
        height=300,
    )
    return fig


def feature_box_plots(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    cols = [c for c in FEATURE_COLS if c in df.columns]
    if not cols or "is_anomaly" not in df.columns:
        fig.update_layout(**PLOTLY_LAYOUT, height=350)
        return fig

    normal = df[~df["is_anomaly"]]
    anom = df[df["is_anomaly"]]

    for i, col in enumerate(cols):
        fig.add_trace(go.Box(
            y=normal[col], name=col, legendgroup="Normal",
            showlegend=(i == 0), marker_color="#00ffff",
            boxmean=True, offsetgroup="N",
        ))
        fig.add_trace(go.Box(
            y=anom[col] if len(anom) > 0 else [], name=col, legendgroup="Anomaly",
            showlegend=(i == 0), marker_color="#ff3366",
            boxmean=True, offsetgroup="A",
        ))

    fig.update_layout(
        **PLOTLY_LAYOUT, boxmode="group",
        title=dict(text="Features: Normal vs Anomaly", font=dict(size=14)),
        height=400,
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig


def anomaly_type_breakdown(df: pd.DataFrame) -> go.Figure:
    if "is_anomaly" not in df.columns or df.empty:
        fig = go.Figure()
        fig.update_layout(**PLOTLY_LAYOUT, height=300)
        return fig

    anom_count = int(df["is_anomaly"].sum())
    normal_count = len(df) - anom_count

    fig = go.Figure(go.Pie(
        labels=["Normal", "Anomaly"],
        values=[normal_count, anom_count],
        hole=0.55,
        marker=dict(colors=["#00ffff", "#ff3366"]),
        textfont=dict(color="#e0e0e0"),
        hovertemplate="%{label}: %{value} (%{percent})<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="JetBrains Mono, monospace", color="#e0e0e0", size=11),
        margin=dict(l=20, r=20, t=40, b=20),
        title=dict(text="Normal vs Anomaly", font=dict(size=14)),
        height=300,
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig
