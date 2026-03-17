import streamlit as st
import pandas as pd


def render_data_table(df: pd.DataFrame):
    if df.empty:
        st.info("No data to display.")
        return

    rows_html = []
    for _, row in df.iterrows():
        is_anom = row.get("is_anomaly", False)
        tr_class = ' class="anomaly-row"' if is_anom else ""
        cells = "".join(
            f"<td>{_fmt(row[c])}</td>" for c in df.columns
        )
        rows_html.append(f"<tr{tr_class}>{cells}</tr>")

    headers = "".join(f"<th>{c}</th>" for c in df.columns)

    st.markdown(
        f'<div class="data-table-container">'
        f"<table><thead><tr>{headers}</tr></thead>"
        f"<tbody>{''.join(rows_html)}</tbody></table></div>",
        unsafe_allow_html=True,
    )


def _fmt(val):
    if isinstance(val, float):
        return f"{val:.4f}"
    if isinstance(val, bool):
        return "YES" if val else "no"
    return str(val)
