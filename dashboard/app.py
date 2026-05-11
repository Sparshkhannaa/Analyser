import os

import streamlit as st

from data import get_last_run
from pages.today import render_today
from pages.positions import render_positions
from pages.history import render_history
from pages.equity import render_equity

_CSS = """
/* ── Reset & base ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #080812 !important;
    color: #cccccc;
    font-family: 'Courier New', Courier, monospace;
}
[data-testid="stSidebar"] {
    background-color: #0d0d1a !important;
    border-right: 1px solid #2a1f5e;
}
[data-testid="stSidebar"] * { font-family: 'Courier New', Courier, monospace; }

/* ── Sidebar title ── */
.sidebar-title {
    font-size: 18px;
    font-weight: 700;
    color: #7c3aed;
    letter-spacing: 4px;
    margin: 0 0 2px 0;
}
.sidebar-badge {
    font-size: 10px;
    color: #444;
    letter-spacing: 1px;
    margin: 0 0 20px 0;
}
.last-run {
    font-size: 10px;
    color: #333;
    margin-top: auto;
    padding-top: 20px;
}

/* ── Navigation radio ── */
[data-testid="stSidebar"] .stRadio > div { gap: 2px; }
[data-testid="stSidebar"] .stRadio label {
    display: block;
    padding: 6px 10px;
    border-radius: 4px;
    color: #555;
    font-size: 13px;
    cursor: pointer;
    transition: color 0.1s;
}
[data-testid="stSidebar"] .stRadio label:hover { color: #a78bfa; }
[data-testid="stSidebar"] .stRadio [aria-checked="true"] + div label,
[data-testid="stSidebar"] .stRadio input:checked ~ div label {
    color: #a78bfa;
    border-left: 2px solid #7c3aed;
    background: #7c3aed18;
    padding-left: 8px;
}
/* Hide radio circles */
[data-testid="stSidebar"] .stRadio [data-baseweb="radio"] > div:first-child { display: none; }

/* ── Stat cards ── */
.stat-card {
    background: #0d0d1a;
    border: 1px solid #2a1f5e;
    border-radius: 6px;
    padding: 14px 16px;
}
.stat-label {
    font-size: 9px;
    color: #555;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.stat-value {
    font-size: 26px;
    font-weight: 700;
}

/* ── Tables ── */
[data-testid="stDataFrame"] { border: 1px solid #1a1a3a; border-radius: 6px; }

/* ── Page headers ── */
.page-date { font-size: 12px; color: #555; margin: -8px 0 16px 0; }

/* ── Dividers ── */
hr { border-color: #1a1a3a; }
"""

_PAGES = ["Today", "Positions", "History", "Equity"]


def _sidebar() -> str:
    st.markdown(
        '<p class="sidebar-title">NIYO</p>'
        '<p class="sidebar-badge">paper trading</p>',
        unsafe_allow_html=True,
    )
    page = st.radio("nav", _PAGES, label_visibility="collapsed", key="page")
    last_run = get_last_run()
    if last_run:
        st.markdown(f'<p class="last-run">last run: {last_run}</p>', unsafe_allow_html=True)
    return page


def main():
    st.set_page_config(page_title="NIYO", layout="wide", initial_sidebar_state="expanded")
    st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)

    if not os.path.exists("paper_trading.db"):
        st.error("paper_trading.db not found. Is this the right directory? Run from the repo root: streamlit run dashboard/app.py")
        st.stop()

    with st.sidebar:
        page = _sidebar()

    if page == "Today":
        render_today()
    elif page == "Positions":
        render_positions()
    elif page == "History":
        render_history()
    elif page == "Equity":
        render_equity()


if __name__ == "__main__":
    main()
