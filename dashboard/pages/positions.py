import pandas as pd
import streamlit as st

from data import get_open_positions_enriched


def render_positions() -> None:
    df = get_open_positions_enriched()

    if df.empty:
        st.markdown(
            '<p style="color:#555;margin-top:40px;text-align:center">No open positions.</p>',
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        f'<p class="page-date">Open Positions · {len(df)} held</p>',
        unsafe_allow_html=True,
    )

    display = df[["ticker", "open_date", "entry_price", "current_price",
                  "pnl_pct", "days_held", "stop_level"]].copy()
    display = display.rename(columns={
        "ticker": "Ticker", "open_date": "Opened", "entry_price": "Entry",
        "current_price": "Current", "pnl_pct": "P&L %",
        "days_held": "Days", "stop_level": "Stop",
    })
    display["Entry"] = display["Entry"].apply(lambda x: f"${x:,.2f}")
    display["Current"] = display["Current"].apply(
        lambda x: f"${x:,.2f}" if pd.notna(x) else "—"
    )
    display["P&L %"] = display["P&L %"].apply(
        lambda x: f"{x:+.2f}%" if pd.notna(x) else "—"
    )
    display["Stop"] = display["Stop"].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "—")

    styled = (
        display.style
        .map(lambda v: "color: #a78bfa", subset=["Ticker"])
        .map(
            lambda v: "color: #10b981" if v.startswith("+") else "color: #ef4444"
            if v.startswith("-") else "color: #888",
            subset=["P&L %"],
        )
        .map(lambda v: "color: #ef444466", subset=["Stop"])
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

    missing_prices = df["current_price"].isna().sum()
    if missing_prices > 0:
        st.warning(f"Could not fetch live prices for {missing_prices} ticker(s). P&L shown as — for those rows.")

    valid_pnl = df["pnl_pct"].dropna()
    if not valid_pnl.empty:
        avg = valid_pnl.mean()
        color = "#10b981" if avg >= 0 else "#ef4444"
        st.markdown(
            f'<div style="text-align:right;margin-top:8px;font-size:13px;color:#555">'
            f'Total Unrealised (avg): '
            f'<span style="color:{color};font-weight:700">{avg:+.2f}%</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
