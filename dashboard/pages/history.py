import pandas as pd
import streamlit as st

from data import get_closed_trades


def render_history() -> None:
    df = get_closed_trades()

    if df.empty:
        st.markdown(
            '<p style="color:#555;margin-top:40px;text-align:center">No closed trades yet.</p>',
            unsafe_allow_html=True,
        )
        return

    total = len(df)
    wins = int((df["pnl_pct"] > 0).sum())
    win_rate = wins / total * 100 if total > 0 else 0.0
    avg_return = df["pnl_pct"].mean()

    col1, col2, col3 = st.columns(3)
    with col1:
        _stat_card("TRADES", str(total), "#a78bfa")
    with col2:
        _stat_card("WIN RATE", f"{win_rate:.0f}%", "#10b981")
    with col3:
        color = "#10b981" if avg_return >= 0 else "#ef4444"
        _stat_card("AVG RETURN", f"{avg_return:+.2f}%", color)

    st.markdown("<br>", unsafe_allow_html=True)

    display = df[["ticker", "open_date", "exit_date", "entry_price",
                  "exit_price", "pnl_pct", "exit_reason"]].copy()
    display = display.rename(columns={
        "ticker": "Ticker", "open_date": "Opened", "exit_date": "Closed",
        "entry_price": "Entry", "exit_price": "Exit",
        "pnl_pct": "P&L %", "exit_reason": "Reason",
    })
    display["Entry"] = display["Entry"].apply(lambda x: f"${x:,.2f}")
    display["Exit"] = display["Exit"].apply(lambda x: f"${x:,.2f}")
    display["P&L %"] = display["P&L %"].apply(lambda x: f"{x:+.2f}%")

    styled = (
        display.style
        .map(lambda v: "color: #a78bfa", subset=["Ticker"])
        .map(
            lambda v: "color: #10b981" if v.startswith("+") else "color: #ef4444",
            subset=["P&L %"],
        )
        .map(
            lambda v: "color: #ef444488" if v == "stop_loss" else "color: #555",
            subset=["Reason"],
        )
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)


def _stat_card(label: str, value: str, color: str) -> None:
    st.markdown(
        f'<div class="stat-card">'
        f'<div class="stat-label">{label}</div>'
        f'<div class="stat-value" style="color:{color}">{value}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
