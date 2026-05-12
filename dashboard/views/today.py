import pandas as pd
import streamlit as st

from data import get_today_signals, get_last_run


def render_today() -> None:
    df, date_str = get_today_signals()

    if df.empty:
        last_run = get_last_run()
        st.warning(f"No scan data yet for today. Last run: {last_run or 'never'}")
        return

    st.markdown(f'<p class="page-date">{date_str}</p>', unsafe_allow_html=True)

    scanned = len(df)
    signals = int((df["signal"] == 1).sum())

    spy_row = df[df["ticker"] == "SPY"] if "ticker" in df.columns else pd.DataFrame()
    if not spy_row.empty:
        regime_ok = bool(spy_row["regime_ok"].iloc[0])
        spy_close = float(spy_row["close"].iloc[0])
        regime_label = "BULL ▲" if regime_ok else "BEAR ▼"
        regime_color = "#10b981" if regime_ok else "#ef4444"
    else:
        regime_label, regime_color, spy_close = "—", "#888", None

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        _stat_card("SCANNED", str(scanned), "#a78bfa")
    with col2:
        _stat_card("SIGNALS", str(signals), "#10b981" if signals > 0 else "#555")
    with col3:
        _stat_card("SPY REGIME", regime_label, regime_color)
    with col4:
        price_str = f"${spy_close:,.2f}" if spy_close else "—"
        _stat_card("SPY CLOSE", price_str, "#888")

    st.markdown("<br>", unsafe_allow_html=True)
    _scanner_table(df)


def _stat_card(label: str, value: str, color: str) -> None:
    st.markdown(
        f'<div class="stat-card">'
        f'<div class="stat-label">{label}</div>'
        f'<div class="stat-value" style="color:{color}">{value}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _scanner_table(df: pd.DataFrame) -> None:
    display = df.copy()
    display["signal"] = display["signal"].apply(lambda x: "LONG" if x == 1 else "—")
    display["regime_ok"] = display["regime_ok"].apply(lambda x: "✓" if x else "✗")
    has_earnings = "near_earnings" in display.columns
    if has_earnings:
        display["near_earnings"] = display["near_earnings"].apply(
            lambda x: "NEAR" if x else "—"
        )
    display = display.rename(columns={
        "ticker": "Ticker", "prob": "Prob", "signal": "Signal",
        "regime_ok": "Regime", "near_earnings": "Earnings",
    })
    cols = ["Ticker", "Prob", "Signal", "Regime"] + (["Earnings"] if has_earnings else [])
    display["Prob"] = display["Prob"].round(3)

    style = (
        display[cols]
        .style
        .map(lambda v: "color: #a78bfa", subset=["Ticker"])
        .map(lambda v: "color: #10b981" if v == "LONG" else "color: #555", subset=["Signal"])
        .map(lambda v: "color: #10b981" if v == "✓" else "color: #ef4444", subset=["Regime"])
    )
    if has_earnings:
        style = style.map(lambda v: "color: #f59e0b" if v == "NEAR" else "color: #555", subset=["Earnings"])
    styled = style
    st.dataframe(styled, use_container_width=True, hide_index=True)
