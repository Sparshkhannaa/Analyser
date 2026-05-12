import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data import get_closed_trades, get_spy_prices


def render_equity() -> None:
    trades = get_closed_trades()

    if len(trades) < 2:
        first_date = trades["open_date"].min() if not trades.empty else None
        since = f" since {first_date}" if first_date else ""
        st.markdown(
            f'<p class="page-date">Cumulative Return{since}</p>',
            unsafe_allow_html=True,
        )
        st.info("Not enough trade history to plot an equity curve yet.")
        return

    first_date = trades["open_date"].min()
    st.markdown(
        f'<p class="page-date">Cumulative Return since {first_date}</p>',
        unsafe_allow_html=True,
    )

    niyo_series = _build_niyo_series(trades)
    start_date = str(trades["open_date"].min())
    end_date = str(trades["exit_date"].max())
    spy_df = get_spy_prices(start_date, end_date)

    fig = _build_chart(niyo_series, spy_df)
    st.plotly_chart(fig, use_container_width=True)

    niyo_total = niyo_series["cumulative_return"].iloc[-1]
    spy_total = (
        (spy_df["Close"].iloc[-1] / spy_df["Close"].iloc[0] - 1) * 100
        if not spy_df.empty and len(spy_df) > 1
        else None
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f'<div class="stat-card"><div class="stat-label">NIYO TOTAL</div>'
            f'<div class="stat-value" style="color:#7c3aed">{niyo_total:+.2f}%</div></div>',
            unsafe_allow_html=True,
        )
    with col2:
        spy_str = f"{spy_total:+.2f}%" if spy_total is not None else "—"
        st.markdown(
            f'<div class="stat-card"><div class="stat-label">SPY TOTAL</div>'
            f'<div class="stat-value" style="color:#555">{spy_str}</div></div>',
            unsafe_allow_html=True,
        )


def _build_niyo_series(trades: pd.DataFrame) -> pd.DataFrame:
    df = trades[["exit_date", "pnl_pct"]].copy()
    df = df.sort_values("exit_date").reset_index(drop=True)
    df["pnl_factor"] = 1 + df["pnl_pct"] / 100
    df["cumulative_return"] = (df["pnl_factor"].cumprod() - 1) * 100
    df["exit_date"] = pd.to_datetime(df["exit_date"])
    return df[["exit_date", "cumulative_return"]]


def _build_chart(niyo: pd.DataFrame, spy: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    if not spy.empty and len(spy) > 1:
        spy_return = (spy["Close"] / spy["Close"].iloc[0] - 1) * 100
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(spy["Date"]),
            y=spy_return,
            name="SPY",
            line=dict(color="#555555", width=1.5),
            hovertemplate="%{x|%Y-%m-%d}<br>SPY: %{y:.2f}%<extra></extra>",
        ))

    fig.add_trace(go.Scatter(
        x=niyo["exit_date"],
        y=niyo["cumulative_return"],
        name="NIYO",
        line=dict(color="#7c3aed", width=2),
        fill="tozeroy",
        fillcolor="rgba(124, 58, 237, 0.08)",
        hovertemplate="%{x|%Y-%m-%d}<br>NIYO: %{y:.2f}%<extra></extra>",
    ))

    fig.update_layout(
        paper_bgcolor="#080812",
        plot_bgcolor="#080812",
        font=dict(family="Courier New, monospace", color="#888"),
        legend=dict(bgcolor="#0d0d1a", bordercolor="#2a1f5e", borderwidth=1),
        xaxis=dict(gridcolor="#1a1a3a", showgrid=True),
        yaxis=dict(gridcolor="#1a1a3a", showgrid=True, ticksuffix="%"),
        margin=dict(l=0, r=0, t=20, b=0),
        hovermode="x unified",
    )
    return fig
