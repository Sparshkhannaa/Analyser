import sqlite3
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

DB_PATH = "paper_trading.db"
SIGNALS_DIR = "signals"


@st.cache_data(ttl=300)
def get_today_signals(signals_dir: str = SIGNALS_DIR) -> tuple[pd.DataFrame, str]:
    """Return (df, date_str) for today's signals CSV, falling back to most recent."""
    today = date.today().isoformat()
    path = Path(signals_dir) / f"{today}.csv"
    if not path.exists():
        csvs = sorted(Path(signals_dir).glob("*.csv"))
        if not csvs:
            return pd.DataFrame(), ""
        path = csvs[-1]
    df = pd.read_csv(path)
    return df, path.stem


@st.cache_data(ttl=300)
def get_closed_trades(db_path: str = DB_PATH) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(
            "SELECT * FROM positions WHERE exit_date IS NOT NULL ORDER BY exit_date DESC",
            conn,
        )


@st.cache_data(ttl=300)
def get_last_run(db_path: str = DB_PATH) -> str | None:
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT date FROM daily_log ORDER BY date DESC LIMIT 1"
        ).fetchone()
    return row[0] if row else None


@st.cache_data(ttl=60)
def get_open_positions_enriched(db_path: str = DB_PATH) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM positions WHERE exit_date IS NULL ORDER BY open_date"
        ).fetchall()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([dict(r) for r in rows])
    prices = _fetch_current_prices(tuple(df["ticker"].tolist()))
    df["current_price"] = df["ticker"].map(prices)
    df["pnl_pct"] = (df["current_price"] - df["entry_price"]) / df["entry_price"] * 100
    df["stop_level"] = df["entry_price"] * 0.98
    today = date.today().isoformat()
    df["days_held"] = df["open_date"].apply(lambda d: int(np.busday_count(d, today)))
    return df


@st.cache_data(ttl=60)
def _fetch_current_prices(tickers: tuple) -> dict:
    result = {}
    for ticker in tickers:
        try:
            hist = yf.Ticker(ticker).history(period="2d")
            result[ticker] = float(hist["Close"].iloc[-1])
        except Exception:
            result[ticker] = None
    return result


@st.cache_data(ttl=300)
def get_spy_prices(start_date: str, end_date: str) -> pd.DataFrame:
    try:
        df = yf.Ticker("SPY").history(start=start_date, end=end_date)
    except Exception:
        return pd.DataFrame(columns=["Date", "Close"])
    if df.empty:
        return pd.DataFrame(columns=["Date", "Close"])
    out = df[["Close"]].reset_index()
    out["Date"] = pd.to_datetime(out["Date"]).dt.date
    return out[["Date", "Close"]]
