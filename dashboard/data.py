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
