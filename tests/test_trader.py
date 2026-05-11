from datetime import date, timedelta
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest


def _scan_df(signal: int = 1, ticker: str = "AAPL") -> pd.DataFrame:
    return pd.DataFrame([{
        "ticker": ticker, "prob": 0.68, "signal": signal,
        "regime_ok": True, "near_earnings": False, "close": 175.0, "sma_200": 160.0,
    }])


def _prices(close: float = 175.0) -> pd.DataFrame:
    dates = pd.date_range("2026-04-01", periods=30, freq="B")
    return pd.DataFrame({"Close": np.full(30, close)}, index=dates)


def test_run_daily_opens_new_position(tmp_path):
    from db import get_open_positions
    from trader import run_daily

    db = str(tmp_path / "test.db")
    with patch("trader.scan", return_value=_scan_df()), \
         patch("trader.fetch_prices", return_value=_prices()), \
         patch("trader.send_telegram", return_value=True):
        run_daily(tickers=["AAPL"], db_path=db)

    positions = get_open_positions(db)
    assert len(positions) == 1
    assert positions[0]["ticker"] == "AAPL"
    assert positions[0]["entry_price"] == pytest.approx(175.0)


def test_run_daily_closes_expired_position(tmp_path):
    from db import get_open_positions, init_db, open_position
    from trader import run_daily

    db = str(tmp_path / "test.db")
    init_db(db)
    old_date = (date.today() - timedelta(days=10)).isoformat()
    open_position("AAPL", old_date, 170.0, 0.68, db)

    with patch("trader.scan", return_value=_scan_df(signal=0)), \
         patch("trader.fetch_prices", return_value=_prices(close=180.0)), \
         patch("trader.send_telegram", return_value=True):
        run_daily(tickers=["AAPL"], db_path=db)

    assert len(get_open_positions(db)) == 0


def test_run_daily_stops_out_position(tmp_path):
    from db import get_open_positions, init_db, open_position
    from trader import run_daily

    db = str(tmp_path / "test.db")
    init_db(db)
    open_position("AAPL", date.today().isoformat(), 200.0, 0.68, db)

    # 192 is below 2% stop (196)
    with patch("trader.scan", return_value=_scan_df(signal=0)), \
         patch("trader.fetch_prices", return_value=_prices(close=192.0)), \
         patch("trader.send_telegram", return_value=True):
        run_daily(tickers=["AAPL"], db_path=db, stop_loss_pct=0.02)

    assert len(get_open_positions(db)) == 0


def test_run_daily_skips_duplicate_ticker(tmp_path):
    from db import get_open_positions, init_db, open_position
    from trader import run_daily

    db = str(tmp_path / "test.db")
    init_db(db)
    open_position("AAPL", date.today().isoformat(), 175.0, 0.65, db)

    with patch("trader.scan", return_value=_scan_df(signal=1, ticker="AAPL")), \
         patch("trader.fetch_prices", return_value=_prices(close=176.0)), \
         patch("trader.send_telegram", return_value=True):
        run_daily(tickers=["AAPL"], db_path=db)

    assert len(get_open_positions(db)) == 1  # not doubled


def test_run_daily_returns_scan_dataframe(tmp_path):
    from trader import run_daily

    db = str(tmp_path / "test.db")
    with patch("trader.scan", return_value=_scan_df()), \
         patch("trader.fetch_prices", return_value=_prices()), \
         patch("trader.send_telegram", return_value=True):
        result = run_daily(tickers=["AAPL"], db_path=db)

    assert isinstance(result, pd.DataFrame)
    assert "signal" in result.columns
