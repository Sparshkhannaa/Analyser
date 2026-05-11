import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "dashboard"))


def test_get_today_signals_reads_todays_csv(tmp_path):
    csv_content = "date,ticker,prob,signal,regime_ok,close,sma_200\n2026-05-11,NVDA,0.46,0,True,215.0,184.0\n"
    (tmp_path / "2026-05-11.csv").write_text(csv_content)

    from data import get_today_signals
    get_today_signals.clear()
    df, date_str = get_today_signals(signals_dir=str(tmp_path))

    assert len(df) == 1
    assert df.iloc[0]["ticker"] == "NVDA"
    assert date_str == "2026-05-11"


def test_get_today_signals_falls_back_to_most_recent(tmp_path):
    csv_content = "date,ticker,prob,signal,regime_ok,close,sma_200\n2026-04-01,AMD,0.38,0,True,455.0,217.0\n"
    (tmp_path / "2026-04-01.csv").write_text(csv_content)

    from data import get_today_signals
    get_today_signals.clear()
    df, date_str = get_today_signals(signals_dir=str(tmp_path))

    assert date_str == "2026-04-01"
    assert df.iloc[0]["ticker"] == "AMD"


def test_get_today_signals_returns_empty_when_no_csvs(tmp_path):
    from data import get_today_signals
    get_today_signals.clear()
    df, date_str = get_today_signals(signals_dir=str(tmp_path))

    assert df.empty
    assert date_str == ""


def _seed_db(db_path: str) -> None:
    """Helper: seed a test DB with positions and daily_log rows."""
    import sqlite3
    from db import init_db, open_position, close_position, log_daily
    init_db(db_path)
    p1 = open_position("NVDA", "2026-04-28", 198.0, 0.61, db_path)
    close_position(p1, "2026-05-05", 206.0, "hold_expired", 4.04, db_path)
    open_position("AMD", "2026-05-08", 448.0, 0.55, db_path)
    log_daily("2026-05-05", 3, 1, 1, db_path)
    log_daily("2026-05-08", 2, 1, 0, db_path)


def test_get_closed_trades_returns_only_closed(tmp_path):
    db = str(tmp_path / "test.db")
    _seed_db(db)

    from data import get_closed_trades
    get_closed_trades.clear()
    df = get_closed_trades(db_path=db)

    assert len(df) == 1
    assert df.iloc[0]["ticker"] == "NVDA"
    assert df.iloc[0]["pnl_pct"] == pytest.approx(4.04)


def test_get_last_run_returns_most_recent_date(tmp_path):
    db = str(tmp_path / "test.db")
    _seed_db(db)

    from data import get_last_run
    get_last_run.clear()
    last = get_last_run(db_path=db)

    assert last == "2026-05-08"


def test_get_last_run_returns_none_when_no_log(tmp_path):
    from db import init_db
    from data import get_last_run

    db = str(tmp_path / "empty.db")
    init_db(db)
    get_last_run.clear()

    assert get_last_run(db_path=db) is None


def test_get_open_positions_enriched(tmp_path, monkeypatch):
    db = str(tmp_path / "test.db")
    _seed_db(db)  # seeds 1 open position: AMD at $448

    import pandas as _pd

    class FakeTicker:
        def history(self, period):
            return _pd.DataFrame({"Close": [460.0, 455.0]})

    monkeypatch.setattr("yfinance.Ticker", lambda ticker: FakeTicker())

    from data import get_open_positions_enriched, _fetch_current_prices
    get_open_positions_enriched.clear()
    _fetch_current_prices.clear()
    df = get_open_positions_enriched(db_path=db)

    assert len(df) == 1
    assert df.iloc[0]["ticker"] == "AMD"
    assert df.iloc[0]["current_price"] == pytest.approx(455.0)
    assert df.iloc[0]["pnl_pct"] == pytest.approx((455.0 - 448.0) / 448.0 * 100)
    assert df.iloc[0]["stop_level"] == pytest.approx(448.0 * 0.98)
    assert df.iloc[0]["days_held"] >= 0


def test_get_open_positions_enriched_empty(tmp_path):
    from db import init_db
    from data import get_open_positions_enriched
    db = str(tmp_path / "empty.db")
    init_db(db)
    get_open_positions_enriched.clear()

    df = get_open_positions_enriched(db_path=db)
    assert df.empty
