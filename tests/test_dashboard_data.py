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
