import sqlite3

import pytest


def test_init_db_creates_tables(tmp_path):
    from db import init_db

    db = str(tmp_path / "test.db")
    init_db(db)
    with sqlite3.connect(db) as conn:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "positions" in tables
    assert "daily_log" in tables


def test_open_position_returns_id(tmp_path):
    from db import init_db, open_position

    db = str(tmp_path / "test.db")
    init_db(db)
    pid = open_position("AAPL", "2026-05-11", 175.0, 0.68, db)
    assert isinstance(pid, int) and pid > 0


def test_get_open_positions_returns_only_open(tmp_path):
    from db import init_db, open_position, close_position, get_open_positions

    db = str(tmp_path / "test.db")
    init_db(db)
    p1 = open_position("AAPL", "2026-05-11", 175.0, 0.68, db)
    open_position("MSFT", "2026-05-11", 420.0, 0.61, db)
    close_position(p1, "2026-05-16", 180.0, "hold_expired", 2.86, db)

    open_pos = get_open_positions(db)
    assert len(open_pos) == 1
    assert open_pos[0]["ticker"] == "MSFT"


def test_close_position_records_pnl(tmp_path):
    from db import init_db, open_position, close_position, get_open_positions

    db = str(tmp_path / "test.db")
    init_db(db)
    pid = open_position("NVDA", "2026-05-11", 900.0, 0.72, db)
    close_position(pid, "2026-05-16", 927.0, "hold_expired", 3.0, db)

    assert len(get_open_positions(db)) == 0


def test_log_daily_upserts(tmp_path):
    from db import init_db, log_daily

    db = str(tmp_path / "test.db")
    init_db(db)
    log_daily("2026-05-11", 3, 2, 1, db)
    log_daily("2026-05-11", 5, 3, 2, db)  # upsert — should not raise

    with sqlite3.connect(db) as conn:
        row = conn.execute("SELECT signals_found FROM daily_log WHERE date='2026-05-11'").fetchone()
    assert row[0] == 5
