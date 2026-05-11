import sqlite3

DB_PATH = "paper_trading.db"


def init_db(db_path: str = DB_PATH) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker      TEXT NOT NULL,
                open_date   TEXT NOT NULL,
                entry_price REAL NOT NULL,
                prob        REAL NOT NULL,
                exit_date   TEXT,
                exit_price  REAL,
                exit_reason TEXT,
                pnl_pct     REAL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS daily_log (
                date              TEXT PRIMARY KEY,
                signals_found     INTEGER,
                positions_opened  INTEGER,
                positions_closed  INTEGER
            )
        """)


def open_position(
    ticker: str,
    open_date: str,
    entry_price: float,
    prob: float,
    db_path: str = DB_PATH,
) -> int:
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO positions (ticker, open_date, entry_price, prob) VALUES (?, ?, ?, ?)",
            (ticker, open_date, entry_price, prob),
        )
        return cur.lastrowid


def close_position(
    position_id: int,
    exit_date: str,
    exit_price: float,
    exit_reason: str,
    pnl_pct: float,
    db_path: str = DB_PATH,
) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """UPDATE positions
               SET exit_date=?, exit_price=?, exit_reason=?, pnl_pct=?
               WHERE id=?""",
            (exit_date, exit_price, exit_reason, pnl_pct, position_id),
        )


def get_open_positions(db_path: str = DB_PATH) -> list[dict]:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM positions WHERE exit_date IS NULL ORDER BY open_date"
        ).fetchall()
        return [dict(row) for row in rows]


def log_daily(
    date: str,
    signals_found: int,
    positions_opened: int,
    positions_closed: int,
    db_path: str = DB_PATH,
) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """INSERT OR REPLACE INTO daily_log
               (date, signals_found, positions_opened, positions_closed)
               VALUES (?, ?, ?, ?)""",
            (date, signals_found, positions_opened, positions_closed),
        )
