import warnings
from datetime import date

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from alerts import format_daily_message, send_telegram
from db import close_position, get_open_positions, init_db, log_daily, open_position
from features import fetch_prices
from scan import DEFAULT_TICKERS, scan

HOLD_DAYS = 5
STOP_LOSS_PCT = 0.02


def _business_days_held(open_date_str: str, today: date) -> int:
    return int(np.busday_count(open_date_str, today.isoformat()))


def run_daily(
    tickers: list[str] | None = None,
    threshold: float = 0.52,
    hold_days: int = HOLD_DAYS,
    stop_loss_pct: float = STOP_LOSS_PCT,
    db_path: str = "paper_trading.db",
) -> pd.DataFrame:
    init_db(db_path)
    today = date.today()
    today_str = today.isoformat()
    tickers = tickers or DEFAULT_TICKERS

    # 1. Check open positions for stop-loss or hold expiry
    open_positions = get_open_positions(db_path)
    closed_today: list[dict] = []
    still_open_ids: set[int] = set()

    for pos in open_positions:
        try:
            prices = fetch_prices(pos["ticker"], period_days=30)
            latest_close = float(prices["Close"].iloc[-1])
        except Exception:
            still_open_ids.add(pos["id"])
            continue

        pnl_pct = (latest_close - pos["entry_price"]) / pos["entry_price"] * 100
        stop_level = pos["entry_price"] * (1 - stop_loss_pct)
        days_held = _business_days_held(pos["open_date"], today)

        if latest_close <= stop_level:
            close_position(pos["id"], today_str, latest_close, "stop_loss", pnl_pct, db_path)
            closed_today.append({**pos, "exit_price": latest_close, "exit_reason": "stop_loss", "pnl_pct": pnl_pct})
        elif days_held >= hold_days:
            close_position(pos["id"], today_str, latest_close, "hold_expired", pnl_pct, db_path)
            closed_today.append({**pos, "exit_price": latest_close, "exit_reason": "hold_expired", "pnl_pct": pnl_pct})
        else:
            still_open_ids.add(pos["id"])

    still_open = [p for p in open_positions if p["id"] in still_open_ids]

    # 2. Run scanner
    scan_df = scan(tickers, threshold=threshold)

    # 3. Open new positions (skip tickers already held)
    already_held = {p["ticker"] for p in still_open}
    new_longs = scan_df[scan_df["signal"] == 1] if not scan_df.empty else pd.DataFrame()
    opened_today: list[dict] = []

    for _, row in new_longs.iterrows():
        if row["ticker"] in already_held:
            continue
        entry = row["close"]
        open_position(row["ticker"], today_str, entry, float(row["prob"]), db_path)
        opened_today.append({"ticker": row["ticker"], "entry_price": entry, "prob": float(row["prob"])})

    # 4. Log daily summary
    log_daily(
        today_str,
        int(scan_df["signal"].sum()) if not scan_df.empty else 0,
        len(opened_today),
        len(closed_today),
        db_path,
    )

    # 5. Send Telegram alert (fall back to stdout if not configured)
    fresh_open = get_open_positions(db_path)
    msg = format_daily_message(scan_df, opened_today, closed_today, fresh_open, today_str)
    if not send_telegram(msg):
        print(msg)

    return scan_df


if __name__ == "__main__":
    run_daily()
