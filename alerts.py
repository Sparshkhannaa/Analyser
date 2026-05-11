import os

import requests
from dotenv import load_dotenv

load_dotenv()


def send_telegram(message: str) -> bool:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        return False
    resp = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"},
        timeout=10,
    )
    return resp.ok


def format_daily_message(
    scan_df,
    opened: list[dict],
    closed: list[dict],
    still_open: list[dict],
    today: str,
) -> str:
    lines = [f"*Scanner — {today}*\n"]

    import pandas as pd
    new_longs = scan_df[scan_df["signal"] == 1] if (scan_df is not None and not scan_df.empty) else pd.DataFrame()

    if not new_longs.empty:
        lines.append("*Signals Today*")
        for _, row in new_longs.iterrows():
            flag = "✅" if row["regime_ok"] else "❌"
            earn = "  ⚠️ earnings" if row["near_earnings"] else ""
            lines.append(f"`{row['ticker']:<6}` {row['prob']:.2f}  {flag}{earn}")
    else:
        lines.append("_No signals today_")

    if opened:
        lines.append("\n*Opened*")
        for p in opened:
            lines.append(f"`{p['ticker']}` @ ${p['entry_price']:.2f}  (prob {p['prob']:.2f})")

    if still_open:
        lines.append("\n*Holding*")
        for p in still_open:
            lines.append(f"`{p['ticker']}` entry ${p['entry_price']:.2f}  since {p['open_date']}")

    if closed:
        lines.append("\n*Closed*")
        for p in closed:
            sign = "+" if p["pnl_pct"] >= 0 else ""
            lines.append(
                f"`{p['ticker']}` {sign}{p['pnl_pct']:.1f}%  [{p['exit_reason']}]"
            )

    return "\n".join(lines)
