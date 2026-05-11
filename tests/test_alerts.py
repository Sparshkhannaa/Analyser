from unittest.mock import MagicMock, patch

import pandas as pd


def test_send_telegram_returns_false_when_unconfigured(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    import alerts
    assert alerts.send_telegram("test") is False


def test_send_telegram_calls_api(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake_token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "12345")
    mock_resp = MagicMock()
    mock_resp.ok = True
    with patch("alerts.requests.post", return_value=mock_resp) as mock_post:
        import alerts
        result = alerts.send_telegram("hello")
    assert result is True
    mock_post.assert_called_once()
    _, kwargs = mock_post.call_args
    assert kwargs["json"]["chat_id"] == "12345"


def test_format_daily_message_includes_ticker():
    from alerts import format_daily_message

    scan_df = pd.DataFrame([{
        "ticker": "AAPL", "prob": 0.68, "signal": 1,
        "regime_ok": True, "near_earnings": False, "close": 175.0, "sma_200": 160.0,
    }])
    msg = format_daily_message(scan_df, [], [], [], "2026-05-11")
    assert "AAPL" in msg
    assert "0.68" in msg


def test_format_daily_message_no_signals():
    from alerts import format_daily_message

    scan_df = pd.DataFrame([{
        "ticker": "AAPL", "prob": 0.45, "signal": 0,
        "regime_ok": True, "near_earnings": False, "close": 175.0, "sma_200": 160.0,
    }])
    msg = format_daily_message(scan_df, [], [], [], "2026-05-11")
    assert "No signals" in msg


def test_format_daily_message_shows_closed_pnl():
    from alerts import format_daily_message

    closed = [{"ticker": "MSFT", "pnl_pct": 3.5, "exit_reason": "hold_expired"}]
    msg = format_daily_message(None, [], closed, [], "2026-05-11")
    assert "MSFT" in msg
    assert "+3.5%" in msg
