import numpy as np
import pandas as pd
import pytest


def _make_backtest_inputs(synthetic_prices, synthetic_vix):
    from features import engineer_features
    from model import walk_forward_predict

    X, y = engineer_features(synthetic_prices, synthetic_vix)
    signals, model = walk_forward_predict(X, y, min_train_days=252, step_days=63)
    return synthetic_prices, signals, model, list(X.columns)


def test_run_backtest_returns_dataframe(synthetic_prices, synthetic_vix):
    from backtest import run_backtest

    prices, signals, _, _ = _make_backtest_inputs(synthetic_prices, synthetic_vix)
    bt = run_backtest(prices, signals)

    assert isinstance(bt, pd.DataFrame)
    assert "portfolio_value" in bt.columns
    assert "bh_value" in bt.columns
    assert "strategy_return" in bt.columns


def test_run_backtest_starts_at_capital(synthetic_prices, synthetic_vix):
    from backtest import run_backtest

    prices, signals, _, _ = _make_backtest_inputs(synthetic_prices, synthetic_vix)
    bt = run_backtest(prices, signals, initial_capital=50_000.0)

    assert bt["portfolio_value"].iloc[0] == pytest.approx(50_000.0, rel=0.05)


def test_costs_reduce_returns():
    from backtest import run_backtest

    n = 100
    dates = pd.bdate_range("2022-01-01", periods=n)
    opens = np.linspace(100, 120, n)
    prices = pd.DataFrame(
        {"Open": opens, "High": opens * 1.002, "Low": opens * 0.999, "Close": opens},
        index=dates,
    )
    signals = pd.DataFrame({"signal": 1, "prob": 0.7}, index=dates)

    bt_free = run_backtest(prices, signals, cost_pct=0.0)
    bt_costly = run_backtest(prices, signals, cost_pct=0.01)

    assert bt_free["cumulative"].iloc[-1] > bt_costly["cumulative"].iloc[-1]


def test_zero_signal_holds_no_position():
    from backtest import run_backtest

    n = 50
    dates = pd.bdate_range("2022-01-01", periods=n)
    opens = np.full(n, 100.0)
    closes = np.linspace(100, 110, n)
    prices = pd.DataFrame(
        {"Open": opens, "High": opens * 1.002, "Low": opens * 0.999, "Close": closes},
        index=dates,
    )
    signals = pd.DataFrame({"signal": 0, "prob": 0.3}, index=dates)

    bt = run_backtest(prices, signals, initial_capital=100_000.0)
    assert bt["portfolio_value"].iloc[-1] == pytest.approx(100_000.0, abs=100)


def test_compute_metrics_has_required_keys(synthetic_prices, synthetic_vix):
    from backtest import run_backtest, compute_metrics

    prices, signals, _, _ = _make_backtest_inputs(synthetic_prices, synthetic_vix)
    bt = run_backtest(prices, signals)
    metrics = compute_metrics(bt)

    for key in ["total_return", "sharpe", "sortino", "calmar", "max_drawdown",
                "win_rate", "trade_count", "bh_return", "beats_bh"]:
        assert key in metrics, f"Missing metric: {key}"


def test_max_drawdown_is_negative_or_zero(synthetic_prices, synthetic_vix):
    from backtest import run_backtest, compute_metrics

    prices, signals, _, _ = _make_backtest_inputs(synthetic_prices, synthetic_vix)
    bt = run_backtest(prices, signals)
    metrics = compute_metrics(bt)

    assert metrics["max_drawdown"] <= 0


def test_win_rate_between_zero_and_one(synthetic_prices, synthetic_vix):
    from backtest import run_backtest, compute_metrics

    prices, signals, _, _ = _make_backtest_inputs(synthetic_prices, synthetic_vix)
    bt = run_backtest(prices, signals)
    metrics = compute_metrics(bt)

    assert 0 <= metrics["win_rate"] <= 1


def test_earnings_filter_suppresses_signals(synthetic_prices, synthetic_vix):
    from backtest import run_backtest

    prices, signals, _, _ = _make_backtest_inputs(synthetic_prices, synthetic_vix)

    # Use all signal dates as earnings dates — should suppress everything
    earnings_dates = pd.DatetimeIndex(signals.index)
    bt_filtered = run_backtest(prices, signals, earnings_dates=earnings_dates, earnings_window=0)
    bt_unfiltered = run_backtest(prices, signals)

    filtered_trades = (bt_filtered["position_size"] > 0).sum()
    unfiltered_trades = (bt_unfiltered["position_size"] > 0).sum()
    assert filtered_trades < unfiltered_trades, "Earnings filter should suppress at least some signals"


# ── Stop-loss ──────────────────────────────────────────────────────────────────

def _make_crash_prices(n: int = 210) -> pd.DataFrame:
    """Prices with a 6% intraday drop (Low) — triggers a 2% stop."""
    dates = pd.bdate_range("2022-01-01", periods=n)
    opens = np.full(n, 100.0)
    closes = np.full(n, 93.0)
    return pd.DataFrame({
        "Open": opens,
        "High": opens * 1.001,
        "Low": closes * 0.99,   # Low ≈ 92 — below 2% stop at 98
        "Close": closes,
        "Volume": np.full(n, 1_000_000.0),
    }, index=dates)


def test_stop_loss_caps_loss():
    """Stop at 2% yields a smaller loss than holding through a 7% daily drop."""
    from backtest import run_backtest

    prices = _make_crash_prices()
    signals = pd.DataFrame(
        {"signal": [1], "prob": [0.70]},
        index=prices.index[200:201],
    )
    bt_stop = run_backtest(prices, signals, stop_loss_pct=0.02)
    bt_none = run_backtest(prices, signals, stop_loss_pct=None)

    assert bt_stop["cumulative"].iloc[-1] > bt_none["cumulative"].iloc[-1]


def test_stop_loss_triggers_early_exit():
    """When stop fires on day 1 of a 5-day hold, days 2-5 have zero position."""
    from backtest import run_backtest

    prices = _make_crash_prices()
    signals = pd.DataFrame(
        {"signal": [1], "prob": [0.70]},
        index=prices.index[200:201],
    )
    bt = run_backtest(prices, signals, stop_loss_pct=0.02, hold_days=5)

    active_days = int((bt["position_size"] > 0).sum())
    assert active_days == 1


def test_stop_loss_none_holds_full_period():
    """Disabling stop-loss (None) keeps the position active for all hold_days."""
    from backtest import run_backtest

    prices = _make_crash_prices()
    signals = pd.DataFrame(
        {"signal": [1], "prob": [0.70]},
        index=prices.index[200:201],
    )
    bt = run_backtest(prices, signals, stop_loss_pct=None, hold_days=5)

    active_days = int((bt["position_size"] > 0).sum())
    assert active_days == 5
