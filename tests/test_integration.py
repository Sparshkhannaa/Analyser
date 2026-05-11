import numpy as np
import pandas as pd


def test_full_pipeline_runs_without_error(monkeypatch):
    import yfinance as yf

    def mock_download(ticker, **kwargs):
        np.random.seed(0)
        n = 700
        dates = pd.bdate_range("2021-01-01", periods=n)
        close = 400 + np.cumsum(np.random.randn(n) * 2)
        high = close + np.abs(np.random.randn(n))
        low = close - np.abs(np.random.randn(n))
        volume = np.random.randint(50_000_000, 150_000_000, n).astype(float)
        if ticker == "^VIX":
            vix = np.clip(15 + np.random.randn(n) * 3, 10, 60)
            return pd.DataFrame(
                {"Open": vix, "High": vix, "Low": vix, "Close": vix, "Volume": 0.0},
                index=dates,
            )
        return pd.DataFrame(
            {"Open": close, "High": high, "Low": low, "Close": close, "Volume": volume},
            index=dates,
        )

    monkeypatch.setattr(yf, "download", mock_download)

    from features import fetch_prices, fetch_vix, engineer_features
    from model import walk_forward_predict
    from backtest import run_backtest, compute_metrics

    prices = fetch_prices("SPY", 400)
    vix = fetch_vix(400)
    X, y = engineer_features(prices, vix)
    signals, model = walk_forward_predict(X, y, min_train_days=252, step_days=63)
    bt = run_backtest(prices, signals)
    metrics = compute_metrics(bt)

    assert "sharpe" in metrics
    assert "total_return" in metrics
    assert "beats_bh" in metrics
    assert len(signals) > 0
    assert model is not None
