import json
from datetime import date
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from xgboost import XGBClassifier

from features import _FEATURE_COLS


def _make_prices(n: int = 600, uptrend: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
    close = 100 + rng.standard_normal(n).cumsum()
    close = np.abs(close) + 50
    if uptrend:
        close[-1] = close.max() * 10
    else:
        close[-1] = 0.001
    return pd.DataFrame({
        "Open": close * 0.999,
        "High": close * 1.002,
        "Low": close * 0.998,
        "Close": close,
        "Volume": rng.integers(1_000_000, 10_000_000, size=n).astype(float),
    }, index=dates)


def _make_vix(n: int = 600) -> pd.DataFrame:
    return pd.DataFrame(
        {"Close": np.full(n, 20.0)},
        index=pd.date_range("2022-01-01", periods=n, freq="B"),
    )


def _mock_model(prob: float = 0.70) -> MagicMock:
    m = MagicMock(spec=XGBClassifier)
    m.predict_proba.return_value = np.array([[1 - prob, prob]])
    return m


def _x_mock() -> pd.DataFrame:
    return pd.DataFrame([[0.0] * len(_FEATURE_COLS)], columns=_FEATURE_COLS)


# ── Regime filter ──────────────────────────────────────────────────────────────

def test_compute_regime_in_uptrend():
    from scan import _compute_regime
    closes = [100.0] * 199 + [9999.0]
    dates = pd.date_range("2020-01-01", periods=200, freq="B")
    prices = pd.DataFrame({"Close": closes}, index=dates)
    regime_ok, sma_200 = _compute_regime(prices)
    assert regime_ok is True
    assert sma_200 < 9999.0


def test_compute_regime_below_sma():
    from scan import _compute_regime
    closes = [200.0] * 199 + [0.001]
    dates = pd.date_range("2020-01-01", periods=200, freq="B")
    prices = pd.DataFrame({"Close": closes}, index=dates)
    regime_ok, _ = _compute_regime(prices)
    assert regime_ok is False


# ── Signal suppression by regime ───────────────────────────────────────────────

def test_regime_blocks_high_confidence_signal():
    from scan import scan
    prices = _make_prices(uptrend=False)
    vix = _make_vix()
    model = _mock_model(prob=0.85)

    with patch("scan.fetch_prices", return_value=prices), \
         patch("scan.fetch_vix", return_value=vix), \
         patch("scan.model_is_stale", return_value=False), \
         patch("scan.load_model", return_value=model), \
         patch("scan.fetch_earnings_dates", return_value=pd.DatetimeIndex([])), \
         patch("scan.engineer_features_inference", return_value=_x_mock()):
        df = scan(["FAKE"], threshold=0.52)

    assert df.loc[0, "signal"] == 0
    assert df.loc[0, "regime_ok"] == False


def test_regime_passes_signal_when_in_uptrend():
    from scan import scan
    prices = _make_prices(uptrend=True)
    vix = _make_vix()
    model = _mock_model(prob=0.85)

    with patch("scan.fetch_prices", return_value=prices), \
         patch("scan.fetch_vix", return_value=vix), \
         patch("scan.model_is_stale", return_value=False), \
         patch("scan.load_model", return_value=model), \
         patch("scan.fetch_earnings_dates", return_value=pd.DatetimeIndex([])), \
         patch("scan.engineer_features_inference", return_value=_x_mock()):
        df = scan(["FAKE"], threshold=0.52)

    assert df.loc[0, "signal"] == 1
    assert df.loc[0, "regime_ok"] == True


# ── Output shape ───────────────────────────────────────────────────────────────

def test_scanner_output_columns():
    from scan import scan
    prices = _make_prices(uptrend=True)
    vix = _make_vix()
    model = _mock_model(prob=0.60)

    with patch("scan.fetch_prices", return_value=prices), \
         patch("scan.fetch_vix", return_value=vix), \
         patch("scan.model_is_stale", return_value=False), \
         patch("scan.load_model", return_value=model), \
         patch("scan.fetch_earnings_dates", return_value=pd.DatetimeIndex([])), \
         patch("scan.engineer_features_inference", return_value=_x_mock()):
        df = scan(["T1", "T2"], threshold=0.52)

    assert df.shape == (2, 7)
    assert set(df.columns) == {"ticker", "prob", "signal", "regime_ok", "near_earnings", "close", "sma_200"}


# ── CSV output ─────────────────────────────────────────────────────────────────

def test_scanner_csv_saved(tmp_path, monkeypatch):
    from scan import main
    monkeypatch.chdir(tmp_path)
    prices = _make_prices(uptrend=True)
    vix = _make_vix()
    model = _mock_model(prob=0.70)

    with patch("scan.fetch_prices", return_value=prices), \
         patch("scan.fetch_vix", return_value=vix), \
         patch("scan.model_is_stale", return_value=False), \
         patch("scan.load_model", return_value=model), \
         patch("scan.fetch_earnings_dates", return_value=pd.DatetimeIndex([])), \
         patch("scan.engineer_features_inference", return_value=_x_mock()):
        main(tickers=["FAKE"], threshold=0.52)

    today = date.today().isoformat()
    csv_path = tmp_path / "signals" / f"{today}.csv"
    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    assert {"date", "ticker", "prob", "signal", "regime_ok", "close", "sma_200"}.issubset(df.columns)


# ── Model caching ──────────────────────────────────────────────────────────────

def test_scanner_uses_cached_model():
    from scan import scan
    prices = _make_prices(uptrend=True)
    vix = _make_vix()
    model = _mock_model(prob=0.60)

    with patch("scan.fetch_prices", return_value=prices), \
         patch("scan.fetch_vix", return_value=vix), \
         patch("scan.model_is_stale", return_value=False), \
         patch("scan.load_model", return_value=model) as mock_load, \
         patch("scan.walk_forward_predict") as mock_train, \
         patch("scan.fetch_earnings_dates", return_value=pd.DatetimeIndex([])), \
         patch("scan.engineer_features_inference", return_value=_x_mock()):
        scan(["FAKE"], threshold=0.52)

    mock_load.assert_called_once_with("FAKE")
    mock_train.assert_not_called()


def test_scanner_retrains_stale_model():
    from scan import scan
    prices = _make_prices(uptrend=True)
    vix = _make_vix()
    model = _mock_model(prob=0.60)
    fake_signals = pd.DataFrame({"signal": [1], "prob": [0.60]})

    with patch("scan.fetch_prices", return_value=prices), \
         patch("scan.fetch_vix", return_value=vix), \
         patch("scan.model_is_stale", return_value=True), \
         patch("scan.engineer_features", return_value=(pd.DataFrame(), pd.Series())), \
         patch("scan.walk_forward_predict", return_value=(fake_signals, model)) as mock_train, \
         patch("scan.save_model") as mock_save, \
         patch("scan.fetch_earnings_dates", return_value=pd.DatetimeIndex([])), \
         patch("scan.engineer_features_inference", return_value=_x_mock()):
        scan(["FAKE"], threshold=0.52)

    mock_train.assert_called_once()
    mock_save.assert_called_once_with(model, "FAKE")


# ── Bad ticker skipped ─────────────────────────────────────────────────────────

def test_bad_ticker_skipped_gracefully():
    from scan import scan

    def raise_on_bad(ticker, *args, **kwargs):
        if ticker == "BAD":
            raise ValueError("No data returned for BAD.")
        return _make_prices(uptrend=True)

    with patch("scan.fetch_prices", side_effect=raise_on_bad), \
         patch("scan.fetch_vix", return_value=_make_vix()), \
         patch("scan.model_is_stale", return_value=False), \
         patch("scan.load_model", return_value=_mock_model()), \
         patch("scan.fetch_earnings_dates", return_value=pd.DatetimeIndex([])), \
         patch("scan.engineer_features_inference", return_value=_x_mock()):
        df = scan(["GOOD", "BAD"], threshold=0.52)

    assert len(df) == 1
    assert df.loc[0, "ticker"] == "GOOD"
