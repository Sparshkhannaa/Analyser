# Phase 2 — Multi-Ticker Scanner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a daily scanner (`scan.py`) that loads/trains XGBoost models for up to 20 stocks, applies a 200-day SMA regime filter, ranks signals by confidence, and outputs a terminal table plus a dated CSV file.

**Architecture:** Extract `_build_feature_data` from `features.py` and add `engineer_features_inference` so the scanner can predict on today's data without a future target. Add `save_model`/`load_model`/`model_is_stale` to `model.py` using joblib. Create `scan.py` that reuses all existing modules. `backtest.py` and `main.py` are untouched.

**Tech Stack:** Python 3.11, XGBoost, joblib, pandas, yfinance, pytest (existing stack).

---

## Task 1: `engineer_features_inference` in `features.py`

**Why a new function:** `engineer_features` drops the last 5 rows because their 5-day forward target is NaN. For live prediction we need today's features without a target. Solution: extract the shared feature computation into `_build_feature_data`, then call it from both `engineer_features` (adds target + full dropna) and the new `engineer_features_inference` (drops NaN on feature cols only, returns last row).

**Files:**
- Modify: `features.py`
- Modify: `tests/test_features.py`

---

- [ ] **Step 1: Write the failing test**

Add to the bottom of `tests/test_features.py`:

```python
def test_engineer_features_inference_returns_one_row(synthetic_prices, synthetic_vix):
    from features import engineer_features_inference, _FEATURE_COLS
    row = engineer_features_inference(synthetic_prices, synthetic_vix)
    assert row.shape == (1, len(_FEATURE_COLS)), f"Expected (1, 44), got {row.shape}"
    assert list(row.columns) == _FEATURE_COLS
    assert not row.isnull().values.any(), "Inference row contains NaN"


def test_engineer_features_inference_index_after_training_data(synthetic_prices, synthetic_vix):
    from features import engineer_features, engineer_features_inference
    X_train, _ = engineer_features(synthetic_prices, synthetic_vix)
    X_infer = engineer_features_inference(synthetic_prices, synthetic_vix)
    assert X_infer.index[-1] >= X_train.index[-1], (
        "Inference row should be at or after last training row"
    )
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /Users/sparshkhanna/Documents/GitHub/Analyser
pytest tests/test_features.py::test_engineer_features_inference_returns_one_row tests/test_features.py::test_engineer_features_inference_index_after_training_data -v
```

Expected: `ImportError: cannot import name 'engineer_features_inference'`

- [ ] **Step 3: Refactor `features.py` — extract `_build_feature_data`, add `engineer_features_inference`**

Replace the entire `engineer_features` function (lines 72–176) in `features.py` with the following three functions. Insert them after `_compute_adx` and before `_FEATURE_COLS`:

```python
def _build_feature_data(prices: pd.DataFrame, vix: pd.DataFrame) -> pd.DataFrame:
    data = prices.copy()
    close = data["Close"]
    high = data["High"]
    low = data["Low"]
    volume = data["Volume"]

    for lag in range(1, 6):
        data[f"lag_{lag}"] = close.shift(lag)

    data["sma_7"] = close.rolling(7).mean()
    data["sma_14"] = close.rolling(14).mean()
    data["sma_21"] = close.rolling(21).mean()
    data["ema_12"] = close.ewm(span=12, adjust=False).mean()
    data["ema_26"] = close.ewm(span=26, adjust=False).mean()

    data["macd"] = data["ema_12"] - data["ema_26"]
    data["macd_signal"] = data["macd"].ewm(span=9, adjust=False).mean()
    data["macd_diff"] = data["macd"] - data["macd_signal"]

    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    data["rsi"] = 100 - (100 / (1 + gain / loss))

    data["bb_middle"] = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    data["bb_upper"] = data["bb_middle"] + 2 * bb_std
    data["bb_lower"] = data["bb_middle"] - 2 * bb_std
    data["bb_width"] = (data["bb_upper"] - data["bb_lower"]) / data["bb_middle"]
    data["bb_position"] = (close - data["bb_lower"]) / (data["bb_upper"] - data["bb_lower"])

    data["volatility"] = close.pct_change().rolling(20).std()
    data["atr"] = high - low
    data["atr_14"] = data["atr"].rolling(14).mean()

    data["pct_change_1"] = close.pct_change(1)
    data["pct_change_5"] = close.pct_change(5)
    data["pct_change_10"] = close.pct_change(10)

    data["volume_change_1"] = volume.pct_change(1)
    data["volume_sma_20"] = volume.rolling(20).mean()
    data["volume_ratio"] = volume / data["volume_sma_20"]

    data["momentum_5"] = close - close.shift(5)
    data["momentum_10"] = close - close.shift(10)
    data["high_low_ratio"] = (close - low) / (high - low)

    vix_close = vix["Close"].reindex(data.index, method="ffill")
    data["vix"] = vix_close
    data["vix_change_5"] = vix_close.pct_change(5)

    data["adx"] = _compute_adx(high, low, close)

    low_14 = low.rolling(14).min()
    high_14 = high.rolling(14).max()
    stoch_k = 100 * (close - low_14) / (high_14 - low_14)
    data["stoch_k"] = stoch_k
    data["stoch_d"] = stoch_k.rolling(3).mean()

    data["williams_r"] = -100 * (high_14 - close) / (high_14 - low_14)

    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    data["obv"] = obv
    data["obv_slope"] = obv.diff(5)

    data["roc_5"] = close.pct_change(5) * 100
    data["roc_10"] = close.pct_change(10) * 100

    data["day_of_week"] = data.index.dayofweek.astype(float)

    rolling_high = close.rolling(252).max()
    rolling_low = close.rolling(252).min()
    data["dist_52w_high"] = (close - rolling_high) / rolling_high
    data["dist_52w_low"] = (close - rolling_low) / rolling_low

    return data


def engineer_features(
    prices: pd.DataFrame, vix: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series]:
    data = _build_feature_data(prices, vix)
    close = prices["Close"]
    fwd_close = close.shift(-5)
    data["target"] = fwd_close.gt(close).astype(float).where(fwd_close.notna())
    data = data.dropna()
    return data[_FEATURE_COLS], data["target"]


def engineer_features_inference(prices: pd.DataFrame, vix: pd.DataFrame) -> pd.DataFrame:
    """Return the single most-recent feature row for live prediction (no target needed)."""
    data = _build_feature_data(prices, vix)
    data = data.dropna(subset=_FEATURE_COLS)
    return data[_FEATURE_COLS].iloc[[-1]]
```

The complete new `features.py` should look like this (replace the whole file):

```python
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta


def fetch_prices(ticker: str, period_days: int = 1825) -> pd.DataFrame:
    end = datetime.utcnow().date()
    start = end - timedelta(days=period_days)
    df = yf.download(
        ticker,
        start=start.isoformat(),
        end=end.isoformat(),
        interval="1d",
        progress=False,
    )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}.")
    df.index = pd.to_datetime(df.index)
    return df


def fetch_vix(period_days: int = 1825) -> pd.DataFrame:
    return fetch_prices("^VIX", period_days)


def _compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)

    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=close.index
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=close.index
    )

    alpha = 1.0 / period
    tr_s = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_s = plus_dm.ewm(alpha=alpha, adjust=False).mean()
    minus_s = minus_dm.ewm(alpha=alpha, adjust=False).mean()

    plus_di = 100 * plus_s / tr_s
    minus_di = 100 * minus_s / tr_s
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=alpha, adjust=False).mean()


_FEATURE_COLS = [
    "lag_1", "lag_2", "lag_3", "lag_4", "lag_5",
    "sma_7", "sma_14", "sma_21", "ema_12", "ema_26",
    "macd", "macd_signal", "macd_diff", "rsi",
    "bb_middle", "bb_upper", "bb_lower", "bb_width", "bb_position",
    "volatility", "atr", "atr_14",
    "pct_change_1", "pct_change_5", "pct_change_10",
    "volume_change_1", "volume_sma_20", "volume_ratio",
    "momentum_5", "momentum_10", "high_low_ratio",
    "vix", "vix_change_5", "adx",
    "stoch_k", "stoch_d", "williams_r",
    "obv", "obv_slope", "roc_5", "roc_10",
    "day_of_week", "dist_52w_high", "dist_52w_low",
]


def _build_feature_data(prices: pd.DataFrame, vix: pd.DataFrame) -> pd.DataFrame:
    data = prices.copy()
    close = data["Close"]
    high = data["High"]
    low = data["Low"]
    volume = data["Volume"]

    for lag in range(1, 6):
        data[f"lag_{lag}"] = close.shift(lag)

    data["sma_7"] = close.rolling(7).mean()
    data["sma_14"] = close.rolling(14).mean()
    data["sma_21"] = close.rolling(21).mean()
    data["ema_12"] = close.ewm(span=12, adjust=False).mean()
    data["ema_26"] = close.ewm(span=26, adjust=False).mean()

    data["macd"] = data["ema_12"] - data["ema_26"]
    data["macd_signal"] = data["macd"].ewm(span=9, adjust=False).mean()
    data["macd_diff"] = data["macd"] - data["macd_signal"]

    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    data["rsi"] = 100 - (100 / (1 + gain / loss))

    data["bb_middle"] = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    data["bb_upper"] = data["bb_middle"] + 2 * bb_std
    data["bb_lower"] = data["bb_middle"] - 2 * bb_std
    data["bb_width"] = (data["bb_upper"] - data["bb_lower"]) / data["bb_middle"]
    data["bb_position"] = (close - data["bb_lower"]) / (data["bb_upper"] - data["bb_lower"])

    data["volatility"] = close.pct_change().rolling(20).std()
    data["atr"] = high - low
    data["atr_14"] = data["atr"].rolling(14).mean()

    data["pct_change_1"] = close.pct_change(1)
    data["pct_change_5"] = close.pct_change(5)
    data["pct_change_10"] = close.pct_change(10)

    data["volume_change_1"] = volume.pct_change(1)
    data["volume_sma_20"] = volume.rolling(20).mean()
    data["volume_ratio"] = volume / data["volume_sma_20"]

    data["momentum_5"] = close - close.shift(5)
    data["momentum_10"] = close - close.shift(10)
    data["high_low_ratio"] = (close - low) / (high - low)

    vix_close = vix["Close"].reindex(data.index, method="ffill")
    data["vix"] = vix_close
    data["vix_change_5"] = vix_close.pct_change(5)

    data["adx"] = _compute_adx(high, low, close)

    low_14 = low.rolling(14).min()
    high_14 = high.rolling(14).max()
    stoch_k = 100 * (close - low_14) / (high_14 - low_14)
    data["stoch_k"] = stoch_k
    data["stoch_d"] = stoch_k.rolling(3).mean()

    data["williams_r"] = -100 * (high_14 - close) / (high_14 - low_14)

    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    data["obv"] = obv
    data["obv_slope"] = obv.diff(5)

    data["roc_5"] = close.pct_change(5) * 100
    data["roc_10"] = close.pct_change(10) * 100

    data["day_of_week"] = data.index.dayofweek.astype(float)

    rolling_high = close.rolling(252).max()
    rolling_low = close.rolling(252).min()
    data["dist_52w_high"] = (close - rolling_high) / rolling_high
    data["dist_52w_low"] = (close - rolling_low) / rolling_low

    return data


def engineer_features(
    prices: pd.DataFrame, vix: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series]:
    data = _build_feature_data(prices, vix)
    close = prices["Close"]
    fwd_close = close.shift(-5)
    data["target"] = fwd_close.gt(close).astype(float).where(fwd_close.notna())
    data = data.dropna()
    return data[_FEATURE_COLS], data["target"]


def engineer_features_inference(prices: pd.DataFrame, vix: pd.DataFrame) -> pd.DataFrame:
    """Return the single most-recent feature row for live prediction (no target needed)."""
    data = _build_feature_data(prices, vix)
    data = data.dropna(subset=_FEATURE_COLS)
    return data[_FEATURE_COLS].iloc[[-1]]
```

- [ ] **Step 4: Run the new tests to confirm they pass**

```bash
pytest tests/test_features.py -v
```

Expected: all 9 tests pass (7 existing + 2 new).

- [ ] **Step 5: Commit**

```bash
git add features.py tests/test_features.py
git commit -m "feat: add engineer_features_inference for live prediction"
```

---

## Task 2: Model Persistence in `model.py`

Add three functions: `save_model`, `load_model`, `model_is_stale`. Models are serialised with joblib. Metadata (trained timestamp) stored as JSON alongside the pickle.

**Files:**
- Modify: `model.py`
- Modify: `requirements.txt`
- Modify: `tests/test_model.py`

---

- [ ] **Step 1: Write the failing tests**

Add to the bottom of `tests/test_model.py`:

```python
import json
import os
from datetime import datetime, timedelta, timezone


def test_save_load_roundtrip(tmp_path, monkeypatch):
    from model import save_model, load_model, train_xgb, MODEL_DIR
    monkeypatch.setattr("model.MODEL_DIR", str(tmp_path))

    X, y = engineer_features(synthetic_prices(), synthetic_vix())
    model = train_xgb(X, y)
    save_model(model, "TEST")

    loaded = load_model("TEST")
    preds_orig = model.predict_proba(X)[:, 1]
    preds_load = loaded.predict_proba(X)[:, 1]
    np.testing.assert_array_almost_equal(preds_orig, preds_load)


def test_model_is_stale_missing(tmp_path, monkeypatch):
    from model import model_is_stale
    monkeypatch.setattr("model.MODEL_DIR", str(tmp_path))
    assert model_is_stale("MISSING") is True


def test_model_is_stale_fresh(tmp_path, monkeypatch):
    from model import model_is_stale
    monkeypatch.setattr("model.MODEL_DIR", str(tmp_path))

    meta = {"trained_at": datetime.now(timezone.utc).isoformat()}
    (tmp_path / "FRESH_xgb.pkl").write_text("dummy")
    (tmp_path / "FRESH_meta.json").write_text(json.dumps(meta))

    assert model_is_stale("FRESH") is False


def test_model_is_stale_old(tmp_path, monkeypatch):
    from model import model_is_stale
    monkeypatch.setattr("model.MODEL_DIR", str(tmp_path))

    old_time = datetime.now(timezone.utc) - timedelta(days=8)
    meta = {"trained_at": old_time.isoformat()}
    (tmp_path / "OLD_xgb.pkl").write_text("dummy")
    (tmp_path / "OLD_meta.json").write_text(json.dumps(meta))

    assert model_is_stale("OLD") is True
```

The `synthetic_prices()` and `synthetic_vix()` calls in `test_save_load_roundtrip` need direct construction since fixtures aren't available outside pytest fixture scope. Replace those calls with:

```python
def test_save_load_roundtrip(tmp_path, monkeypatch, synthetic_prices, synthetic_vix):
    from model import save_model, load_model, train_xgb
    monkeypatch.setattr("model.MODEL_DIR", str(tmp_path))

    X, y = engineer_features(synthetic_prices, synthetic_vix)
    model = train_xgb(X, y)
    save_model(model, "TEST")

    loaded = load_model("TEST")
    preds_orig = model.predict_proba(X)[:, 1]
    preds_load = loaded.predict_proba(X)[:, 1]
    np.testing.assert_array_almost_equal(preds_orig, preds_load)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_model.py::test_save_load_roundtrip tests/test_model.py::test_model_is_stale_missing tests/test_model.py::test_model_is_stale_fresh tests/test_model.py::test_model_is_stale_old -v
```

Expected: `ImportError: cannot import name 'save_model'` (or similar).

- [ ] **Step 3: Add `joblib` to `requirements.txt`**

Add this line to `requirements.txt`:

```
joblib>=1.3.0
```

- [ ] **Step 4: Implement persistence functions in `model.py`**

Add these imports at the top of `model.py` (after existing imports):

```python
import json
import os
from datetime import datetime, timezone

import joblib
```

Add this constant and three functions after the existing `train_xgb` and `walk_forward_predict` functions:

```python
MODEL_DIR = "models"


def save_model(model: XGBClassifier, ticker: str) -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, f"{MODEL_DIR}/{ticker}_xgb.pkl")
    meta = {"trained_at": datetime.now(timezone.utc).isoformat()}
    with open(f"{MODEL_DIR}/{ticker}_meta.json", "w") as f:
        json.dump(meta, f)


def load_model(ticker: str) -> XGBClassifier:
    return joblib.load(f"{MODEL_DIR}/{ticker}_xgb.pkl")


def model_is_stale(ticker: str, max_age_days: int = 7) -> bool:
    pkl_path = f"{MODEL_DIR}/{ticker}_xgb.pkl"
    meta_path = f"{MODEL_DIR}/{ticker}_meta.json"
    if not os.path.exists(pkl_path) or not os.path.exists(meta_path):
        return True
    with open(meta_path) as f:
        meta = json.load(f)
    trained_at = datetime.fromisoformat(meta["trained_at"])
    if trained_at.tzinfo is None:
        trained_at = trained_at.replace(tzinfo=timezone.utc)
    age = datetime.now(timezone.utc) - trained_at
    return age.days > max_age_days
```

- [ ] **Step 5: Run tests to confirm they pass**

```bash
pytest tests/test_model.py -v
```

Expected: all 11 tests pass (7 existing + 4 new).

- [ ] **Step 6: Commit**

```bash
git add model.py requirements.txt tests/test_model.py
git commit -m "feat: add model persistence (save/load/stale check) with joblib"
```

---

## Task 3: `scan.py` — Scanner Core

Build the scanner: regime filter, caching logic, per-ticker prediction loop. Output (table + CSV) comes in Task 4.

**Files:**
- Create: `scan.py`
- Create: `tests/test_scan.py`

---

- [ ] **Step 1: Write the failing tests**

Create `tests/test_scan.py`:

```python
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
         patch("scan.engineer_features_inference", return_value=_x_mock()):
        df = scan(["FAKE"], threshold=0.52)

    assert df.loc[0, "signal"] == 0
    assert df.loc[0, "regime_ok"] is False


def test_regime_passes_signal_when_in_uptrend():
    from scan import scan
    prices = _make_prices(uptrend=True)
    vix = _make_vix()
    model = _mock_model(prob=0.85)

    with patch("scan.fetch_prices", return_value=prices), \
         patch("scan.fetch_vix", return_value=vix), \
         patch("scan.model_is_stale", return_value=False), \
         patch("scan.load_model", return_value=model), \
         patch("scan.engineer_features_inference", return_value=_x_mock()):
        df = scan(["FAKE"], threshold=0.52)

    assert df.loc[0, "signal"] == 1
    assert df.loc[0, "regime_ok"] is True


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
         patch("scan.engineer_features_inference", return_value=_x_mock()):
        df = scan(["T1", "T2"], threshold=0.52)

    assert df.shape == (2, 6)
    assert set(df.columns) == {"ticker", "prob", "signal", "regime_ok", "close", "sma_200"}


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
         patch("scan.engineer_features_inference", return_value=_x_mock()):
        df = scan(["GOOD", "BAD"], threshold=0.52)

    assert len(df) == 1
    assert df.loc[0, "ticker"] == "GOOD"
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_scan.py -v
```

Expected: `ModuleNotFoundError: No module named 'scan'`

- [ ] **Step 3: Create `scan.py`**

Create `/Users/sparshkhanna/Documents/GitHub/Analyser/scan.py`:

```python
import argparse
import os
import warnings
from datetime import date, datetime, timezone

warnings.filterwarnings("ignore")

import pandas as pd

from features import (
    engineer_features,
    engineer_features_inference,
    fetch_prices,
    fetch_vix,
)
from model import load_model, model_is_stale, save_model, walk_forward_predict

DEFAULT_TICKERS = [
    "SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "GOOGL",
    "META", "AMD", "NFLX", "JPM", "V", "UNH", "XOM", "BABA",
    "COIN", "PLTR", "SMCI", "MSTR",
]


def _compute_regime(prices: pd.DataFrame) -> tuple[bool, float]:
    sma_200 = float(prices["Close"].rolling(200).mean().iloc[-1])
    close_today = float(prices["Close"].iloc[-1])
    return close_today > sma_200, sma_200


def _get_or_train_model(
    ticker: str,
    prices: pd.DataFrame,
    vix: pd.DataFrame,
    threshold: float,
    force_retrain: bool,
):
    if not force_retrain and not model_is_stale(ticker):
        print(f"  [{ticker}] Loading cached model...")
        return load_model(ticker)
    print(f"  [{ticker}] Training model (walk-forward, ~2 min)...")
    X, y = engineer_features(prices, vix)
    _, model = walk_forward_predict(X, y, threshold=threshold)
    save_model(model, ticker)
    return model


def scan(
    tickers: list[str],
    period: int = 1825,
    threshold: float = 0.52,
    force_retrain: bool = False,
) -> pd.DataFrame:
    results = []
    for ticker in tickers:
        try:
            prices = fetch_prices(ticker, period)
            vix = fetch_vix(period)
            model = _get_or_train_model(ticker, prices, vix, threshold, force_retrain)
            X_today = engineer_features_inference(prices, vix)
            prob = float(model.predict_proba(X_today)[:, 1][0])
            regime_ok, sma_200 = _compute_regime(prices)
            signal = 1 if (prob >= threshold and regime_ok) else 0
            results.append({
                "ticker": ticker,
                "prob": prob,
                "signal": signal,
                "regime_ok": regime_ok,
                "close": float(prices["Close"].iloc[-1]),
                "sma_200": sma_200,
            })
        except Exception as e:
            print(f"  [{ticker}] SKIPPED — {e}")
            continue

    if not results:
        return pd.DataFrame(columns=["ticker", "prob", "signal", "regime_ok", "close", "sma_200"])

    return (
        pd.DataFrame(results)
        .sort_values("prob", ascending=False)
        .reset_index(drop=True)
    )


def _print_results(df: pd.DataFrame, threshold: float) -> None:
    today = date.today().isoformat()
    n = len(df)
    longs = int((df["signal"] == 1).sum())
    print(f"\n{'=' * 65}")
    print(f"  SCANNER — {today}  |  threshold: {threshold}  |  {n} tickers")
    print(f"{'=' * 65}")
    print(f"  {'Ticker':<8} {'Prob':>6}  {'Signal':<8} {'Regime'}")
    print(f"  {'-' * 7} {'-' * 6}  {'-' * 7} {'-' * 7}")
    for _, row in df.iterrows():
        signal_str = "LONG" if row["signal"] == 1 else "-"
        regime_str = "YES" if row["regime_ok"] else "NO"
        print(f"  {row['ticker']:<8} {row['prob']:>6.2f}  {signal_str:<8} {regime_str}")
    print(f"{'=' * 65}")
    print(f"  {longs} long signal{'s' if longs != 1 else ''}")
    print(f"{'=' * 65}\n")


def _save_csv(df: pd.DataFrame) -> str:
    os.makedirs("signals", exist_ok=True)
    today = date.today().isoformat()
    path = f"signals/{today}.csv"
    out = df.copy()
    out.insert(0, "date", today)
    out.to_csv(path, index=False)
    return path


def main(
    tickers: list[str] | None = None,
    period: int = 1825,
    threshold: float = 0.52,
    force_retrain: bool = False,
) -> pd.DataFrame:
    tickers = tickers or DEFAULT_TICKERS
    print(f"\nScanning {len(tickers)} ticker{'s' if len(tickers) != 1 else ''}...")
    df = scan(tickers, period=period, threshold=threshold, force_retrain=force_retrain)
    if df.empty:
        print("No results — all tickers skipped.")
        return df
    _print_results(df, threshold)
    path = _save_csv(df)
    print(f"Saved: {path}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-ticker XGBoost Scanner")
    parser.add_argument(
        "--tickers", nargs="+", default=None,
        help="Tickers to scan (default: 20 large-caps)"
    )
    parser.add_argument(
        "--period", type=int, default=1825,
        help="Days of history to fetch (default: 1825)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.52,
        help="Signal confidence threshold (default: 0.52)"
    )
    parser.add_argument(
        "--retrain", action="store_true",
        help="Force retrain all models regardless of cache age"
    )
    args = parser.parse_args()
    main(
        tickers=args.tickers,
        period=args.period,
        threshold=args.threshold,
        force_retrain=args.retrain,
    )
```

- [ ] **Step 4: Run all tests to confirm they pass**

```bash
pytest tests/test_scan.py -v
```

Expected: all 9 tests pass.

- [ ] **Step 5: Run the full test suite to confirm no regressions**

```bash
pytest -v
```

Expected: all tests pass (22 existing + 2 features + 4 model + 9 scan = 37 total).

- [ ] **Step 6: Update `.gitignore` to exclude generated directories**

Add these two lines at the bottom of `.gitignore`:

```
models/
signals/
```

- [ ] **Step 7: Commit**

```bash
git add scan.py tests/test_scan.py .gitignore
git commit -m "feat: add multi-ticker scanner with regime filter and model caching"
```

---

## Task 4: Smoke Test

Verify the scanner works end-to-end on real market data with a small ticker list (avoids waiting 40+ min for 20 tickers to train).

**Files:** None (read-only verification)

---

- [ ] **Step 1: Run a 3-ticker smoke test with `--retrain`**

```bash
cd /Users/sparshkhanna/Documents/GitHub/Analyser
python scan.py --tickers NVDA TSLA SPY --retrain
```

Expected output (values will differ, structure must match):
```
Scanning 3 tickers...
  [NVDA] Training model (walk-forward, ~2 min)...
  [TSLA] Training model (walk-forward, ~2 min)...
  [SPY] Training model (walk-forward, ~2 min)...

=================================================================
  SCANNER — 2026-05-11  |  threshold: 0.52  |  3 tickers
=================================================================
  Ticker   Prob    Signal   Regime
  ------- ------  ------- -------
  NVDA     0.XX    LONG     YES
  ...
=================================================================
  X long signals
=================================================================

Saved: signals/2026-05-11.csv
```

- [ ] **Step 2: Run again without `--retrain` to confirm caching works**

```bash
python scan.py --tickers NVDA TSLA SPY
```

Expected: output shows `Loading cached model...` for each ticker, runs in under 10 seconds total.

- [ ] **Step 3: Confirm CSV was written**

```bash
ls -la signals/
```

Expected: `signals/YYYY-MM-DD.csv` exists with today's date.

- [ ] **Step 4: Run full test suite one final time**

```bash
pytest -v
```

Expected: all tests pass.

---

## Self-Review Checklist

After writing this plan, verified against spec:

- [x] `engineer_features_inference` — Task 1 ✓
- [x] `save_model` / `load_model` / `model_is_stale` — Task 2 ✓
- [x] Regime filter (`_compute_regime`, 200-day SMA) — Task 3 ✓
- [x] Scanner loop with error handling per ticker — Task 3 ✓
- [x] Model caching (7-day staleness, `--retrain` flag) — Task 3 ✓
- [x] Terminal table output — Task 3 (`_print_results`) ✓
- [x] CSV output to `signals/YYYY-MM-DD.csv` — Task 3 (`_save_csv`) ✓
- [x] CLI with `--tickers`, `--period`, `--threshold`, `--retrain` — Task 3 ✓
- [x] Default 20-ticker list — Task 3 ✓
- [x] `.gitignore` updated for `models/` and `signals/` — Task 3 ✓
- [x] All 9 test cases from spec covered — Task 1 (2) + Task 2 (4) + Task 3 (9) ✓
- [x] `joblib` added to `requirements.txt` — Task 2 ✓
