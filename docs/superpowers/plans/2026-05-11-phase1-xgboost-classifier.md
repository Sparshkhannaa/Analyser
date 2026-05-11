# Phase 1 — XGBoost Direction Classifier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the broken LSTM price-regression model with an XGBoost direction classifier (up/down), add 13 new features including VIX, walk-forward validation, and realistic backtesting with transaction costs — split across 4 focused modules.

**Architecture:** Single `main.py` is decomposed into `features.py` (data + indicators), `model.py` (XGBoost + walk-forward), `backtest.py` (simulation + metrics), and a thin `main.py` orchestrator. All modules are tested before integration. Old `main.py` is replaced wholesale — no backward compatibility needed.

**Tech Stack:** Python 3.12, XGBoost 1.7+, yfinance, pandas, numpy, matplotlib, pytest

---

## File Map

| File | Status | Responsibility |
|---|---|---|
| `features.py` | Create | Fetch prices + VIX, compute 44 indicators, build binary target |
| `model.py` | Create | Train XGBoost classifier, walk-forward validation |
| `backtest.py` | Create | Simulate trades with costs, compute Sharpe/Sortino/Calmar |
| `main.py` | Replace | CLI entry point, print results, save charts |
| `requirements.txt` | Modify | Add xgboost |
| `tests/conftest.py` | Create | Shared synthetic data fixtures |
| `tests/test_features.py` | Create | Feature engineering tests |
| `tests/test_model.py` | Create | Model training and walk-forward tests |
| `tests/test_backtest.py` | Create | Backtest simulation and metrics tests |

---

## Task 1: Setup — requirements.txt + test scaffold

**Files:**
- Modify: `requirements.txt`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Update requirements.txt**

Replace the entire file with:

```
yfinance>=0.2.28
pandas>=2.0.0
numpy>=1.24.0
xgboost>=1.7.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
pytest>=7.4.0
```

- [ ] **Step 2: Install new dependencies**

Run: `pip install xgboost pytest`

Expected: `Successfully installed xgboost-...` (no errors)

- [ ] **Step 3: Create tests/__init__.py**

Create an empty file at `tests/__init__.py`.

- [ ] **Step 4: Create tests/conftest.py with synthetic fixtures**

```python
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_prices():
    np.random.seed(42)
    n = 600
    dates = pd.bdate_range("2020-01-01", periods=n)
    close = 400 + np.cumsum(np.random.randn(n) * 2)
    high = close + np.abs(np.random.randn(n))
    low = close - np.abs(np.random.randn(n))
    open_ = close + np.random.randn(n) * 0.5
    volume = np.random.randint(50_000_000, 150_000_000, n).astype(float)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


@pytest.fixture
def synthetic_vix():
    np.random.seed(43)
    n = 600
    dates = pd.bdate_range("2020-01-01", periods=n)
    vix = np.clip(15 + np.random.randn(n) * 5, 10, 80)
    return pd.DataFrame({"Close": vix}, index=dates)
```

- [ ] **Step 5: Verify pytest is found**

Run: `pytest --collect-only`

Expected: `no tests ran` (no errors — test discovery works)

- [ ] **Step 6: Commit**

```bash
git add requirements.txt tests/
git commit -m "feat: add xgboost dependency and test scaffold"
```

---

## Task 2: features.py — data fetching functions

**Files:**
- Create: `features.py`
- Create: `tests/test_features.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_features.py`:

```python
import pandas as pd
import pytest


def test_fetch_prices_returns_ohlcv(monkeypatch):
    import yfinance as yf
    import features

    sample = pd.DataFrame(
        {"Open": [100.0], "High": [101.0], "Low": [99.0], "Close": [100.5], "Volume": [1_000_000.0]},
        index=pd.DatetimeIndex(["2024-01-02"]),
    )

    def mock_download(*args, **kwargs):
        return sample

    monkeypatch.setattr(yf, "download", mock_download)
    df = features.fetch_prices("SPY", period_days=5)
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert len(df) == 1


def test_fetch_vix_uses_vix_ticker(monkeypatch):
    import yfinance as yf
    import features

    captured = {}

    def mock_download(ticker, **kwargs):
        captured["ticker"] = ticker
        return pd.DataFrame(
            {"Open": [20.0], "High": [21.0], "Low": [19.0], "Close": [20.5], "Volume": [0.0]},
            index=pd.DatetimeIndex(["2024-01-02"]),
        )

    monkeypatch.setattr(yf, "download", mock_download)
    features.fetch_vix(period_days=5)
    assert captured["ticker"] == "^VIX"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_features.py -v`

Expected: `ModuleNotFoundError: No module named 'features'`

- [ ] **Step 3: Create features.py with fetch functions**

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_features.py -v`

Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add features.py tests/test_features.py
git commit -m "feat: add features.py with price and VIX fetch functions"
```

---

## Task 3: features.py — engineer_features (all 44 features + binary target)

**Files:**
- Modify: `features.py` — add `_compute_adx` and `engineer_features`
- Modify: `tests/test_features.py` — add feature engineering tests

- [ ] **Step 1: Add tests for engineer_features**

Append to `tests/test_features.py`:

```python
from features import engineer_features


def test_engineer_features_column_count(synthetic_prices, synthetic_vix):
    X, y = engineer_features(synthetic_prices, synthetic_vix)
    assert X.shape[1] == 44, f"Expected 44 features, got {X.shape[1]}"


def test_engineer_features_no_nan(synthetic_prices, synthetic_vix):
    X, y = engineer_features(synthetic_prices, synthetic_vix)
    assert not X.isna().any().any(), "Feature matrix contains NaN values"


def test_engineer_features_target_is_binary(synthetic_prices, synthetic_vix):
    X, y = engineer_features(synthetic_prices, synthetic_vix)
    assert set(y.unique()).issubset({0, 1}), f"Target has non-binary values: {y.unique()}"


def test_engineer_features_lengths_match(synthetic_prices, synthetic_vix):
    X, y = engineer_features(synthetic_prices, synthetic_vix)
    assert len(X) == len(y)


def test_engineer_features_no_lookahead(synthetic_prices, synthetic_vix):
    X, y = engineer_features(synthetic_prices, synthetic_vix)
    # The last row of X must be before the last date in prices
    # because target requires shift(-1) — last price row has no target
    assert X.index[-1] < synthetic_prices.index[-1]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_features.py::test_engineer_features_column_count -v`

Expected: `FAILED` with `ImportError` or `AttributeError: module 'features' has no attribute 'engineer_features'`

- [ ] **Step 3: Add _compute_adx helper to features.py**

Append after `fetch_vix` in `features.py`:

```python
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
```

- [ ] **Step 4: Add engineer_features to features.py**

Append after `_compute_adx` in `features.py`:

```python
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


def engineer_features(
    prices: pd.DataFrame, vix: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series]:
    data = prices.copy()
    close = data["Close"]
    high = data["High"]
    low = data["Low"]
    volume = data["Volume"]

    # Lag features
    for lag in range(1, 6):
        data[f"lag_{lag}"] = close.shift(lag)

    # Moving averages
    data["sma_7"] = close.rolling(7).mean()
    data["sma_14"] = close.rolling(14).mean()
    data["sma_21"] = close.rolling(21).mean()
    data["ema_12"] = close.ewm(span=12, adjust=False).mean()
    data["ema_26"] = close.ewm(span=26, adjust=False).mean()

    # MACD
    data["macd"] = data["ema_12"] - data["ema_26"]
    data["macd_signal"] = data["macd"].ewm(span=9, adjust=False).mean()
    data["macd_diff"] = data["macd"] - data["macd_signal"]

    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    data["rsi"] = 100 - (100 / (1 + gain / loss))

    # Bollinger Bands
    data["bb_middle"] = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    data["bb_upper"] = data["bb_middle"] + 2 * bb_std
    data["bb_lower"] = data["bb_middle"] - 2 * bb_std
    data["bb_width"] = (data["bb_upper"] - data["bb_lower"]) / data["bb_middle"]
    data["bb_position"] = (close - data["bb_lower"]) / (
        data["bb_upper"] - data["bb_lower"]
    )

    # Volatility
    data["volatility"] = close.pct_change().rolling(20).std()
    data["atr"] = high - low
    data["atr_14"] = data["atr"].rolling(14).mean()

    # Price changes
    data["pct_change_1"] = close.pct_change(1)
    data["pct_change_5"] = close.pct_change(5)
    data["pct_change_10"] = close.pct_change(10)

    # Volume
    data["volume_change_1"] = volume.pct_change(1)
    data["volume_sma_20"] = volume.rolling(20).mean()
    data["volume_ratio"] = volume / data["volume_sma_20"]

    # Momentum
    data["momentum_5"] = close - close.shift(5)
    data["momentum_10"] = close - close.shift(10)
    data["high_low_ratio"] = (close - low) / (high - low)

    # VIX (forward-filled to match trading days)
    vix_close = vix["Close"].reindex(data.index, method="ffill")
    data["vix"] = vix_close
    data["vix_change_5"] = vix_close.pct_change(5)

    # ADX
    data["adx"] = _compute_adx(high, low, close)

    # Stochastic
    low_14 = low.rolling(14).min()
    high_14 = high.rolling(14).max()
    stoch_k = 100 * (close - low_14) / (high_14 - low_14)
    data["stoch_k"] = stoch_k
    data["stoch_d"] = stoch_k.rolling(3).mean()

    # Williams %R
    data["williams_r"] = -100 * (high_14 - close) / (high_14 - low_14)

    # OBV
    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    data["obv"] = obv
    data["obv_slope"] = obv.diff(5)

    # Rate of Change
    data["roc_5"] = close.pct_change(5) * 100
    data["roc_10"] = close.pct_change(10) * 100

    # Calendar
    data["day_of_week"] = data.index.dayofweek.astype(float)

    # 52-week distances
    rolling_high = close.rolling(252).max()
    rolling_low = close.rolling(252).min()
    data["dist_52w_high"] = (close - rolling_high) / rolling_high
    data["dist_52w_low"] = (close - rolling_low) / rolling_low

    # Binary target: 1 if next-day close > today * 1.001
    data["target"] = (close.shift(-1) > close * 1.001).astype(int)

    data = data.dropna()

    return data[_FEATURE_COLS], data["target"]
```

- [ ] **Step 5: Run all feature tests**

Run: `pytest tests/test_features.py -v`

Expected: `7 passed`

- [ ] **Step 6: Commit**

```bash
git add features.py tests/test_features.py
git commit -m "feat: add engineer_features with 44 indicators and binary direction target"
```

---

## Task 4: model.py — train_xgb

**Files:**
- Create: `model.py`
- Create: `tests/test_model.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_model.py`:

```python
import numpy as np
import pandas as pd
import pytest
from features import engineer_features


def test_train_xgb_returns_probabilities(synthetic_prices, synthetic_vix):
    from model import train_xgb

    X, y = engineer_features(synthetic_prices, synthetic_vix)
    model = train_xgb(X, y)
    probs = model.predict_proba(X)

    assert probs.shape == (len(X), 2)
    assert (probs >= 0).all()
    assert (probs <= 1).all()


def test_train_xgb_probabilities_sum_to_one(synthetic_prices, synthetic_vix):
    from model import train_xgb

    X, y = engineer_features(synthetic_prices, synthetic_vix)
    model = train_xgb(X, y)
    probs = model.predict_proba(X)

    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_model.py -v`

Expected: `ModuleNotFoundError: No module named 'model'`

- [ ] **Step 3: Create model.py with train_xgb**

```python
import pandas as pd
from xgboost import XGBClassifier


def train_xgb(X: pd.DataFrame, y: pd.Series) -> XGBClassifier:
    split = int(len(X) * 0.8)
    X_tr, X_val = X.iloc[:split], X.iloc[split:]
    y_tr, y_val = y.iloc[:split], y.iloc[split:]

    model = XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="auc",
        early_stopping_rounds=30,
        random_state=42,
        verbosity=0,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return model
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_model.py::test_train_xgb_returns_probabilities tests/test_model.py::test_train_xgb_probabilities_sum_to_one -v`

Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add model.py tests/test_model.py
git commit -m "feat: add model.py with XGBoost classifier training"
```

---

## Task 5: model.py — walk_forward_predict

**Files:**
- Modify: `model.py` — add `walk_forward_predict`
- Modify: `tests/test_model.py` — add walk-forward tests

- [ ] **Step 1: Add walk-forward tests**

Append to `tests/test_model.py`:

```python
def test_walk_forward_returns_dataframe(synthetic_prices, synthetic_vix):
    from model import walk_forward_predict

    X, y = engineer_features(synthetic_prices, synthetic_vix)
    signals, model = walk_forward_predict(X, y, min_train_days=252, step_days=63)

    assert isinstance(signals, pd.DataFrame)
    assert set(signals.columns) == {"actual", "prob", "signal"}


def test_walk_forward_no_duplicate_dates(synthetic_prices, synthetic_vix):
    from model import walk_forward_predict

    X, y = engineer_features(synthetic_prices, synthetic_vix)
    signals, _ = walk_forward_predict(X, y, min_train_days=252, step_days=63)

    assert signals.index.is_unique, "Walk-forward produced duplicate prediction dates"


def test_walk_forward_signals_are_binary(synthetic_prices, synthetic_vix):
    from model import walk_forward_predict

    X, y = engineer_features(synthetic_prices, synthetic_vix)
    signals, _ = walk_forward_predict(X, y, min_train_days=252, step_days=63)

    assert set(signals["signal"].unique()).issubset({0, 1})


def test_walk_forward_dates_in_x_index(synthetic_prices, synthetic_vix):
    from model import walk_forward_predict

    X, y = engineer_features(synthetic_prices, synthetic_vix)
    signals, _ = walk_forward_predict(X, y, min_train_days=252, step_days=63)

    assert signals.index.isin(X.index).all(), "Signal dates not all in feature index"


def test_walk_forward_returns_last_model(synthetic_prices, synthetic_vix):
    from model import walk_forward_predict
    from xgboost import XGBClassifier

    X, y = engineer_features(synthetic_prices, synthetic_vix)
    _, model = walk_forward_predict(X, y, min_train_days=252, step_days=63)

    assert isinstance(model, XGBClassifier)
    assert hasattr(model, "feature_importances_")
```

- [ ] **Step 2: Run to verify failures**

Run: `pytest tests/test_model.py -v`

Expected: `2 passed, 5 failed` (train tests pass, walk-forward tests fail)

- [ ] **Step 3: Add walk_forward_predict to model.py**

Append to `model.py`:

```python
import numpy as np


def walk_forward_predict(
    X: pd.DataFrame,
    y: pd.Series,
    min_train_days: int = 504,
    step_days: int = 63,
    threshold: float = 0.55,
) -> tuple[pd.DataFrame, XGBClassifier]:
    records = []
    model = None
    n = len(X)
    start = min_train_days

    while start < n:
        end = min(start + step_days, n)

        model = train_xgb(X.iloc[:start], y.iloc[:start])
        probs = model.predict_proba(X.iloc[start:end])[:, 1]

        for i, (idx, prob) in enumerate(zip(X.index[start:end], probs)):
            records.append(
                {
                    "date": idx,
                    "actual": int(y.iloc[start + i]),
                    "prob": float(prob),
                    "signal": int(prob >= threshold),
                }
            )
        start = end

    signals = pd.DataFrame(records).set_index("date")
    return signals, model
```

- [ ] **Step 4: Run all model tests**

Run: `pytest tests/test_model.py -v`

Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add model.py tests/test_model.py
git commit -m "feat: add walk-forward validation to model.py"
```

---

## Task 6: backtest.py — run_backtest

**Files:**
- Create: `backtest.py`
- Create: `tests/test_backtest.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_backtest.py`:

```python
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
    prices = pd.DataFrame(
        {"Open": np.linspace(100, 120, n), "Close": np.linspace(100, 120, n)},
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
    prices = pd.DataFrame(
        {"Open": 100.0, "Close": np.linspace(100, 110, n)}, index=dates
    )
    signals = pd.DataFrame({"signal": 0, "prob": 0.3}, index=dates)

    bt = run_backtest(prices, signals, initial_capital=100_000.0)
    # With no signal, position_size is 0 — portfolio stays at initial capital (minus rounding)
    assert bt["portfolio_value"].iloc[-1] == pytest.approx(100_000.0, abs=100)
```

- [ ] **Step 2: Run to verify failures**

Run: `pytest tests/test_backtest.py -v`

Expected: `4 failed` with `ModuleNotFoundError: No module named 'backtest'`

- [ ] **Step 3: Create backtest.py with run_backtest**

```python
import numpy as np
import pandas as pd


def run_backtest(
    prices: pd.DataFrame,
    signals: pd.DataFrame,
    initial_capital: float = 100_000.0,
    cost_pct: float = 0.001,
) -> pd.DataFrame:
    df = prices[["Open", "Close"]].join(signals[["signal", "prob"]], how="inner")

    # Execute at next-day open; measure return to that day's close
    df["exec_open"] = df["Open"].shift(-1)
    df["next_close"] = df["Close"].shift(-1)
    df = df.dropna(subset=["exec_open", "next_close"])

    # Half-Kelly: position = (prob - 0.5) / 0.5, capped at 0.5, zero when no signal
    df["position_size"] = ((df["prob"] - 0.5) / 0.5).clip(0, 0.5) * df["signal"]

    # One-day return from exec_open to next_close
    df["trade_return"] = (df["next_close"] - df["exec_open"]) / df["exec_open"]

    # Cost on position change (round-trip spread)
    prev_pos = df["position_size"].shift(1).fillna(0)
    df["cost"] = (df["position_size"] - prev_pos).abs() * cost_pct
    df["strategy_return"] = df["position_size"] * df["trade_return"] - df["cost"]

    df["cumulative"] = (1 + df["strategy_return"]).cumprod()
    df["portfolio_value"] = initial_capital * df["cumulative"]

    # Buy-and-hold benchmark over same period
    df["bh_daily"] = df["Close"].pct_change().fillna(0)
    df["bh_cumulative"] = (1 + df["bh_daily"]).cumprod()
    df["bh_value"] = initial_capital * df["bh_cumulative"]

    return df
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_backtest.py -v`

Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add backtest.py tests/test_backtest.py
git commit -m "feat: add backtest.py with cost-aware simulation"
```

---

## Task 7: backtest.py — compute_metrics

**Files:**
- Modify: `backtest.py` — add `compute_metrics`
- Modify: `tests/test_backtest.py` — add metrics tests

- [ ] **Step 1: Add metrics tests**

Append to `tests/test_backtest.py`:

```python
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
```

- [ ] **Step 2: Run to verify failures**

Run: `pytest tests/test_backtest.py::test_compute_metrics_has_required_keys -v`

Expected: `FAILED` with `ImportError` (compute_metrics not defined)

- [ ] **Step 3: Add compute_metrics to backtest.py**

Append to `backtest.py`:

```python
def compute_metrics(bt: pd.DataFrame, risk_free_rate: float = 0.0) -> dict:
    trading_days = 252
    ret = bt["strategy_return"].dropna()

    ann_return = ret.mean() * trading_days
    ann_vol = ret.std() * np.sqrt(trading_days)

    downside = ret[ret < 0]
    ann_downside = (
        downside.std() * np.sqrt(trading_days) if len(downside) > 1 else 1e-9
    )

    cumulative = bt["cumulative"]
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()

    trades = bt[bt["position_size"] > 0]
    win_rate = float((trades["strategy_return"] > 0).mean()) if len(trades) > 0 else 0.0

    sharpe = (ann_return - risk_free_rate) / ann_vol if ann_vol > 1e-9 else 0.0
    sortino = (ann_return - risk_free_rate) / ann_downside
    calmar = ann_return / abs(max_drawdown) if max_drawdown < -1e-9 else 0.0

    return {
        "total_return": float(cumulative.iloc[-1] - 1),
        "ann_return": float(ann_return),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "calmar": float(calmar),
        "max_drawdown": float(max_drawdown),
        "win_rate": float(win_rate),
        "trade_count": int((bt["position_size"] > 0).sum()),
        "bh_return": float(bt["bh_cumulative"].iloc[-1] - 1),
        "beats_bh": bool(
            (cumulative.iloc[-1] - 1) > (bt["bh_cumulative"].iloc[-1] - 1)
        ),
    }
```

- [ ] **Step 4: Run all backtest tests**

Run: `pytest tests/test_backtest.py -v`

Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add backtest.py tests/test_backtest.py
git commit -m "feat: add compute_metrics with Sharpe, Sortino, Calmar, and drawdown"
```

---

## Task 8: main.py — new orchestrator with 3 charts

**Files:**
- Replace: `main.py` — new thin orchestrator

- [ ] **Step 1: Replace main.py entirely**

Overwrite `main.py` with:

```python
import argparse
import os
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd

from backtest import compute_metrics, run_backtest
from features import engineer_features, fetch_prices, fetch_vix
from model import walk_forward_predict


def _print_results(metrics: dict, ticker: str, period: int) -> None:
    print(f"\n{'='*65}")
    print(f"  {ticker}  |  {period} days of data")
    print(f"{'='*65}")
    print(f"  {'Strategy Return':<28} {metrics['total_return']:>10.2%}")
    print(f"  {'Buy & Hold Return':<28} {metrics['bh_return']:>10.2%}")
    print(f"  {'Beats Buy & Hold':<28} {'YES ✓' if metrics['beats_bh'] else 'NO':>10}")
    print(f"  {'-'*43}")
    print(f"  {'Sharpe Ratio':<28} {metrics['sharpe']:>10.2f}")
    print(f"  {'Sortino Ratio':<28} {metrics['sortino']:>10.2f}")
    print(f"  {'Calmar Ratio':<28} {metrics['calmar']:>10.2f}")
    print(f"  {'Max Drawdown':<28} {metrics['max_drawdown']:>10.2%}")
    print(f"  {'Win Rate':<28} {metrics['win_rate']:>10.2%}")
    print(f"  {'Total Trades':<28} {metrics['trade_count']:>10}")
    print(f"{'='*65}\n")


def _save_charts(
    bt: pd.DataFrame,
    signals: pd.DataFrame,
    model,
    feature_names: list,
    ticker: str,
) -> None:
    fig = plt.figure(figsize=(16, 13))
    gs = gridspec.GridSpec(3, 1, height_ratios=[2, 1, 1.5], hspace=0.4)

    # Chart 1: Equity curve
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(bt.index, bt["portfolio_value"], label="XGBoost Strategy", color="green", linewidth=2)
    ax1.plot(bt.index, bt["bh_value"], label="Buy & Hold", color="royalblue", linewidth=2, alpha=0.7)
    final_s = bt["portfolio_value"].iloc[-1]
    final_b = bt["bh_value"].iloc[-1]
    ax1.annotate(f"${final_s:,.0f}", xy=(bt.index[-1], final_s), color="green", fontsize=9)
    ax1.annotate(f"${final_b:,.0f}", xy=(bt.index[-1], final_b), color="royalblue", fontsize=9)
    ax1.set_title(f"{ticker} — XGBoost Strategy vs Buy & Hold", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Chart 2: Walk-forward signals
    ax2 = fig.add_subplot(gs[1])
    ax2.bar(signals.index, signals["signal"], width=1, alpha=0.6, color="darkorange")
    ax2.set_title("Walk-Forward Trade Signals (1 = Long)", fontsize=11)
    ax2.set_ylabel("Signal")
    ax2.set_yticks([0, 1])
    ax2.grid(alpha=0.3)

    # Chart 3: Feature importance (top 20)
    ax3 = fig.add_subplot(gs[2])
    importance = (
        pd.Series(model.feature_importances_, index=feature_names)
        .sort_values()
        .tail(20)
    )
    importance.plot(kind="barh", ax=ax3, color="steelblue")
    ax3.set_title("Top 20 Feature Importances (last walk-forward fold)", fontsize=11)
    ax3.set_xlabel("Importance Score")

    os.makedirs("charts", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"charts/{ticker}_xgb_{timestamp}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Chart saved: {path}")
    plt.show()


def main(
    ticker: str = "SPY",
    period: int = 1825,
    capital: float = 100_000.0,
    threshold: float = 0.55,
) -> dict:
    print(f"\nFetching {period} days of {ticker} data...")
    prices = fetch_prices(ticker, period)
    vix = fetch_vix(period)
    print(f"  Got {len(prices)} rows.")

    print("Engineering 44 features...")
    X, y = engineer_features(prices, vix)
    print(f"  {len(X)} rows ready for training.")

    print("Running walk-forward validation (this may take a few minutes)...")
    signals, model = walk_forward_predict(X, y, threshold=threshold)
    print(f"  {len(signals)} predictions | {signals['signal'].sum()} long signals")

    print("Backtesting with transaction costs...")
    bt = run_backtest(prices, signals, initial_capital=capital)
    metrics = compute_metrics(bt)

    _print_results(metrics, ticker, period)
    _save_charts(bt, signals, model, list(X.columns), ticker)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XGBoost Stock Direction Analyser")
    parser.add_argument("--ticker", type=str, default="SPY", help="Stock ticker (default: SPY)")
    parser.add_argument("--period", type=int, default=1825, help="Days of history (default: 1825 = 5 years)")
    parser.add_argument("--capital", type=float, default=100_000.0, help="Starting capital (default: 100000)")
    parser.add_argument("--threshold", type=float, default=0.55, help="Signal confidence threshold (default: 0.55)")
    args = parser.parse_args()
    main(ticker=args.ticker, period=args.period, capital=args.capital, threshold=args.threshold)
```

- [ ] **Step 2: Run all tests to confirm nothing broke**

Run: `pytest tests/ -v`

Expected: `all tests pass` (no regressions)

- [ ] **Step 3: Commit**

```bash
git add main.py
git commit -m "feat: replace main.py with modular XGBoost orchestrator and 3-panel chart"
```

---

## Task 9: Integration — end-to-end smoke test

**Files:**
- Create: `tests/test_integration.py`

- [ ] **Step 1: Write integration test**

Create `tests/test_integration.py`:

```python
def test_full_pipeline_runs_without_error(synthetic_prices, synthetic_vix, monkeypatch):
    """End-to-end: features → model → backtest → metrics, no real network calls."""
    import yfinance as yf
    import features as feat

    call_count = {"n": 0}

    def mock_download(ticker, **kwargs):
        call_count["n"] += 1
        import numpy as np
        import pandas as pd

        n = 400
        dates = pd.bdate_range("2021-01-01", periods=n)
        close = 400 + np.cumsum(np.random.randn(n) * 2)
        vix_close = np.clip(15 + np.random.randn(n) * 3, 10, 60)
        if ticker == "^VIX":
            return pd.DataFrame({"Open": vix_close, "High": vix_close, "Low": vix_close,
                                  "Close": vix_close, "Volume": 0.0}, index=dates)
        high = close + abs(np.random.randn(n))
        low = close - abs(np.random.randn(n))
        volume = np.random.randint(50_000_000, 150_000_000, n).astype(float)
        return pd.DataFrame({"Open": close, "High": high, "Low": low,
                              "Close": close, "Volume": volume}, index=dates)

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
    assert len(signals) > 0
```

- [ ] **Step 2: Run integration test**

Run: `pytest tests/test_integration.py -v`

Expected: `1 passed`

- [ ] **Step 3: Run full test suite**

Run: `pytest tests/ -v`

Expected: all tests pass with no warnings about missing fixtures or imports.

- [ ] **Step 4: Smoke test the real CLI**

Run: `python main.py --ticker SPY --period 1825`

Expected: prints results table, saves chart to `charts/`, no errors. Takes ~3–5 minutes.

- [ ] **Step 5: Final commit**

```bash
git add tests/test_integration.py
git commit -m "feat: add integration test for full XGBoost pipeline"
```

---

## Self-Review Checklist

- [x] **Spec coverage:** All Phase 1 requirements covered — 4-file split, 44 features, XGBoost classifier, walk-forward, transaction costs, 3 charts, updated CLI
- [x] **Placeholder scan:** No TBDs or TODOs — all steps include actual code
- [x] **Type consistency:** `engineer_features` returns `(pd.DataFrame, pd.Series)` used consistently in Tasks 3–9; `walk_forward_predict` returns `(pd.DataFrame, XGBClassifier)` used consistently in Tasks 5–9; `run_backtest` returns `pd.DataFrame` with `cumulative`, `portfolio_value`, `bh_value`, `strategy_return`, `position_size` columns used in `compute_metrics`
- [x] **No phantom references:** All functions referenced in later tasks are defined in earlier tasks
