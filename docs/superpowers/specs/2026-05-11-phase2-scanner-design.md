# Phase 2 — Multi-Ticker Scanner with Regime Filter

**Goal:** A daily scanner that trains/loads XGBoost models for up to 20 stocks, applies a 200-day SMA regime filter, ranks signals by confidence, and outputs a terminal table + dated CSV.

**Architecture:** New `scan.py` CLI script reuses all existing modules. `model.py` gains three helpers for model persistence. `features.py` gains one inference helper. `backtest.py` and `main.py` stay untouched.

**Tech Stack:** Python, XGBoost, joblib (model serialisation), pandas, yfinance, existing project modules.

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `scan.py` | Scanner CLI — orchestrates ticker loop, caching, filter, output |
| Modify | `model.py` | Add `save_model`, `load_model`, `model_is_stale` helpers |
| Modify | `features.py` | Add `engineer_features_inference` for live prediction |
| Create | `tests/test_scan.py` | Unit + integration tests for scanner |
| Modify | `tests/test_model.py` | Add tests for save/load round-trip and staleness |
| Modify | `tests/test_features.py` | Add test for `engineer_features_inference` |

---

## Model Caching

Saved under `models/`:
```
models/
  AAPL_xgb.pkl        ← trained XGBClassifier (joblib)
  AAPL_meta.json      ← {"trained_at": "2026-05-04T09:00:00"}
  NVDA_xgb.pkl
  NVDA_meta.json
  ...
```

**Decision logic per ticker:**
1. No `.pkl` → train via `walk_forward_predict`, save, continue
2. `.pkl` exists, `trained_at` > 7 days ago → retrain, overwrite, continue
3. `.pkl` exists, fresh (≤ 7 days) → load directly
4. `--retrain` flag passed → always retrain all tickers

The saved model is the last fold's model from `walk_forward_predict` — trained on all available data.

**New functions in `model.py`:**
- `save_model(model: XGBClassifier, ticker: str) -> None` — serialises with joblib, writes meta JSON
- `load_model(ticker: str) -> tuple[XGBClassifier, datetime]` — returns model + trained_at timestamp
- `model_is_stale(ticker: str, max_age_days: int = 7) -> bool` — True if missing or older than threshold

---

## Regime Filter

Applied in `scan.py` after prediction, before emitting a signal.

**Rule:** `signal = 0` if `close_today <= SMA_200`

- SMA_200 computed from the price data already fetched (`prices["Close"].rolling(200).mean().iloc[-1]`)
- Keeps strategy long-only and only active in uptrends
- No feature engineering changes needed — regime is a post-prediction gate in `scan.py`

---

## `engineer_features_inference` (new function in `features.py`)

`engineer_features` drops the last 5 rows because the forward target is NaN there. For live prediction we need TODAY's features, so we add:

```python
def engineer_features_inference(prices: pd.DataFrame, vix: pd.DataFrame) -> pd.DataFrame:
    """Return the single most-recent feature row for live prediction (no target needed)."""
    # Same feature computation as engineer_features
    # Drop NaN only on _FEATURE_COLS, not on target
    # Return data[_FEATURE_COLS].dropna().iloc[[-1]]
```

This gives the scanner a valid feature vector for today without requiring a future return.

---

## Scanner Workflow

```
for ticker in tickers:
    prices  = fetch_prices(ticker, period=1825)
    vix     = fetch_vix(period=1825)

    if model_is_stale(ticker) or --retrain:
        X, y    = engineer_features(prices, vix)       # for training (drops last 5 rows)
        signals, model = walk_forward_predict(X, y, threshold=threshold)
        save_model(model, ticker)
    else:
        model = load_model(ticker)

    X_today = engineer_features_inference(prices, vix)  # today's features for prediction
    prob    = model.predict_proba(X_today)[:, 1][0]
    signal  = 1 if prob >= threshold else 0

    sma_200 = prices["Close"].rolling(200).mean().iloc[-1]
    regime  = prices["Close"].iloc[-1] > sma_200
    if not regime:
        signal = 0

    results.append({ticker, prob, signal, regime, sma_200, close})

sort results by prob descending
print table
save CSV to signals/YYYY-MM-DD.csv
```

---

## CLI Interface

```
python scan.py                              # scan default 20 tickers
python scan.py --tickers AAPL NVDA TSLA    # custom list
python scan.py --retrain                    # force retrain all
python scan.py --threshold 0.55            # tighter cutoff
python scan.py --period 1825               # days of history (default)
```

**Default ticker list (20):**
SPY, QQQ, AAPL, MSFT, NVDA, TSLA, AMZN, GOOGL, META, AMD, NFLX, JPM, V, UNH, XOM, BABA, COIN, PLTR, SMCI, MSTR

---

## Output

**Terminal:**
```
=================================================================
  SCANNER — 2026-05-11  |  threshold: 0.52  |  20 tickers
=================================================================
  Ticker   Prob    Signal   Regime   Model Age
  -------  ------  -------  -------  ---------
  NVDA     0.74    LONG     YES      2d
  TSLA     0.69    LONG     YES      2d
  AMD      0.63    LONG     YES      2d
  AAPL     0.48    -        YES      2d
  META     0.38    -        NO       2d
=================================================================
  3 long signals  |  Saved: signals/2026-05-11.csv
=================================================================
```

**CSV — `signals/YYYY-MM-DD.csv`:**
```
date,ticker,prob,signal,regime_ok,close,sma_200
2026-05-11,NVDA,0.74,1,True,891.23,712.45
2026-05-11,TSLA,0.69,1,True,174.56,201.12
```

Sorted by `prob` descending. One file per day.

---

## Tests

**`tests/test_model.py` additions:**
- `test_save_load_roundtrip` — save model, load it, verify predictions match
- `test_model_is_stale_missing` — returns True when no file
- `test_model_is_stale_fresh` — returns False when < 7 days old
- `test_model_is_stale_old` — returns True when > 7 days old

**`tests/test_features.py` addition:**
- `test_engineer_features_inference_returns_one_row` — returns exactly 1 row with all 44 feature columns

**`tests/test_scan.py`:**
- `test_regime_filter_blocks_signal` — signal suppressed when close < SMA_200
- `test_regime_filter_passes_signal` — signal emits when close > SMA_200
- `test_scanner_output_shape` — returns DataFrame with expected columns (ticker, prob, signal, regime_ok, close, sma_200)
- `test_scanner_csv_saved` — CSV file written to `signals/YYYY-MM-DD.csv` with correct columns
- `test_scanner_uses_cached_model` — monkeypatch confirms retrain skipped when model fresh
- `test_scanner_retrains_stale_model` — retrain triggered when model > 7 days old

---

## Error Handling

- Ticker fetch fails (delisted, bad symbol): log warning, skip ticker, continue loop
- Insufficient data for walk_forward_predict (< 504 rows): log warning, skip ticker
- Models directory created automatically if missing
- Signals directory created automatically if missing
