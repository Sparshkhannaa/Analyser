# Stock Direction Analyser

An algorithmic trading system that uses XGBoost to predict stock direction (up or down over the next 5 days) and scan multiple tickers daily for high-confidence signals.

## What It Does

Two modes:

- **`main.py`** — Deep single-ticker analysis. Trains a walk-forward model, backtests the strategy, and generates charts comparing the strategy against buy & hold.
- **`scan.py`** — Daily multi-ticker scanner. Loads cached models for up to 20 stocks, applies a 200-day SMA regime filter, and outputs a ranked signal table + CSV.

## Quick Start

```bash
pip install -r requirements.txt

# Deep analysis on a single stock
python main.py --ticker NVDA

# Scan 20 stocks for today's signals
python scan.py

# Scan specific tickers
python scan.py --tickers NVDA TSLA AMD AAPL MSFT

# Force retrain all cached models
python scan.py --retrain
```

## How It Works

### 1. Feature Engineering (44 features)

Raw OHLCV data is transformed into 44 technical indicators:

| Category | Features |
|----------|----------|
| Lag | Previous 5 closing prices |
| Moving Averages | SMA 7/14/21, EMA 12/26 |
| Trend | MACD, MACD signal, MACD diff |
| Momentum | RSI, Stochastic %K/%D, Williams %R, ROC 5/10 |
| Volatility | Bollinger Bands (width, position), ATR, ATR 14, rolling vol |
| Volume | Volume change, volume SMA 20, volume ratio, OBV, OBV slope |
| Price | % change 1/5/10 days, momentum 5/10, high-low ratio |
| Market | VIX, VIX 5-day change, ADX |
| Calendar | Day of week |
| Distance | Distance from 52-week high/low |

**Target:** Binary — will the stock close higher 5 days from now? (1 = yes, 0 = no)

### 2. Walk-Forward Validation

The model is trained using an expanding window to prevent look-ahead bias:

```
Training window 1: Days 1–504  → Predict days 505–567
Training window 2: Days 1–567  → Predict days 568–630
Training window 3: Days 1–630  → Predict days 631–693
...and so on
```

Each prediction is made using only data that would have been available at that time. No future data leaks into training.

### 3. XGBoost Classifier

Key settings:
- `n_estimators`: 500 (with early stopping at 30 rounds)
- `max_depth`: 4
- `learning_rate`: 0.05
- `scale_pos_weight`: auto-computed to correct class imbalance
- `eval_metric`: AUC

### 4. Position Sizing

Position size scales with model confidence:

```
position_size = (probability - 0.5) × 4, capped at 1.0
```

A 0.52 probability → 8% position. A 0.75 probability → 100% position.

### 5. Backtesting

- Signals hold for 5 days (non-overlapping)
- Executes at next-day open price
- 0.1% transaction cost per trade
- Benchmarked against buy & hold

### 6. Regime Filter (scanner only)

Before emitting a signal, the scanner checks: `close > 200-day SMA`. If the stock is in a downtrend, the signal is suppressed regardless of model confidence. This keeps the strategy long-only and avoids catching falling knives.

### 7. Model Caching (scanner only)

Models are saved to `models/` after training. On subsequent runs, cached models load in seconds instead of retraining (~2 min per ticker). Models auto-retrain after 7 days, or on demand with `--retrain`.

## CLI Reference

### `main.py`

```bash
python main.py --ticker SPY --period 1825 --capital 100000 --threshold 0.52
```

| Flag | Default | Description |
|------|---------|-------------|
| `--ticker` | `SPY` | Stock ticker |
| `--period` | `1825` | Days of history (5 years) |
| `--capital` | `100000` | Starting capital |
| `--threshold` | `0.52` | Signal confidence cutoff |

### `scan.py`

```bash
python scan.py --tickers AAPL NVDA TSLA --threshold 0.55 --retrain
```

| Flag | Default | Description |
|------|---------|-------------|
| `--tickers` | 20 large-caps | Space-separated list of tickers |
| `--period` | `1825` | Days of history |
| `--threshold` | `0.52` | Signal confidence cutoff |
| `--retrain` | off | Force retrain all models |

**Default ticker list (20):**
SPY, QQQ, AAPL, MSFT, NVDA, TSLA, AMZN, GOOGL, META, AMD, NFLX, JPM, V, UNH, XOM, BABA, COIN, PLTR, SMCI, MSTR

## Output

### `main.py` — Terminal

```
=================================================================
  NVDA  |  1825 days of data
=================================================================
  Strategy Return              +24.16%
  Buy & Hold Return          +1441.23%
  Beats Buy & Hold                  NO
  -------------------------------------------
  Sharpe Ratio                      0.51
  Sortino Ratio                     0.74
  Calmar Ratio                      0.31
  Max Drawdown                     -7.40%
  Win Rate                         57.14%
  Total Trades                        105
=================================================================
```

### `main.py` — Chart (saved to `charts/`)

3-panel PNG:
1. Equity curve: strategy vs buy & hold
2. Walk-forward trade signals over time
3. Top 20 feature importances

### `scan.py` — Terminal

```
=================================================================
  SCANNER — 2026-05-11  |  threshold: 0.52  |  20 tickers
=================================================================
  Ticker     Prob  Signal   Regime
  ------- ------  ------- -------
  NVDA       0.74  LONG     YES
  TSLA       0.69  LONG     YES
  AMD        0.63  LONG     YES
  AAPL       0.48  -        YES
  META       0.38  -        NO
=================================================================
  3 long signals
=================================================================
```

### `scan.py` — CSV (saved to `signals/YYYY-MM-DD.csv`)

```
date,ticker,prob,signal,regime_ok,close,sma_200
2026-05-11,NVDA,0.74,1,True,891.23,712.45
2026-05-11,TSLA,0.69,1,True,174.56,201.12
```

One file per day. Build up a backlog over time.

## Metrics Explained

| Metric | What it means |
|--------|--------------|
| **Sharpe Ratio** | Return per unit of risk. Above 1.0 is good, above 2.0 is excellent |
| **Sortino Ratio** | Like Sharpe but only penalises downside volatility |
| **Calmar Ratio** | Annualised return divided by max drawdown |
| **Max Drawdown** | Largest peak-to-trough loss during the period |
| **Win Rate** | % of trade days with positive returns |

## File Structure

```
Analyser/
├── main.py              # Single-ticker deep analysis
├── scan.py              # Multi-ticker daily scanner
├── features.py          # Feature engineering (44 features + inference)
├── model.py             # XGBoost training, walk-forward, save/load
├── backtest.py          # Simulation engine and metrics
├── requirements.txt
├── charts/              # PNG charts from main.py runs
├── models/              # Cached XGBoost models (auto-generated)
├── signals/             # Daily CSV outputs from scan.py (auto-generated)
└── tests/
    ├── conftest.py
    ├── test_features.py
    ├── test_model.py
    ├── test_backtest.py
    ├── test_integration.py
    └── test_scan.py
```

## Running Tests

```bash
pytest -v
```

37 tests covering feature engineering, model training, backtesting, scanner logic, model caching, and regime filtering.

## Requirements

- Python 3.11+
- At least 2 years of price history per ticker (stocks launched less than ~2 years ago won't have enough data)

```bash
pip install -r requirements.txt
```

## Limitations

- **Long-only**: No short selling
- **US equities only**: Requires Yahoo Finance data
- **New stocks unsupported**: Needs ~2 years of history minimum
- **No intraday**: Daily bars only
- **Transaction costs**: Fixed at 0.1% per trade (real costs may vary)
- **Past performance**: Backtested results do not guarantee future returns

## Roadmap

- **Phase 3 (next):** Paper trading via Alpaca API with daily automated runs and Telegram alerts
- **Phase 4:** Live trading on a real account with client reporting dashboard

---

*Research tool only. Not financial advice. Always do your own due diligence.*
