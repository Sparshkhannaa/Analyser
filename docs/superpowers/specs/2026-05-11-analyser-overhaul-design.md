# Analyser Overhaul — Design Spec
**Date:** 2026-05-11  
**Goal:** Transform the current broken LSTM price-predictor into a production-grade XGBoost direction classifier with realistic backtesting — the foundation of a live trading algo and finance pivot portfolio.

---

## Problem Statement

The current system predicts the exact next-day price (regression). It doesn't work:
- RMSE of 45.8 on a ~$500 stock = ~9% average error
- 0% win rate on all pure strategies
- All pure strategies lose money; buy-and-hold beats everything
- Backtest is unrealistic: no transaction costs, single train/test split (looks at the future)

The fix is switching to direction prediction (up or down) — simpler, more actionable, and works far better with the amount of data available.

---

## Architecture

Four focused modules replacing the single 600-line `main.py`:

```
Analyser/
├── main.py          # CLI entry point — thin orchestrator
├── features.py      # All feature engineering (existing + new)
├── model.py         # XGBoost classifier + walk-forward validation
├── backtest.py      # Realistic backtesting engine
└── requirements.txt # Updated dependencies
```

**Data flow:**
```
Yahoo Finance (5+ years) + VIX data
        ↓
  features.py  →  44 indicators
        ↓
  model.py     →  XGBoost: predicts UP (1) or DOWN (0) for tomorrow
                  walk-forward: every prediction uses only past data
        ↓
  backtest.py  →  simulates trades with costs, position sizing, risk metrics
        ↓
  main.py      →  prints results table, saves equity curve + feature importance charts
```

---

## Target Variable

```
y = 1  if close[t+1] > close[t] * 1.001   (up more than 0.1%)
y = 0  otherwise
```

The 0.1% threshold filters noise trades that wouldn't cover transaction costs.

---

## Features (44 total)

### Existing 31 (kept from current system)
- Lag features: lag_1 to lag_5
- Moving averages: SMA 7, 14, 21; EMA 12, 26
- MACD, MACD signal, MACD diff
- RSI (14)
- Bollinger Bands: middle, upper, lower, width, position
- Volatility (20-day), ATR, ATR_14
- Price changes: pct_change 1, 5, 10
- Volume: volume_change_1, volume_sma_20, volume_ratio
- Momentum: momentum_5, momentum_10
- High/low ratio

### New 13 (added in Phase 1)
- **VIX close** — market fear index; single most important regime signal
- **VIX 5-day change** — is fear rising or falling?
- **ADX (14)** — how strong is the current trend (0–100)?
- **Stochastic %K, %D** — overbought/oversold oscillator
- **Williams %R** — momentum oscillator, confirms reversals
- **OBV (On-Balance Volume)** — is volume supporting the price move?
- **OBV slope (5-day)** — is OBV trending up or down?
- **ROC (5), ROC (10)** — rate of change / momentum confirmation
- **Day of week** (0–4) — Monday/Friday effects are real
- **Distance from 52-week high** — how far below the peak are we?
- **Distance from 52-week low** — how close to the floor are we?

---

## Model: XGBoost Classifier

```python
XGBClassifier(
    n_estimators=500,
    max_depth=4,               # shallow trees reduce overfitting
    learning_rate=0.05,        # slow learning = more robust
    subsample=0.8,             # use 80% of rows per tree
    colsample_bytree=0.8,      # use 80% of features per tree
    eval_metric='auc',
    early_stopping_rounds=30
)
```

**Signal threshold:** Only trade when predicted probability ≥ 0.55.  
Fewer trades, better quality. Avoids borderline calls.

**Why XGBoost over LSTM here:**  
XGBoost is designed for tabular data with ~1000 rows. LSTM needs sequence data with tens of thousands of rows to perform reliably. For daily stock data with 5 years of history (~1250 rows), XGBoost consistently wins in academic benchmarks and industry practice.

---

## Walk-Forward Validation

Prevents the "studying the answer key" problem in the current backtest.

```
Window 1:  Train on years 1–2  →  predict year 3 Q1
Window 2:  Train on years 1–3 Q1  →  predict year 3 Q2
Window 3:  Train on years 1–3 Q2  →  predict year 3 Q3
... expanding window, rolling forward every quarter
```

Every single prediction is made on data the model has never seen.  
Minimum training window: 2 years. Step size: 63 trading days (1 quarter).

---

## Backtesting Engine

### Transaction costs
- Commission: $0 (Alpaca free tier)
- Bid-ask spread: 0.05% per trade
- Round-trip cost: ~0.1% (deducted on every open + close)

### Execution assumption
- Trades execute at the **next day's open price** (not close — more realistic)

### Position sizing
- Kelly fraction based on predicted probability
- Capped at 50% of portfolio (half-Kelly for safety)
- Cash position earns 0% (conservative)
- Short selling: disabled in Phase 1 (added in Phase 2)

### Metrics reported
| Metric | What it means |
|---|---|
| Total return | Overall % gain/loss |
| Sharpe ratio | Return per unit of risk |
| Sortino ratio | Return per unit of *downside* risk |
| Calmar ratio | Return ÷ worst drawdown (what clients care about) |
| Max drawdown | Worst peak-to-trough loss |
| Win rate | % of trades that were profitable |
| Trade count | How many trades were made |
| Buy & Hold comparison | Did we beat doing nothing? |

### Charts saved
1. **Equity curve** — strategy vs buy-and-hold over time
2. **Feature importance** — which indicators drove the model (great for presentations)
3. **Walk-forward performance** — return per quarter (shows consistency)

---

## CLI Interface

```bash
# Analyse SPY with defaults (5 years of data)
python main.py

# Custom ticker and period
python main.py --ticker AAPL --period 1825

# Parameters:
#   --ticker    Stock symbol (default: SPY)
#   --period    Days of history to fetch (default: 1825 = 5 years)
#   --capital   Starting capital (default: 100000)
#   --threshold Signal confidence threshold (default: 0.55)
```

---

## Phased Roadmap

### Phase 1 — Fix the Core (current)
- Modular code structure (4 files)
- XGBoost classifier replacing LSTM
- 45 features including VIX
- Walk-forward validation
- Realistic backtesting with costs
- Feature importance chart
- **Deliverable:** A model that produces honest, trustworthy backtest results

### Phase 2 — Make it Robust
- Model save/load (don't retrain every run)
- Multi-ticker support (scan a watchlist)
- More features (options implied volatility, sector ETF data)
- Hyperparameter tuning (Bayesian optimisation)
- Signal confidence calibration
- **Deliverable:** A reliable scanner across multiple assets

### Phase 3 — Paper Trading
- Alpaca API integration
- Daily automated signal generation (runs on a schedule)
- Telegram or email alerts when a signal fires
- Trade log with entry/exit prices
- **Deliverable:** Runs live, no money at risk, real-world validation

### Phase 4 — Live Trading
- $1000 live Alpaca account
- Automated order execution
- Portfolio risk controls (max daily loss, position limits)
- Client-facing performance dashboard
- **Deliverable:** Real returns, client-ready system

---

## What Success Looks Like (Phase 1)

After Phase 1, running `python main.py` should show:
- Walk-forward Sharpe ratio > 1.0 consistently
- At least some strategies beating buy-and-hold after costs
- Feature importance chart showing VIX and momentum indicators as top drivers
- Honest results (no look-ahead bias)

If it still can't beat buy-and-hold after Phase 1, the answer is more/better data and features (Phase 2) — not adding live trading on top of a broken model.

---

## Dependencies Added

```
xgboost>=1.7.0
pandas-ta>=0.3.14b    # technical indicators library (replaces manual RSI/MACD code)
```

---

## What Is NOT in Scope

- Real-time intraday data (Phase 3+)
- Options trading (out of scope entirely for now)
- Cryptocurrency (different market microstructure, Phase 3+)
- Regulatory compliance / fund registration (out of scope)
- Sentiment analysis / news data (Phase 2+)
