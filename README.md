# S&P 500 Trading Strategy Analyzer

A professional quantitative trading system that uses Long Short-Term Memory (LSTM) neural networks to predict stock prices and optimize trading strategies through comprehensive backtesting.

## Overview

This system analyzes the S&P 500 (SPY) using machine learning to predict future price movements and automatically evaluates multiple position sizing strategies to find the optimal risk-adjusted approach. It combines active trading signals with passive buy-and-hold allocation to maximize returns while controlling risk.

## How It Works

### 1. Data Collection

The system fetches historical price data from Yahoo Finance for the specified time period (default: 2 years).

**What it retrieves:**
- Open, High, Low, Close prices
- Trading volume
- Daily price movements

### 2. Feature Engineering

Raw price data is transformed into 31 technical indicators that capture different market dynamics:

**Price-Based Features:**
- **Lag features (lag_1 to lag_5)**: Previous 5 days of closing prices
- **Moving averages**: 7-day, 14-day, and 21-day simple moving averages (SMA)
- **Exponential moving averages**: 12-day and 26-day EMAs (more weight on recent prices)

**Momentum Indicators:**
- **MACD (Moving Average Convergence Divergence)**: Shows trend direction and strength
- **RSI (Relative Strength Index)**: Measures overbought/oversold conditions (0-100 scale)
- **Price momentum**: 5-day and 10-day price changes

**Volatility Measures:**
- **Bollinger Bands**: Price channels showing standard deviation ranges
- **ATR (Average True Range)**: Measures market volatility
- **Rolling volatility**: 20-day standard deviation of returns

**Volume Analysis:**
- **Volume changes**: Day-over-day volume shifts
- **Volume ratio**: Current volume vs 20-day average

### 3. LSTM Model Training

**What is LSTM?**
Long Short-Term Memory networks are a type of neural network designed to remember patterns over time. Unlike traditional models that look at data points independently, LSTMs understand sequences and temporal dependencies.

**Architecture:**
```
Input: 60 days of historical data × 31 features
  ↓
Bidirectional LSTM Layer (128 units) → Processes sequences forward and backward
  ↓
Dropout (20%) → Prevents overfitting
  ↓
Bidirectional LSTM Layer (64 units)
  ↓
Dropout (20%)
  ↓
LSTM Layer (32 units)
  ↓
Dropout (20%)
  ↓
Dense Layer (16 units) → Pattern recognition
  ↓
Output Layer → Predicted next-day price
```

**Training Process:**
- **Data split**: 70% training, 30% testing
- **Lookback window**: 60 days (the model sees 60 days to predict day 61)
- **Scaling**: All features normalized to 0-1 range for better learning
- **Early stopping**: Training stops if performance plateaus (prevents overfitting)
- **Learning rate reduction**: Automatically adjusts when improvement slows

**Why LSTM over traditional models?**
- Captures time-series dependencies
- Remembers long-term patterns
- Handles non-linear relationships
- Better for sequential data like stock prices

### 4. Position Sizing Strategies

The system tests 5 different approaches to determine how much capital to allocate:

#### Pure Strategies (100% Active Trading)

**a) Binary (All-In or All-Out)**
- If predicted return > 0.1%: Invest 100%
- Otherwise: Stay in cash (0%)
- Simplest approach, highest risk/reward

**b) Fixed (Conservative)**
- If signal is positive: Invest 50%
- Otherwise: Cash
- Reduces exposure compared to binary

**c) Proportional (Confidence-Based)**
- Position size = Predicted return strength
- Higher confidence = Larger position
- Example: 2% predicted gain = 2% allocation (capped at 100%)

**d) Kelly Criterion (Mathematically Optimal)**
- Uses half-Kelly formula for safety
- Position = (Predicted Edge × 2), capped at 50%
- Based on expected value and probabilities
- Optimal for long-term growth

**e) Volatility-Adjusted (Risk-Aware)**
- Reduces position size during high volatility
- Increases position when market is calm
- Position = (Median volatility / Current volatility) × Signal strength

#### Hybrid Strategies (Blended Approach)

Combines active trading with passive buy-and-hold:

**30/70 Hybrid** (Recommended)
- 30% allocated to active LSTM strategy
- 70% always invested (buy-and-hold)
- Captures most market upside while adding active alpha

**50/50 Hybrid**
- Equal split between active and passive
- Balanced risk/reward

**70/30 Hybrid**
- More aggressive active trading
- Lower passive allocation

**Why Hybrids Work:**
- Reduces drawdowns (maximum losses)
- Captures major market moves
- Improves Sharpe ratio (risk-adjusted returns)
- More practical for real-world implementation

### 5. Backtesting & Performance Metrics

Each strategy is tested on historical data to evaluate performance:

**Return Metrics:**
- **Total Return**: Overall percentage gain/loss
- **Buy & Hold Comparison**: How strategy performs vs. simply holding

**Risk Metrics:**
- **Sharpe Ratio**: Return per unit of risk (higher is better)
  - < 1.0: Poor risk-adjusted returns
  - 1.0-2.0: Good
  - \> 2.0: Excellent
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades

### 6. Strategy Selection

The system automatically:
1. Ranks all strategies by Sharpe ratio
2. Identifies which strategies beat buy-and-hold
3. Selects the optimal approach
4. Generates visualization charts
5. Saves results with timestamps

## Usage

### Basic Usage (S&P 500 Analysis)
```bash
python main.py
```

This runs the complete analysis on SPY with default settings (2 years of data).

### Custom Analysis
```bash
python main.py --ticker AAPL --period 365
```

**Parameters:**
- `--ticker`: Stock symbol (default: SPY)
- `--period`: Days of historical data (default: 730)

## Output

### Console Output
```
======================================================================
S&P 500 PREDICTION & STRATEGY ANALYSIS: SPY
Period: 730 days (2.0 years)
======================================================================

Training LSTM model (lookback=60 steps)...
Training sequences: (276, 60, 31)
Test sequences: (84, 60, 31)
LSTM Training Complete!
   Final RMSE: 8.45, MAE: 6.32
   Trained for 26 epochs

Testing hybrid approaches with PROPORTIONAL strategy...

================================================================================
STRATEGY COMPARISON SUMMARY
================================================================================
Method                         Type                Return   Sharpe     Max DD
--------------------------------------------------------------------------------
binary                         Pure                15.2%     1.45    -12.3%
proportional                   Pure                 8.4%     1.82     -6.1%
proportional_hybrid_70_30      30/70 split         22.5%     2.35     -8.2%  *BEST*
--------------------------------------------------------------------------------
Buy & Hold (Benchmark)         100% Long           25.3%
================================================================================

BEST STRATEGY: PROPORTIONAL_HYBRID_70_30
   Type: 30/70 split
   Total Return: 22.5%
   Sharpe Ratio: 2.35
```

### Generated Files

**Charts** (saved in `charts/` directory):
- Filename format: `{TICKER}_{STRATEGY}_{TIMESTAMP}.png`
- Example: `SPY_proportional_hybrid_70_30_20251017_143522.png`

**Chart Contents:**
1. **Top Panel**: Equity curves comparing strategy vs. buy-and-hold
2. **Bottom Panel**: Position sizing over time (how much capital deployed each day)

## Key Concepts Explained

### Look-Ahead Bias Prevention

The system never uses future information to make past predictions:
- **Training**: Uses only data available up to each prediction point
- **Feature calculation**: Rolling windows prevent data leakage
- **Walk-forward approach**: Each prediction uses only historical data

### Short Selling

When `allow_short=True` (default):
- Negative predictions trigger short positions
- Position size becomes negative (betting on price decline)
- Example: -0.5 position = 50% short (profit when price drops)

### Rebalancing

In hybrid strategies:
- Capital is split at the beginning
- Active portion trades based on signals
- Passive portion stays fully invested
- No rebalancing between allocations (set-and-forget)

## Performance Interpretation

### When Strategies Outperform

**Best scenarios:**
- Sideways/choppy markets (lots of up/down movement)
- High volatility periods
- Market corrections (LSTM avoids drawdowns)

### When Buy-and-Hold Wins

**Challenging scenarios:**
- Strong bull markets (missing upside when in cash)
- Low volatility trends (fewer trading opportunities)
- Consistently rising markets

### Hybrid Advantage

30/70 hybrids typically:
- Capture 70-80% of bull market gains
- Reduce drawdowns by 20-40%
- Improve Sharpe ratios significantly
- More realistic for actual trading

## Technical Requirements

**Python Packages:**
```
yfinance          # Market data
pandas, numpy     # Data manipulation
tensorflow/keras  # LSTM neural networks
scikit-learn      # Data preprocessing
matplotlib        # Visualization
```

**Installation:**
```bash
pip install yfinance pandas numpy tensorflow scikit-learn matplotlib
```

## Limitations & Considerations

1. **Transaction Costs**: Not included in backtest (would reduce returns)
2. **Slippage**: Assumes perfect execution at closing prices
3. **Market Impact**: Assumes trades don't affect prices
4. **Survivorship Bias**: SPY has survived; doesn't include delisted stocks
5. **Regime Changes**: Past patterns may not repeat
6. **Overfitting Risk**: LSTM might learn noise instead of signal

## File Structure

```
Analyser/
├── main.py                 # Core analysis system
├── charts/                 # Generated visualizations
│   └── *.png              # Timestamped chart files
├── README.md              # This file
└── requirements.txt       # Python dependencies
```

## System Workflow

```
1. Fetch Data (Yahoo Finance)
         ↓
2. Engineer Features (31 indicators)
         ↓
3. Train LSTM (60-day sequences)
         ↓
4. Generate Predictions
         ↓
5. Test Position Sizing (5 methods)
         ↓
6. Test Hybrid Approaches (3 allocations)
         ↓
7. Select Best Strategy (Sharpe ratio)
         ↓
8. Generate Report & Charts
```

## Understanding Results

**Scenario 1: Strategy Beats B&H**
```
Strategy: 28% return, Sharpe: 2.1, Max DD: -8%
Buy & Hold: 25% return
```
Interpretation: Strategy delivers higher returns with better risk management.

**Scenario 2: Strategy Underperforms**
```
Strategy: 12% return, Sharpe: 1.8, Max DD: -5%
Buy & Hold: 30% return
```
Interpretation: Lower returns, but much lower risk (Sharpe is good). Suitable for risk-averse investors or during uncertain markets.

**Scenario 3: Hybrid Optimal**
```
Pure Strategy: 5% return
30/70 Hybrid: 22% return
Buy & Hold: 28% return
```
Interpretation: Hybrid captures most upside while reducing risk. Practical real-world approach.

## Future Enhancements

Potential improvements:
- Add transaction cost modeling
- Implement stop-loss orders
- Include multiple timeframes
- Add market regime detection
- Ensemble multiple models
- Real-time trading integration

---

**Note**: This is a research and analysis tool. Past performance does not guarantee future results. Always conduct your own due diligence before making investment decisions.



