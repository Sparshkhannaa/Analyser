import os
import argparse
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import yfinance as yf

def fetch_prices_yf(ticker: str, period_days: int = 365, interval: str = "1d") -> pd.DataFrame:

    end = datetime.utcnow().date()
    start = end - timedelta(days=period_days)
    df = yf.download(ticker, start=start.isoformat(), end=end.isoformat(), interval=interval, progress=False)
    print(df.head())
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    print(df.head())
    if df.empty:
        raise ValueError(f"No data for {ticker}.")
    df.index = pd.to_datetime(df.index)
    print(df.head())
    return df

def engineer_features(df: pd.DataFrame, n_lags: int = 5):
    data = df.copy()
    data['close'] = data['Close']
    data['high'] = data['High']
    data['low'] = data['Low']
    data['volume'] = data['Volume']

    # Price lags
    for lag in range(1, n_lags + 1):
        data[f'lag_{lag}'] = data['close'].shift(lag)

    # Moving averages
    data['sma_7'] = data['close'].rolling(window=7).mean()
    data['sma_14'] = data['close'].rolling(window=14).mean()
    data['sma_21'] = data['close'].rolling(window=21).mean()
    
    # Exponential moving average
    data['ema_12'] = data['close'].ewm(span=12, adjust=False).mean()
    data['ema_26'] = data['close'].ewm(span=26, adjust=False).mean()
    
    # MACD
    data['macd'] = data['ema_12'] - data['ema_26']
    data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()
    data['macd_diff'] = data['macd'] - data['macd_signal']
    
    # RSI (Relative Strength Index)
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['rsi'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    data['bb_middle'] = data['close'].rolling(window=20).mean()
    bb_std = data['close'].rolling(window=20).std()
    data['bb_upper'] = data['bb_middle'] + (bb_std * 2)
    data['bb_lower'] = data['bb_middle'] - (bb_std * 2)
    data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
    data['bb_position'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
    
    # Volatility
    data['volatility'] = data['close'].pct_change().rolling(window=20).std()
    data['atr'] = data['high'] - data['low']  # Simplified ATR
    data['atr_14'] = data['atr'].rolling(window=14).mean()
    
    # Price changes
    data['pct_change_1'] = data['close'].pct_change(1)
    data['pct_change_5'] = data['close'].pct_change(5)
    data['pct_change_10'] = data['close'].pct_change(10)
    
    # Volume indicators
    data['volume_change_1'] = data['volume'].pct_change(1)
    data['volume_sma_20'] = data['volume'].rolling(window=20).mean()
    data['volume_ratio'] = data['volume'] / data['volume_sma_20']
    
    # Momentum
    data['momentum_5'] = data['close'] - data['close'].shift(5)
    data['momentum_10'] = data['close'] - data['close'].shift(10)
    
    # Price position relative to high/low
    data['high_low_ratio'] = (data['close'] - data['low']) / (data['high'] - data['low'])

    data['target'] = data['close'].shift(-1)
    print(data.head())
    data = data.dropna()
    for c in data.columns:
        print(c, type(c))

    # Select all engineered features
    feature_cols = [c for c in data.columns
                    if isinstance(c, str) and 
                    (c.startswith('lag_') or c.startswith('sma_') or c.startswith('ema_') or
                     c.startswith('macd') or c == 'rsi' or c.startswith('bb_') or
                     c.startswith('volatility') or c.startswith('atr') or 
                     c.startswith('pct_change') or c.startswith('volume_') or
                     c.startswith('momentum') or c == 'high_low_ratio')]
    
    features = data[feature_cols]
    target = data['target']
    
    print(f"\nCreated {len(feature_cols)} features for training")

    return features, target, data
def backtest_strategy(df_with_features: pd.DataFrame, preds: np.ndarray, sentiment_df: pd.DataFrame = None, 
                      position_sizing: str = 'kelly', max_position: float = 1.0, risk_free_rate: float = 0.0, 
                      allow_short: bool = True, hybrid_weight: float = 0.0, initial_capital: float = 100000.0):
    """
    Enhanced backtesting with position sizing and risk management.
    
    Args:
        df_with_features: DataFrame with features and actual prices
        preds: Array of predictions
        sentiment_df: Optional sentiment data (unused for now)
        position_sizing: 'fixed', 'proportional', 'kelly', or 'volatility_adjusted'
        max_position: Maximum position size (0-1, where 1 = 100% of capital)
        risk_free_rate: Annual risk-free rate for Sharpe calculation
        allow_short: If True, allow short positions when predictions are negative
        hybrid_weight: Weight for buy-and-hold (0=pure strategy, 1=pure B&H, 0.5=50/50)
    """
    df = df_with_features.copy()
    test_idx = df.index[-len(preds):]
    df_test = df.loc[test_idx].copy()
    df_test['pred_next_close'] = preds
    df_test['pred_return'] = (df_test['pred_next_close'] - df_test['close']) / df_test['close']

    threshold = 0.002
    
    if position_sizing == 'fixed':
        df_test['position_size'] = 0.0
        if allow_short:
            # Long when positive prediction, short when negative
            df_test.loc[df_test['pred_return'] > threshold, 'position_size'] = 0.5
            df_test.loc[df_test['pred_return'] < -threshold, 'position_size'] = -0.5
        else:
            df_test.loc[df_test['pred_return'] > threshold, 'position_size'] = 0.5
        
    elif position_sizing == 'proportional':
        df_test['position_size'] = 0.0
        if allow_short:
            # Position size proportional to predicted return (can be negative for shorts)
            df_test['position_size'] = df_test['pred_return'].clip(-max_position, max_position)
        else:
            mask = df_test['pred_return'] > threshold
            df_test.loc[mask, 'position_size'] = df_test.loc[mask, 'pred_return'].clip(0, max_position)
        
    elif position_sizing == 'kelly':
        df_test['position_size'] = 0.0
        if allow_short:
            # Kelly for both long and short
            kelly_fraction = (df_test['pred_return'] * 2).clip(-max_position * 0.5, max_position * 0.5)
            df_test['position_size'] = kelly_fraction
        else:
            mask = df_test['pred_return'] > threshold
            kelly_fraction = (df_test.loc[mask, 'pred_return'] * 2).clip(0, max_position * 0.5)
            df_test.loc[mask, 'position_size'] = kelly_fraction
        
    elif position_sizing == 'volatility_adjusted':
        lookback = 20
        df_test['volatility'] = df_test['close'].pct_change().rolling(lookback).std()
        median_vol = df_test['volatility'].median()
        
        df_test['position_size'] = 0.0
        if allow_short:
            # Both long and short, adjusted by volatility
            vol_adj = median_vol / df_test['volatility']
            df_test['position_size'] = (df_test['pred_return'] * vol_adj * 10).clip(-max_position, max_position)
        else:
            mask = df_test['pred_return'] > threshold
            vol_adj_size = (median_vol / df_test.loc[mask, 'volatility']).clip(0.1, max_position)
            df_test.loc[mask, 'position_size'] = vol_adj_size
        
    else:  # binary (original)
        df_test['position_size'] = 0.0
        if allow_short:
            df_test.loc[df_test['pred_return'] > threshold, 'position_size'] = 1.0
            df_test.loc[df_test['pred_return'] < -threshold, 'position_size'] = -1.0
        else:
            df_test.loc[df_test['pred_return'] > threshold, 'position_size'] = 1.0

    df_test['signal'] = (df_test['position_size'] > 0).astype(int)
    
    df_test['actual_next_close'] = df_test['target']
    df_test['actual_return'] = (df_test['actual_next_close'] - df_test['close']) / df_test['close']
    df_test['strategy_return'] = df_test['position_size'] * df_test['actual_return']
    
    if hybrid_weight > 0:
        df_test['hybrid_return'] = (1 - hybrid_weight) * df_test['strategy_return'] + hybrid_weight * df_test['actual_return']
        df_test['cumulative'] = (1 + df_test['hybrid_return']).cumprod()
    else:
        df_test['cumulative'] = (1 + df_test['strategy_return']).cumprod()
    
    df_test['buy_and_hold'] = (1 + df_test['actual_return']).cumprod()
    
    df_test['portfolio_value'] = initial_capital * df_test['cumulative']
    df_test['bh_portfolio_value'] = initial_capital * df_test['buy_and_hold']

    return df_test

def create_lstm_sequences(X, y, lookback=60):
    """
    Create sequences for LSTM input.
    
    Args:
        X: Feature matrix
        y: Target values
        lookback: Number of time steps to look back
    
    Returns:
        X_seq, y_seq: Sequenced data
    """
    X_seq, y_seq = [], []
    for i in range(lookback, len(X)):
        X_seq.append(X[i-lookback:i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)

def build_lstm_model(input_shape, learning_rate=0.001):
    """
    Build LSTM model architecture.
    
    Args:
        input_shape: Tuple of (timesteps, features)
        learning_rate: Learning rate for optimizer
    
    Returns:
        Compiled Keras model
    """
    model = Sequential([
        Bidirectional(LSTM(128, return_sequences=True), input_shape=input_shape),
        Dropout(0.2),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    return model

def train_lstm_model(X: pd.DataFrame, y: pd.Series, lookback=60, epochs=50, batch_size=32):
    """
    Train LSTM model with proper scaling and early stopping.
    
    Args:
        X: Feature matrix
        y: Target variable
        lookback: Number of timesteps to look back
        epochs: Maximum training epochs
        batch_size: Batch size for training
    
    Returns:
        model: Trained model
        tuple: (X_train, X_test, y_train, y_test, preds, scalers)
    """
    print(f"\nTraining LSTM model (lookback={lookback} steps)...")
    
    # Split data
    train_size = int(len(X) * 0.7)
    X_train_raw = X.iloc[:train_size].values
    y_train_raw = y.iloc[:train_size].values
    X_test_raw = X.iloc[train_size:].values
    y_test_raw = y.iloc[train_size:].values
    
    # Scale features and target
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train_raw)
    X_test_scaled = scaler_X.transform(X_test_raw)
    
    y_train_scaled = scaler_y.fit_transform(y_train_raw.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test_raw.reshape(-1, 1)).flatten()
    
    # Create sequences
    X_train_seq, y_train_seq = create_lstm_sequences(X_train_scaled, y_train_scaled, lookback)
    X_test_seq, y_test_seq = create_lstm_sequences(X_test_scaled, y_test_scaled, lookback)
    
    print(f"Training sequences: {X_train_seq.shape}")
    print(f"Test sequences: {X_test_seq.shape}")
    
    # Build model
    model = build_lstm_model(input_shape=(lookback, X.shape[1]))
    
    # Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)
    
    # Train
    print(f"Training for up to {epochs} epochs...")
    history = model.fit(
        X_train_seq, y_train_seq,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop, reduce_lr],
        verbose=0
    )
    
    # Predict
    y_pred_scaled = model.predict(X_test_seq, verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()
    
    # Adjust y_test to match prediction length
    y_test_adjusted = y_test_raw[lookback:]
    
    rmse = np.sqrt(mean_squared_error(y_test_adjusted, y_pred))
    mae = mean_absolute_error(y_test_adjusted, y_pred)
    
    print(f"LSTM Training Complete!")
    print(f"   Final RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    print(f"   Trained for {len(history.history['loss'])} epochs")
    
    # Return compatible with existing code
    X_train_idx = X.iloc[:train_size].index
    X_test_idx = X.iloc[train_size + lookback:].index
    
    return model, (X.iloc[:train_size], X.iloc[train_size:], 
                   y.iloc[:train_size], y_test_adjusted, y_pred), (scaler_X, scaler_y, lookback)

def train_model(X: pd.DataFrame, y: pd.Series):
    return train_lstm_model(X, y, lookback=60, epochs=50, batch_size=32)

def main(ticker='SPY', period=730, initial_capital=100000.0):
    """
    Main function - automatically tests all strategies and shows the best one.
    Optimized for S&P 500 (SPY) analysis.
    
    Args:
        ticker: Stock ticker symbol (default: SPY for S&P 500)
        period: Days of historical data to fetch (default: 730 days / 2 years)
        initial_capital: Starting portfolio value in USD (default: $100,000)
    """
    print(f"\n{'='*70}")
    print(f"S&P 500 PREDICTION & STRATEGY ANALYSIS: {ticker}")
    print(f"Period: {period} days ({period/365:.1f} years)")
    print(f"Initial Capital: {format_currency(initial_capital)}")
    print(f"{'='*70}\n")
    
    print(f"Fetching price data for {ticker}...")
    prices = fetch_prices_yf(ticker, period_days=period)
    print(f"Got {len(prices)} rows")

    print("\nEngineering features...")
    X, y, df_full = engineer_features(prices, n_lags=5)

    print("\nTraining LSTM model...")
    model, (X_train, X_test, y_train, y_test, preds), scalers = train_model(X, y)

    print("\n" + "="*70)
    print("TESTING ALL POSITION SIZING STRATEGIES...")
    print("="*70)
    
    # Test all position sizing methods (with short selling enabled)
    # Also test hybrid approaches: 50% strategy + 50% B&H, and 70/30 split
    methods = ['binary', 'fixed', 'proportional', 'kelly', 'volatility_adjusted']
    hybrid_configs = [
        ('pure', 0.0),
        ('hybrid_50_50', 0.5),
        ('hybrid_30_70', 0.3),
        ('hybrid_70_30', 0.7)
    ]
    
    results = {}
    
    # Test pure strategies
    for method in methods:
        bt = backtest_strategy(df_full, preds, position_sizing=method, max_position=1.0, 
                              allow_short=True, hybrid_weight=0.0, initial_capital=initial_capital)
        
        # Store key metrics
        strategy_return_col = 'strategy_return' if 'strategy_return' in bt.columns else 'hybrid_return'
        final_value = bt['portfolio_value'].iloc[-1]
        profit_loss = final_value - initial_capital
        
        results[method] = {
            'total_return': bt['cumulative'].iloc[-1] - 1,
            'final_value': final_value,
            'profit_loss': profit_loss,
            'sharpe': calculate_sharpe(bt[strategy_return_col]),
            'max_drawdown': calculate_max_drawdown(bt['cumulative']),
            'win_rate': (bt[strategy_return_col] > 0).sum() / (bt['position_size'] > 0).sum() if (bt['position_size'] > 0).sum() > 0 else 0,
            'backtest_df': bt,
            'type': 'pure'
        }
    
    # Test hybrid strategies with best performing pure strategy
    best_pure = max(results.items(), key=lambda x: x[1]['sharpe'])
    best_method = best_pure[0]
    
    print(f"\nTesting hybrid approaches with {best_method.upper()} strategy...")
    
    for hybrid_name, weight in hybrid_configs[1:]:
        bt = backtest_strategy(df_full, preds, position_sizing=best_method, max_position=1.0, 
                              allow_short=True, hybrid_weight=weight, initial_capital=initial_capital)
        
        strategy_return_col = 'hybrid_return' if 'hybrid_return' in bt.columns else 'strategy_return'
        strategy_pct = int((1 - weight) * 100)
        bh_pct = int(weight * 100)
        final_value = bt['portfolio_value'].iloc[-1]
        profit_loss = final_value - initial_capital
        
        results[f'{best_method}_{hybrid_name}'] = {
            'total_return': bt['cumulative'].iloc[-1] - 1,
            'final_value': final_value,
            'profit_loss': profit_loss,
            'sharpe': calculate_sharpe(bt[strategy_return_col]),
            'max_drawdown': calculate_max_drawdown(bt['cumulative']),
            'win_rate': (bt[strategy_return_col] > 0).sum() / len(bt),
            'backtest_df': bt,
            'type': f'{strategy_pct}/{bh_pct} split'
        }
    
    # Find best strategy based on Sharpe ratio (risk-adjusted returns)
    best_method = max(results.items(), key=lambda x: x[1]['sharpe'])
    
    print("\n" + "="*95)
    print("STRATEGY COMPARISON SUMMARY")
    print("="*95)
    print(f"{'Method':<30} {'Type':<15} {'Return':>10} {'Final Value':>15} {'Profit/Loss':>15} {'Sharpe':>8}")
    print("-"*95)
    
    bh_return = results['binary']['backtest_df']['buy_and_hold'].iloc[-1] - 1
    bh_final_value = initial_capital * (1 + bh_return)
    bh_profit = bh_final_value - initial_capital
    
    best_overall = max(results.items(), key=lambda x: x[1]['total_return'])
    
    pure_results = {k: v for k, v in results.items() if v['type'] == 'pure'}
    hybrid_results = {k: v for k, v in results.items() if v['type'] != 'pure'}
    
    for method, metrics in pure_results.items():
        beats_bh = "YES" if metrics['total_return'] > bh_return else "NO"
        marker = " *BEST*" if method == best_overall[0] else ""
        print(f"{method:<30} {'Pure':<15} {format_percent(metrics['total_return']):>10} "
              f"{format_currency(metrics['final_value']):>15} {format_currency(metrics['profit_loss']):>15} "
              f"{metrics['sharpe']:>8.2f} {beats_bh}{marker}")
    
    if hybrid_results:
        print("-"*95)
        for method, metrics in hybrid_results.items():
            beats_bh = "YES" if metrics['total_return'] > bh_return else "NO"
            marker = " *BEST*" if method == best_overall[0] else ""
            print(f"{method:<30} {metrics['type']:<15} {format_percent(metrics['total_return']):>10} "
                  f"{format_currency(metrics['final_value']):>15} {format_currency(metrics['profit_loss']):>15} "
                  f"{metrics['sharpe']:>8.2f} {beats_bh}{marker}")
    
    print("-"*95)
    print(f"{'Buy & Hold (Benchmark)':<30} {'100% Long':<15} {format_percent(bh_return):>10} "
          f"{format_currency(bh_final_value):>15} {format_currency(bh_profit):>15}")
    print("="*95)
    
    best_overall_method = best_overall[0]
    best_overall_metrics = best_overall[1]
    strategies_beat_bh = [m for m, r in results.items() if r['total_return'] > bh_return]
    
    print(f"\n{'='*70}")
    print("PERFORMANCE SUMMARY")
    print(f"{'='*70}")
    print(f"Initial Capital:      {format_currency(initial_capital)}")
    print(f"Period:               {period} days ({period/365:.1f} years)")
    print()
    
    if strategies_beat_bh:
        print(f"BEST STRATEGY: {best_overall_method.upper()}")
        print(f"  Type:               {best_overall_metrics['type']}")
        print(f"  Final Value:        {format_currency(best_overall_metrics['final_value'])}")
        print(f"  Profit/Loss:        {format_currency(best_overall_metrics['profit_loss'])}")
        print(f"  Total Return:       {format_percent(best_overall_metrics['total_return'])}")
        print(f"  Sharpe Ratio:       {best_overall_metrics['sharpe']:.2f}")
        print(f"  Max Drawdown:       {format_percent(best_overall_metrics['max_drawdown'])}")
        print(f"  Beat B&H by:        {format_percent(best_overall_metrics['total_return'] - bh_return)}")
    else:
        print(f"WARNING: No strategy beat Buy & Hold")
        print(f"  Buy & Hold Return:  {format_percent(bh_return)}")
        print(f"  Buy & Hold Value:   {format_currency(bh_final_value)}")
        print(f"  Buy & Hold Profit:  {format_currency(bh_profit)}")
        print()
        print(f"BEST STRATEGY: {best_overall_method.upper()}")
        print(f"  Type:               {best_overall_metrics['type']}")
        print(f"  Final Value:        {format_currency(best_overall_metrics['final_value'])}")
        print(f"  Profit/Loss:        {format_currency(best_overall_metrics['profit_loss'])}")
        print(f"  Total Return:       {format_percent(best_overall_metrics['total_return'])}")
        print(f"  Sharpe Ratio:       {best_overall_metrics['sharpe']:.2f}")
        print(f"  Underperformance:   {format_percent(best_overall_metrics['total_return'] - bh_return)}")
        
        if 'hybrid' in best_overall_method:
            base_method = best_overall_method.split('_')[0]
            if base_method in results:
                improvement = best_overall_metrics['profit_loss'] - results[base_method]['profit_loss']
                print(f"  Hybrid improved by: {format_currency(improvement)} over pure {base_method}")
        
        print()
        print("RECOMMENDATION: Consider hybrid approaches or different market conditions.")
        print("This strategy may work better in sideways/bear markets.")
    
    print(f"{'='*70}\n")
    
    # Plot the best strategy
    print(f"\nPlotting {best_overall_method.upper()} strategy results...")
    bt_best = results[best_overall_method]['backtest_df']
    
    plt.figure(figsize=(14, 8))
    
    # Calculate portfolio values from $100k
    initial_capital = 100000.0
    strategy_values = initial_capital * bt_best['cumulative']
    bh_values = initial_capital * bt_best['buy_and_hold']
    
    # Subplot 1: Portfolio value in dollars
    plt.subplot(2, 1, 1)
    plt.plot(bt_best.index, strategy_values, label=f'{best_overall_method.upper()}', linewidth=2, color='green')
    plt.plot(bt_best.index, bh_values, label='Buy & Hold', linewidth=2, alpha=0.7, color='blue')
    
    # Add final values as annotations
    final_strategy = strategy_values.iloc[-1]
    final_bh = bh_values.iloc[-1]
    profit_strategy = final_strategy - initial_capital
    profit_bh = final_bh - initial_capital
    
    plt.text(0.02, 0.98, f'{best_overall_method.upper()}: {format_currency(final_strategy)} ({format_currency(profit_strategy)} profit)', 
             transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='green', alpha=0.3))
    plt.text(0.02, 0.90, f'Buy & Hold: {format_currency(final_bh)} ({format_currency(profit_bh)} profit)', 
             transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='blue', alpha=0.3))
    
    plt.legend(fontsize=12)
    plt.title(f'{ticker} - Best Strategy: {best_overall_method.upper()} ({best_overall_metrics["type"]}) | Starting Capital: {format_currency(initial_capital)}', 
              fontsize=14, fontweight='bold')
    plt.ylabel('Portfolio Value ($)', fontsize=11)
    plt.grid(alpha=0.3)
    
    # Subplot 2: Position sizes over time
    plt.subplot(2, 1, 2)
    plt.bar(bt_best.index, bt_best['position_size'], label='Position Size', alpha=0.6, width=1, color='orange')
    plt.ylabel('Position Size', fontsize=11)
    plt.xlabel('Date', fontsize=11)
    plt.legend(fontsize=12)
    plt.title('Dynamic Position Sizing Over Time', fontsize=12)
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Save chart with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    chart_filename = f'charts/{ticker}_{best_overall_method}_{timestamp}.png'
    plt.savefig(chart_filename, dpi=150, bbox_inches='tight')
    print(f"Chart saved as {chart_filename}")
    plt.show()
    
    return results, best_overall_method

def calculate_sharpe(returns, risk_free_rate=0.0):
    trading_days = 252
    mean_return = returns.mean() * trading_days
    std_return = returns.std() * np.sqrt(trading_days)
    return (mean_return - risk_free_rate) / std_return if std_return > 0 else 0

def calculate_max_drawdown(cumulative_returns):
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - running_max) / running_max
    return drawdown.min()

def format_currency(value):
    return f"${value:,.2f}"

def format_percent(value):
    return f"{value:.2%}"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='S&P 500 Prediction & Strategy Analysis')
    parser.add_argument('--ticker', type=str, default='SPY', help='Stock ticker symbol (default: SPY - S&P 500 ETF)')
    parser.add_argument('--period', type=int, default=365, help='Days of historical data (default: 365 - 1 year)')
    args = parser.parse_args()
    
    # Create charts directory if it doesn't exist
    os.makedirs('charts', exist_ok=True)
    
    results, best_strategy = main(ticker=args.ticker, period=args.period)