import os
import argparse
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

try:
    import yfinance as yf
except Exception:
    raise ImportError("yfinance not found")

try:
    from transformers import pipeline
    SENT_PIPELINE = pipeline("sentiment-analysis")
except Exception:
    SENT_PIPELINE = None

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

    for lag in range(1, n_lags + 1):
        data[f'lag_{lag}'] = data['close'].shift(lag)

    data['rolling_mean_3'] = data['close'].rolling(window=7).mean()
    data['rolling_mean_7'] = data['close'].rolling(window=10).mean()
    data['pct_change_1'] = data['close'].pct_change(1)
    data['volume_change_1'] = data['Volume'].pct_change(1)

    data['target'] = data['close'].shift(-1)
    print(data.head())
    data = data.dropna()
    for c in data.columns:
        print(c, type(c))

    features = data[[c for c in data.columns
                     if isinstance(c, str) and
                        (c.startswith('lag_') or 'rolling_mean' in c or c.endswith('_change'))]]
    target = data['target']

    return features, target, data
def backtest_strategy(df_with_features: pd.DataFrame, preds: np.ndarray, sentiment_df: pd.DataFrame = None):
    df = df_with_features.copy()
    test_idx = df.index[-len(preds):]
    df_test = df.loc[test_idx].copy()
    df_test['pred_next_close'] = preds
    df_test['pred_return'] = (df_test['pred_next_close'] - df_test['close']) / df_test['close']

    threshold = 0.001
    df_test['signal'] = 0
    df_test.loc[df_test['pred_return'] > threshold, 'signal'] = 1

    df_test['actual_next_close'] = df_test['target']
    df_test['actual_return'] = (df_test['actual_next_close'] - df_test['close']) / df_test['close']
    df_test['strategy_return'] = df_test['signal'] * df_test['actual_return']
    df_test['cumulative'] = (1 + df_test['strategy_return']).cumprod()
    df_test['buy_and_hold'] = (1 + df_test['actual_return']).cumprod()

    total_strategy = df_test['cumulative'].iloc[-1] - 1
    total_bh = df_test['buy_and_hold'].iloc[-1] - 1
    print(f"Strategy total return: {total_strategy:.2%}, Buy & Hold return: {total_bh:.2%}")

    plt.figure(figsize=(10, 5))
    plt.plot(df_test.index, df_test['cumulative'], label='Strategy equity')
    plt.plot(df_test.index, df_test['buy_and_hold'], label='Buy & Hold')
    plt.legend()
    plt.title('Strategy vs Buy & Hold')
    plt.show()

    return df_test

def train_model(X: pd.DataFrame, y: pd.Series):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    print(f"Model test RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    return model, (X_train, X_test, y_train, y_test, preds)

def main(ticker='AAPL', period=365):
    print(f"Fetching price data for {ticker}...")
    prices = fetch_prices_yf(ticker, period_days=period)
    print(f"Got {len(prices)} rows")

    print("Engineering features...")
    X, y, df_full = engineer_features(prices, n_lags=5)

    print("Training model...")
    model, (X_train, X_test, y_train, y_test, preds) = train_model(X, y)

 #   joblib.dump(model, f"{ticker}_rf_model.joblib")
    print("Backtesting...")
    _, X_test_idx = X_train.shape, X_test.index
    test_preds = preds
    bt = backtest_strategy(df_full, test_preds)

if __name__ == '__main__':
    print(sklearn.__version__)
    print(sklearn.__file__)
    parser = argparse.ArgumentParser(description='Run stock prediction')
    parser.add_argument('--ticker', type=str, default='AAPL')
    parser.add_argument('--period', type=int, default=365)
    args = parser.parse_args()
    main(ticker=args.ticker, period=args.period)