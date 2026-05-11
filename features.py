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

    # Binary target: 1 if next-day close > today (any up day)
    # Costs live in the backtest, not here — keeping target clean avoids class imbalance
    next_close = close.shift(-1)
    data["target"] = next_close.gt(close).astype(float).where(next_close.notna())

    data = data.dropna()

    return data[_FEATURE_COLS], data["target"]
