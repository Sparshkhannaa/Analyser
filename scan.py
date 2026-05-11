import argparse
import os
import warnings
from datetime import date

warnings.filterwarnings("ignore")

import pandas as pd

from features import (
    engineer_features,
    engineer_features_inference,
    fetch_earnings_dates,
    fetch_prices,
    fetch_vix,
    is_near_earnings,
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
            earnings_dates = fetch_earnings_dates(ticker)
            near_earnings = is_near_earnings(date.today(), earnings_dates)
            signal = 1 if (prob >= threshold and regime_ok and not near_earnings) else 0
            results.append({
                "ticker": ticker,
                "prob": prob,
                "signal": signal,
                "regime_ok": regime_ok,
                "near_earnings": near_earnings,
                "close": float(prices["Close"].iloc[-1]),
                "sma_200": sma_200,
            })
        except Exception as e:
            print(f"  [{ticker}] SKIPPED — {e}")
            continue

    if not results:
        return pd.DataFrame(columns=["ticker", "prob", "signal", "regime_ok", "near_earnings", "close", "sma_200"])

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
    print(f"  {'Ticker':<8} {'Prob':>6}  {'Signal':<8} {'Regime':<8} {'Earnings'}")
    print(f"  {'-' * 7} {'-' * 6}  {'-' * 7} {'-' * 7} {'-' * 8}")
    for _, row in df.iterrows():
        signal_str = "LONG" if row["signal"] == 1 else "-"
        regime_str = "YES" if row["regime_ok"] else "NO"
        earnings_str = "NEAR" if row["near_earnings"] else "-"
        print(f"  {row['ticker']:<8} {row['prob']:>6.2f}  {signal_str:<8} {regime_str:<8} {earnings_str}")
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
