import argparse
import os
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd

from backtest import compute_metrics, run_backtest
from features import engineer_features, fetch_earnings_dates, fetch_prices, fetch_vix
from model import walk_forward_predict


def _print_results(metrics: dict, ticker: str, period: int) -> None:
    print(f"\n{'='*65}")
    print(f"  {ticker}  |  {period} days of data")
    print(f"{'='*65}")
    print(f"  {'Strategy Return':<28} {metrics['total_return']:>10.2%}")
    print(f"  {'Buy & Hold Return':<28} {metrics['bh_return']:>10.2%}")
    print(f"  {'Beats Buy & Hold':<28} {'YES' if metrics['beats_bh'] else 'NO':>10}")
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
    ax1.set_title(f"{ticker} — XGBoost Direction Strategy vs Buy & Hold", fontsize=13, fontweight="bold")
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
    threshold: float = 0.52,
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

    print("Fetching earnings dates...")
    earnings_dates = fetch_earnings_dates(ticker)
    print(f"  {len(earnings_dates)} earnings dates found — suppressing signals within ±2 days.")

    print("Backtesting with transaction costs...")
    bt = run_backtest(prices, signals, initial_capital=capital, hold_days=5, earnings_dates=earnings_dates)
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
