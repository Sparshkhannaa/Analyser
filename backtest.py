import numpy as np
import pandas as pd


def run_backtest(
    prices: pd.DataFrame,
    signals: pd.DataFrame,
    initial_capital: float = 100_000.0,
    cost_pct: float = 0.001,
    hold_days: int = 5,
) -> pd.DataFrame:
    all_dates = prices.index

    # Expand each signal across hold_days, non-overlapping
    # Position size scales with confidence: (prob - 0.5) * 4, capped at 1.0
    position_prob = pd.Series(0.0, index=all_dates)
    in_position_until = pd.Timestamp.min

    for date in signals.index:
        if date not in all_dates:
            continue
        row = signals.loc[date]
        if row["signal"] != 1 or date <= in_position_until:
            continue
        idx = all_dates.get_loc(date)
        end_idx = min(idx + hold_days - 1, len(all_dates) - 2)
        position_prob.iloc[idx : end_idx + 1] = row["prob"]
        in_position_until = all_dates[end_idx]

    df = prices[["Open", "Close"]].copy()
    pos_size = ((position_prob - 0.5) * 4).clip(0, 1.0)
    df["position_size"] = pos_size

    # Execute at next-day open; return to that day's close
    df["exec_open"] = df["Open"].shift(-1)
    df["next_close"] = df["Close"].shift(-1)
    df = df.dropna(subset=["exec_open", "next_close"])

    df["trade_return"] = (df["next_close"] - df["exec_open"]) / df["exec_open"]

    prev_pos = df["position_size"].shift(1).fillna(0)
    df["cost"] = (df["position_size"] - prev_pos).abs() * cost_pct
    df["strategy_return"] = df["position_size"] * df["trade_return"] - df["cost"]

    df["cumulative"] = (1 + df["strategy_return"]).cumprod()
    df["portfolio_value"] = initial_capital * df["cumulative"]

    df["bh_daily"] = df["Close"].pct_change().fillna(0)
    df["bh_cumulative"] = (1 + df["bh_daily"]).cumprod()
    df["bh_value"] = initial_capital * df["bh_cumulative"]

    return df


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
    max_drawdown = float(drawdown.min())

    trades = bt[bt["position_size"] > 0]
    win_rate = float((trades["strategy_return"] > 0).mean()) if len(trades) > 0 else 0.0

    sharpe = (ann_return - risk_free_rate) / ann_vol if ann_vol > 1e-9 else 0.0
    sortino = (ann_return - risk_free_rate) / ann_downside if ann_downside > 1e-9 else 0.0
    calmar = ann_return / abs(max_drawdown) if max_drawdown < -1e-9 else 0.0

    return {
        "total_return": float(cumulative.iloc[-1] - 1),
        "ann_return": float(ann_return),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "calmar": float(calmar),
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "trade_count": int((bt["position_size"] > 0).sum()),
        "bh_return": float(bt["bh_cumulative"].iloc[-1] - 1),
        "beats_bh": bool(
            (cumulative.iloc[-1] - 1) > (bt["bh_cumulative"].iloc[-1] - 1)
        ),
    }
