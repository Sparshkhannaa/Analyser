import numpy as np
import pandas as pd


def run_backtest(
    prices: pd.DataFrame,
    signals: pd.DataFrame,
    initial_capital: float = 100_000.0,
    cost_pct: float = 0.001,
    hold_days: int = 5,
    stop_loss_pct: "float | None" = 0.02,
    earnings_dates: "pd.DatetimeIndex | None" = None,
    earnings_window: int = 2,
) -> pd.DataFrame:
    all_dates = prices.index

    position_prob = pd.Series(0.0, index=all_dates)
    stop_exit = pd.Series(False, index=all_dates)
    stop_ret = pd.Series(0.0, index=all_dates)
    in_position_until = pd.Timestamp.min

    near = set()
    if earnings_dates is not None and len(earnings_dates) > 0:
        for ed in earnings_dates:
            ed = pd.Timestamp(ed).normalize()
            for delta in range(-earnings_window, earnings_window + 1):
                near.add(ed + pd.Timedelta(days=delta))

    for date in signals.index:
        if date not in all_dates:
            continue
        row = signals.loc[date]
        if row["signal"] != 1 or date <= in_position_until:
            continue
        if pd.Timestamp(date).normalize() in near:
            continue
        idx = all_dates.get_loc(date)
        if idx + 1 >= len(all_dates):
            continue
        entry_price = float(prices["Open"].iloc[idx + 1])
        stop_level = entry_price * (1 - stop_loss_pct) if stop_loss_pct else None
        end_idx = min(idx + hold_days - 1, len(all_dates) - 2)
        actual_end = end_idx

        for hold_i in range(idx, end_idx + 1):
            next_i = hold_i + 1
            if next_i >= len(all_dates):
                actual_end = hold_i
                break
            position_prob.iloc[hold_i] = row["prob"]
            if stop_level is not None:
                day_open = float(prices["Open"].iloc[next_i])
                if float(prices["Low"].iloc[next_i]) <= stop_level:
                    stop_exit.iloc[hold_i] = True
                    stop_ret.iloc[hold_i] = (stop_level - day_open) / day_open
                    actual_end = hold_i
                    break
        else:
            actual_end = end_idx

        in_position_until = all_dates[actual_end]

    df = prices[["Open", "Close"]].copy()
    pos_size = ((position_prob - 0.5) * 4).clip(0, 1.0)
    df["position_size"] = pos_size

    df["exec_open"] = df["Open"].shift(-1)
    df["next_close"] = df["Close"].shift(-1)
    df = df.dropna(subset=["exec_open", "next_close"])

    df["trade_return"] = (df["next_close"] - df["exec_open"]) / df["exec_open"]

    stop_mask = stop_exit.reindex(df.index).fillna(False).astype(bool)
    df.loc[stop_mask, "trade_return"] = stop_ret.reindex(df.index)[stop_mask]

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
