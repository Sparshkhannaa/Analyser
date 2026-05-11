import numpy as np
import pandas as pd
from xgboost import XGBClassifier


def train_xgb(X: pd.DataFrame, y: pd.Series) -> XGBClassifier:
    split = int(len(X) * 0.8)
    X_tr, X_val = X.iloc[:split], X.iloc[split:]
    y_tr, y_val = y.iloc[:split], y.iloc[split:]

    model = XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="auc",
        early_stopping_rounds=30,
        random_state=42,
        verbosity=0,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return model


def walk_forward_predict(
    X: pd.DataFrame,
    y: pd.Series,
    min_train_days: int = 504,
    step_days: int = 63,
    threshold: float = 0.55,
) -> tuple[pd.DataFrame, XGBClassifier]:
    n = len(X)
    if n <= min_train_days:
        raise ValueError(
            f"Not enough data: {n} feature rows but min_train_days={min_train_days}. "
            "Fetch more history (e.g. --period 1825) or reduce min_train_days."
        )

    records = []
    model = None
    start = min_train_days

    while start < n:
        end = min(start + step_days, n)

        model = train_xgb(X.iloc[:start], y.iloc[:start])
        probs = model.predict_proba(X.iloc[start:end])[:, 1]

        for i, (idx, prob) in enumerate(zip(X.index[start:end], probs)):
            records.append(
                {
                    "date": idx,
                    "actual": int(y.iloc[start + i]),
                    "prob": float(prob),
                    "signal": int(prob >= threshold),
                }
            )
        start = end

    signals = pd.DataFrame(records).set_index("date")
    return signals, model
