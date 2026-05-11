import json
import os
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier


def train_xgb(X: pd.DataFrame, y: pd.Series) -> XGBClassifier:
    split = int(len(X) * 0.8)
    X_tr, X_val = X.iloc[:split], X.iloc[split:]
    y_tr, y_val = y.iloc[:split], y.iloc[split:]

    neg = (y_tr == 0).sum()
    pos = (y_tr == 1).sum()
    scale_pos_weight = neg / pos if pos > 0 else 1.0

    model = XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
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


MODEL_DIR = "models"


def save_model(model: XGBClassifier, ticker: str) -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, f"{MODEL_DIR}/{ticker}_xgb.pkl")
    meta = {"trained_at": datetime.now(timezone.utc).isoformat()}
    with open(f"{MODEL_DIR}/{ticker}_meta.json", "w") as f:
        json.dump(meta, f)


def load_model(ticker: str) -> XGBClassifier:
    return joblib.load(f"{MODEL_DIR}/{ticker}_xgb.pkl")


def model_is_stale(ticker: str, max_age_days: int = 7) -> bool:
    pkl_path = f"{MODEL_DIR}/{ticker}_xgb.pkl"
    meta_path = f"{MODEL_DIR}/{ticker}_meta.json"
    if not os.path.exists(pkl_path) or not os.path.exists(meta_path):
        return True
    with open(meta_path) as f:
        meta = json.load(f)
    trained_at = datetime.fromisoformat(meta["trained_at"])
    if trained_at.tzinfo is None:
        trained_at = trained_at.replace(tzinfo=timezone.utc)
    age = datetime.now(timezone.utc) - trained_at
    return age.days > max_age_days
