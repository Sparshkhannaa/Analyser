import numpy as np
import pandas as pd
import pytest
from features import engineer_features


def test_train_xgb_returns_probabilities(synthetic_prices, synthetic_vix):
    from model import train_xgb

    X, y = engineer_features(synthetic_prices, synthetic_vix)
    model = train_xgb(X, y)
    probs = model.predict_proba(X)

    assert probs.shape == (len(X), 2)
    assert (probs >= 0).all()
    assert (probs <= 1).all()


def test_train_xgb_probabilities_sum_to_one(synthetic_prices, synthetic_vix):
    from model import train_xgb

    X, y = engineer_features(synthetic_prices, synthetic_vix)
    model = train_xgb(X, y)
    probs = model.predict_proba(X)

    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)


def test_walk_forward_returns_dataframe(synthetic_prices, synthetic_vix):
    from model import walk_forward_predict

    X, y = engineer_features(synthetic_prices, synthetic_vix)
    signals, model = walk_forward_predict(X, y, min_train_days=252, step_days=63)

    assert isinstance(signals, pd.DataFrame)
    assert set(signals.columns) == {"actual", "prob", "signal"}


def test_walk_forward_no_duplicate_dates(synthetic_prices, synthetic_vix):
    from model import walk_forward_predict

    X, y = engineer_features(synthetic_prices, synthetic_vix)
    signals, _ = walk_forward_predict(X, y, min_train_days=252, step_days=63)

    assert signals.index.is_unique, "Walk-forward produced duplicate prediction dates"


def test_walk_forward_signals_are_binary(synthetic_prices, synthetic_vix):
    from model import walk_forward_predict

    X, y = engineer_features(synthetic_prices, synthetic_vix)
    signals, _ = walk_forward_predict(X, y, min_train_days=252, step_days=63)

    assert set(signals["signal"].unique()).issubset({0, 1})


def test_walk_forward_dates_in_x_index(synthetic_prices, synthetic_vix):
    from model import walk_forward_predict

    X, y = engineer_features(synthetic_prices, synthetic_vix)
    signals, _ = walk_forward_predict(X, y, min_train_days=252, step_days=63)

    assert signals.index.isin(X.index).all(), "Signal dates not all in feature index"


def test_walk_forward_returns_last_model(synthetic_prices, synthetic_vix):
    from model import walk_forward_predict
    from xgboost import XGBClassifier

    X, y = engineer_features(synthetic_prices, synthetic_vix)
    _, model = walk_forward_predict(X, y, min_train_days=252, step_days=63)

    assert isinstance(model, XGBClassifier)
    assert hasattr(model, "feature_importances_")
