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


import json
from datetime import datetime, timedelta, timezone


def test_save_load_roundtrip(tmp_path, monkeypatch, synthetic_prices, synthetic_vix):
    from model import save_model, load_model, train_xgb
    monkeypatch.setattr("model.MODEL_DIR", str(tmp_path))

    X, y = engineer_features(synthetic_prices, synthetic_vix)
    model = train_xgb(X, y)
    save_model(model, "TEST")

    loaded = load_model("TEST")
    preds_orig = model.predict_proba(X)[:, 1]
    preds_load = loaded.predict_proba(X)[:, 1]
    np.testing.assert_array_almost_equal(preds_orig, preds_load)


def test_model_is_stale_missing(tmp_path, monkeypatch):
    from model import model_is_stale
    monkeypatch.setattr("model.MODEL_DIR", str(tmp_path))
    assert model_is_stale("MISSING") is True


def test_model_is_stale_fresh(tmp_path, monkeypatch):
    from model import model_is_stale
    monkeypatch.setattr("model.MODEL_DIR", str(tmp_path))

    meta = {"trained_at": datetime.now(timezone.utc).isoformat()}
    (tmp_path / "FRESH_xgb.pkl").write_text("dummy")
    (tmp_path / "FRESH_meta.json").write_text(json.dumps(meta))

    assert model_is_stale("FRESH") is False


def test_model_is_stale_old(tmp_path, monkeypatch):
    from model import model_is_stale
    monkeypatch.setattr("model.MODEL_DIR", str(tmp_path))

    old_time = datetime.now(timezone.utc) - timedelta(days=8)
    meta = {"trained_at": old_time.isoformat()}
    (tmp_path / "OLD_xgb.pkl").write_text("dummy")
    (tmp_path / "OLD_meta.json").write_text(json.dumps(meta))

    assert model_is_stale("OLD") is True
