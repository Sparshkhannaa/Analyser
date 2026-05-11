import pandas as pd
import pytest


def test_fetch_prices_returns_ohlcv(monkeypatch):
    import yfinance as yf
    import features

    sample = pd.DataFrame(
        {"Open": [100.0], "High": [101.0], "Low": [99.0], "Close": [100.5], "Volume": [1_000_000.0]},
        index=pd.DatetimeIndex(["2024-01-02"]),
    )

    def mock_download(*args, **kwargs):
        return sample

    monkeypatch.setattr(yf, "download", mock_download)
    df = features.fetch_prices("SPY", period_days=5)
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert len(df) == 1


def test_fetch_vix_uses_vix_ticker(monkeypatch):
    import yfinance as yf
    import features

    captured = {}

    def mock_download(ticker, **kwargs):
        captured["ticker"] = ticker
        return pd.DataFrame(
            {"Open": [20.0], "High": [21.0], "Low": [19.0], "Close": [20.5], "Volume": [0.0]},
            index=pd.DatetimeIndex(["2024-01-02"]),
        )

    monkeypatch.setattr(yf, "download", mock_download)
    features.fetch_vix(period_days=5)
    assert captured["ticker"] == "^VIX"


from features import engineer_features


def test_engineer_features_column_count(synthetic_prices, synthetic_vix):
    X, y = engineer_features(synthetic_prices, synthetic_vix)
    assert X.shape[1] == 44, f"Expected 44 features, got {X.shape[1]}"


def test_engineer_features_no_nan(synthetic_prices, synthetic_vix):
    X, y = engineer_features(synthetic_prices, synthetic_vix)
    assert not X.isna().any().any(), "Feature matrix contains NaN values"


def test_engineer_features_target_is_binary(synthetic_prices, synthetic_vix):
    X, y = engineer_features(synthetic_prices, synthetic_vix)
    assert set(y.unique()).issubset({0, 1}), f"Target has non-binary values: {y.unique()}"


def test_engineer_features_lengths_match(synthetic_prices, synthetic_vix):
    X, y = engineer_features(synthetic_prices, synthetic_vix)
    assert len(X) == len(y)


def test_engineer_features_no_lookahead(synthetic_prices, synthetic_vix):
    X, y = engineer_features(synthetic_prices, synthetic_vix)
    assert X.index[-1] < synthetic_prices.index[-1]


def test_engineer_features_inference_returns_one_row(synthetic_prices, synthetic_vix):
    from features import engineer_features_inference, _FEATURE_COLS
    row = engineer_features_inference(synthetic_prices, synthetic_vix)
    assert row.shape == (1, len(_FEATURE_COLS)), f"Expected (1, 44), got {row.shape}"
    assert list(row.columns) == _FEATURE_COLS
    assert not row.isnull().values.any(), "Inference row contains NaN"


def test_engineer_features_inference_index_after_training_data(synthetic_prices, synthetic_vix):
    from features import engineer_features_inference
    X_train, _ = engineer_features(synthetic_prices, synthetic_vix)
    X_infer = engineer_features_inference(synthetic_prices, synthetic_vix)
    assert X_infer.index[-1] >= X_train.index[-1], (
        "Inference row should be at or after last training row"
    )


def test_is_near_earnings_within_window():
    from features import is_near_earnings
    earnings = pd.DatetimeIndex(["2024-05-10"])
    assert is_near_earnings(pd.Timestamp("2024-05-11"), earnings, window=2) is True
    assert is_near_earnings(pd.Timestamp("2024-05-08"), earnings, window=2) is True


def test_is_near_earnings_outside_window():
    from features import is_near_earnings
    earnings = pd.DatetimeIndex(["2024-05-10"])
    assert is_near_earnings(pd.Timestamp("2024-05-15"), earnings, window=2) is False


def test_is_near_earnings_empty():
    from features import is_near_earnings
    assert is_near_earnings(pd.Timestamp("2024-05-10"), pd.DatetimeIndex([]), window=2) is False
