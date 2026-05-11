import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_prices():
    np.random.seed(42)
    n = 600
    dates = pd.bdate_range("2020-01-01", periods=n)
    close = 400 + np.cumsum(np.random.randn(n) * 2)
    high = close + np.abs(np.random.randn(n))
    low = close - np.abs(np.random.randn(n))
    open_ = close + np.random.randn(n) * 0.5
    volume = np.random.randint(50_000_000, 150_000_000, n).astype(float)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


@pytest.fixture
def synthetic_vix():
    np.random.seed(43)
    n = 600
    dates = pd.bdate_range("2020-01-01", periods=n)
    vix = np.clip(15 + np.random.randn(n) * 5, 10, 80)
    return pd.DataFrame({"Close": vix}, index=dates)
