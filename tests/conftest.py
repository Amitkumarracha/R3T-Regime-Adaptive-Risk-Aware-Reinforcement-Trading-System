"""tests/conftest.py — Shared fixtures for R3T test suite."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_ohlcv():
    """Generate synthetic daily OHLCV data (500 bars)."""
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2020-01-02", periods=n, freq="B")
    close = 3000 + np.cumsum(np.random.randn(n) * 20)
    close = np.clip(close, 500, 10000)
    high  = close * (1 + np.abs(np.random.randn(n) * 0.005))
    low   = close * (1 - np.abs(np.random.randn(n) * 0.005))
    open_ = close + np.random.randn(n) * 10
    vol   = np.random.randint(100000, 1000000, n)
    df = pd.DataFrame({
        "Open": open_, "High": high, "Low": low,
        "Close": close, "Volume": vol.astype(float)
    }, index=dates)
    return df


@pytest.fixture
def sample_config():
    return {
        "initial_capital": 100000.0,
        "slippage_bps": 5,
        "brokerage_per_leg": 20.0,
        "brokerage_pct": 0.0003,
        "lambda_cost": 0.5,
        "lambda_drawdown": 0.8,
        "lambda_volatility": 0.2,
        "lambda_turnover": 0.1,
        "max_daily_loss_pct": 1.5,
        "max_drawdown_pct": 5.0,
        "max_position_pct": 25.0,
        "risk_per_trade_pct": 0.5,
        "atr_stop_multiplier": 3.0,
    }
