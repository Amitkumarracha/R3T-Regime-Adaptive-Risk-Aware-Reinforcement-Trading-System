"""
High-level data loader for the R3T trading system.
"""

import logging
from typing import Any, Optional

import pandas as pd

from .feature_engineering import add_technical_indicators
from .yfinance_provider import YFinanceProvider

logger = logging.getLogger(__name__)


def load_data(
    symbol: str,
    provider: str = "yfinance",
    interval: str = "1d",
    period: str = "5y",
    start: Optional[str] = None,
    end: Optional[str] = None,
    config: Optional[Any] = None,
) -> pd.DataFrame:
    """Fetch raw OHLCV data and enrich it with technical indicators."""
    
    if provider.lower() == "yfinance":
        data_provider = YFinanceProvider()
    elif provider.lower() == "fyers":
        from .fyers_provider import FyersProvider
        data_provider = FyersProvider()
    else:
        raise ValueError(f"Unknown provider '{provider}'")

    raw_df = data_provider.get_historical(
        symbol=symbol, interval=interval, period=period, start=start, end=end
    )

    if raw_df.empty:
        return pd.DataFrame()

    df = add_technical_indicators(raw_df)
    return df

