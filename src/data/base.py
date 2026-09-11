"""
Abstract base class for all data providers in the R3T trading system.

Concrete providers (yfinance, Fyers, etc.) must implement this interface
so that higher-level components remain provider-agnostic.
"""

from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd


class AbstractDataProvider(ABC):
    """Base interface every data provider must satisfy."""

    @abstractmethod
    def get_historical(
        self,
        symbol: str,
        interval: str,
        period: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data for *symbol*.

        Returns
        -------
        pd.DataFrame
            DataFrame with a DatetimeIndex and columns Open, High, Low, Close, Volume.
        """

    @abstractmethod
    def get_name(self) -> str:
        """Return a short, human-readable identifier for this provider."""
