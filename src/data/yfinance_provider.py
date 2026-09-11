"""
yfinance-backed data provider for the R3T trading system.
"""

import logging
from typing import Optional

import pandas as pd
import yfinance as yf

from .base import AbstractDataProvider

logger = logging.getLogger(__name__)


class YFinanceProvider(AbstractDataProvider):
    """Fetch historical price data from Yahoo Finance.

    Requires no API credentials. Default provider for research / back-testing.
    """

    def __init__(self, auto_adjust: bool = True) -> None:
        self._auto_adjust = auto_adjust

    def get_name(self) -> str:
        return "yfinance"

    def get_historical(
        self,
        symbol: str,
        interval: str,
        period: Optional[str] = "5y",
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Download OHLCV bars from Yahoo Finance."""
        try:
            download_kwargs: dict = dict(
                tickers=symbol,
                interval=interval,
                auto_adjust=self._auto_adjust,
                progress=False,
            )
            if start is not None and end is not None:
                download_kwargs["start"] = start
                download_kwargs["end"] = end
            else:
                download_kwargs["period"] = period or "5y"

            raw: pd.DataFrame = yf.download(**download_kwargs)

            if raw.empty:
                logger.warning("yfinance returned no data for symbol=%s", symbol)
                return pd.DataFrame()

            df = self._flatten_columns(raw)
            df = self._select_ohlcv(df)
            df = df.dropna()

            logger.info(
                "[yfinance] %s | %s | rows=%d",
                symbol, interval, len(df),
            )
            return df

        except Exception as exc:
            logger.error("yfinance download failed for %s: %s", symbol, exc, exc_info=True)
            return pd.DataFrame()

    @staticmethod
    def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Collapse yfinance MultiIndex columns to a flat level."""
        if isinstance(df.columns, pd.MultiIndex):
            if df.columns.nlevels == 2:
                df.columns = df.columns.get_level_values(0)
            else:
                df.columns = ["_".join(str(c) for c in col).strip() for col in df.columns]
        return df

    @staticmethod
    def _select_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame containing only the OHLCV columns."""
        df.columns = [str(c).strip().title() for c in df.columns]
        required = ["Open", "High", "Low", "Close", "Volume"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"yfinance data missing columns: {missing}. Available: {list(df.columns)}")
        return df[required].copy()
