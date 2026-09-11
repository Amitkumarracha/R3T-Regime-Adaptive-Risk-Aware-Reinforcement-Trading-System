"""
Fyers API v3 data provider for the R3T trading system.

Required environment variables:
  FYERS_APP_ID       – Your Fyers app ID
  FYERS_ACCESS_TOKEN – A valid Fyers access token (generate via scripts/fyers_auth.py)
"""

import logging
import os
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

from .base import AbstractDataProvider

logger = logging.getLogger(__name__)

load_dotenv()

_INTERVAL_MAP: dict = {
    "1m": "1", "2m": "2", "3m": "3", "5m": "5", "10m": "10",
    "15m": "15", "20m": "20", "25m": "25", "30m": "30", "45m": "45",
    "60m": "60", "1h": "60", "2h": "120", "4h": "240",
    "1d": "D", "1w": "W", "1mo": "M",
}


class FyersProvider(AbstractDataProvider):
    """Fetch historical price data from the Fyers API v3 for NSE instruments.

    Credentials are read from FYERS_APP_ID and FYERS_ACCESS_TOKEN environment
    variables. Run scripts/fyers_auth.py once to generate the access token.
    """

    def __init__(self) -> None:
        self._app_id: Optional[str] = os.environ.get("FYERS_APP_ID")
        self._access_token: Optional[str] = os.environ.get("FYERS_ACCESS_TOKEN")
        self._fyers = None

        if not self._app_id or not self._access_token:
            logger.warning(
                "FyersProvider: FYERS_APP_ID or FYERS_ACCESS_TOKEN not set. "
                "Run scripts/fyers_auth.py to generate a token. "
                "All data requests will return empty DataFrames until credentials are set."
            )
        else:
            self._init_client()

    def get_name(self) -> str:
        return "fyers"

    def get_historical(
        self,
        symbol: str,
        interval: str,
        period: Optional[str] = "1y",
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV bars from Fyers.

        symbol: TCS.NS format (auto-converted) or NSE:TCS-EQ (native Fyers format)
        """
        if self._fyers is None:
            logger.warning("FyersProvider: client not initialised — returning empty DataFrame.")
            return pd.DataFrame()

        fyers_symbol = self._to_fyers_symbol(symbol)
        resolution = _INTERVAL_MAP.get(interval, "D")
        range_from, range_to = self._resolve_date_range(period, start, end)

        data = {
            "symbol": fyers_symbol,
            "resolution": resolution,
            "date_format": "1",
            "range_from": range_from,
            "range_to": range_to,
            "cont_flag": "1",
        }

        try:
            response = self._fyers.history(data=data)
        except Exception as exc:
            logger.error("Fyers history API call failed for %s: %s", fyers_symbol, exc)
            return pd.DataFrame()

        if response.get("s") != "ok":
            logger.error("Fyers history error for %s: %s", fyers_symbol, response)
            return pd.DataFrame()

        df = self._parse_response(response)
        logger.info("[fyers] %s | %s | rows=%d", fyers_symbol, interval, len(df))
        return df

    def _init_client(self) -> None:
        try:
            from fyers_apiv3 import fyersModel
            self._fyers = fyersModel.FyersModel(
                client_id=self._app_id,
                token=self._access_token,
                log_path="",
            )
            logger.info("FyersProvider: client initialised (app_id=%s).", self._app_id)
        except ImportError:
            logger.error("fyers_apiv3 not installed. Run: pip install fyers-apiv3")
        except Exception as exc:
            logger.error("FyersProvider: failed to initialise client: %s", exc)

    @staticmethod
    def _to_fyers_symbol(symbol: str) -> str:
        """Convert TCS.NS → NSE:TCS-EQ (pass-through if already Fyers format)."""
        if ":" in symbol:
            return symbol
        ticker = symbol.upper()
        exchange = "NSE"
        if ticker.endswith(".NS"):
            ticker = ticker[:-3]
        elif ticker.endswith(".BO"):
            ticker = ticker[:-3]
            exchange = "BSE"
        _indices = {"NIFTY50", "NIFTY", "BANKNIFTY", "SENSEX", "FINNIFTY"}
        suffix = "INDEX" if ticker in _indices else "EQ"
        return f"{exchange}:{ticker}-{suffix}"

    @staticmethod
    def _resolve_date_range(period, start, end):
        today = datetime.today()
        if start and end:
            return start, end
        if start:
            return start, today.strftime("%Y-%m-%d")
        period = (period or "1y").lower().strip()
        if period.endswith("y"):
            delta = timedelta(days=365 * int(period[:-1]))
        elif period.endswith("m"):
            delta = timedelta(days=30 * int(period[:-1]))
        elif period.endswith("d"):
            delta = timedelta(days=int(period[:-1]))
        else:
            delta = timedelta(days=365)
        return (today - delta).strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")

    @staticmethod
    def _parse_response(response: dict) -> pd.DataFrame:
        candles = response.get("candles", [])
        if not candles:
            return pd.DataFrame()
        df = pd.DataFrame(candles, columns=["timestamp", "Open", "High", "Low", "Close", "Volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        df = df.set_index("timestamp")
        df.index = df.index.tz_convert("Asia/Kolkata").tz_localize(None)
        df.index.name = "Date"
        return df.dropna()
