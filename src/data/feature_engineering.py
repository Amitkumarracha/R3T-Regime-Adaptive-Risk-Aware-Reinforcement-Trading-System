"""
Feature engineering for the R3T trading system.

Public API:
  - add_technical_indicators(df) -> df with features
  - prepare_dataset(df, feature_cols, scaler=None, fit_scaler=True) -> (df_scaled, scaler)
  - chronological_split(df, train_ratio, val_ratio) -> (train, val, test)
  - FEATURE_COLS: list of feature column names
"""

import logging
import math
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

try:
    import ta
except ImportError:
    raise ImportError("Run: pip install ta")

logger = logging.getLogger(__name__)

# Market + time feature columns (regime cols added by RegimeDetector)
FEATURE_COLS = [
    "returns", "log_returns", "volume_change",
    "SMA_20", "EMA_5", "EMA_20", "EMA_slope",
    "RSI", "MACD", "MACD_signal", "MACD_hist",
    "ATR", "ATR_pct", "realized_vol",
    "momentum_5", "momentum_20",
    "price_to_sma20", "price_to_ema20",
    "minutes_since_open", "minutes_to_close",
    # Regime features (added by RegimeDetector.predict)
    "regime_0", "regime_1", "regime_2", "regime_3", "regime_confidence",
]

_NSE_OPEN_H, _NSE_OPEN_M = 9, 15
_NSE_CLOSE_H, _NSE_CLOSE_M = 15, 30
_SESSION_MINS = (_NSE_CLOSE_H * 60 + _NSE_CLOSE_M) - (_NSE_OPEN_H * 60 + _NSE_OPEN_M)  # 375


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all market and time features and return enriched DataFrame copy."""
    df = df.copy()

    close  = df["Close"]
    high   = df["High"]
    low    = df["Low"]
    volume = df["Volume"]

    df["Close_raw"] = close.values

    # Returns
    df["returns"]      = close.pct_change()
    df["log_returns"]  = np.log(close / close.shift(1))
    df["volume_change"] = volume.pct_change()

    # Trend
    df["SMA_20"] = ta.trend.sma_indicator(close, window=20)
    df["EMA_5"]  = ta.trend.ema_indicator(close, window=5)
    df["EMA_20"] = ta.trend.ema_indicator(close, window=20)
    ema5_shifted = df["EMA_5"].shift(5).replace(0, np.nan)
    df["EMA_slope"] = (df["EMA_5"] - ema5_shifted) / ema5_shifted

    # RSI
    df["RSI"] = ta.momentum.rsi(close, window=14)

    # MACD
    macd = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    df["MACD"]        = macd.macd()
    df["MACD_signal"] = macd.macd_signal()
    df["MACD_hist"]   = df["MACD"] - df["MACD_signal"]

    # ATR
    df["ATR"]     = ta.volatility.average_true_range(high, low, close, window=14)
    df["ATR_pct"] = df["ATR"] / close.replace(0, np.nan)

    # Realized vol
    df["realized_vol"] = df["log_returns"].rolling(20).std() * math.sqrt(252)

    # Momentum
    df["momentum_5"]  = (close - close.shift(5))  / close.shift(5).replace(0, np.nan)
    df["momentum_20"] = (close - close.shift(20)) / close.shift(20).replace(0, np.nan)

    # Price vs MAs
    df["price_to_sma20"] = (close - df["SMA_20"].replace(0, np.nan)) / df["SMA_20"].replace(0, np.nan)
    df["price_to_ema20"] = (close - df["EMA_20"].replace(0, np.nan)) / df["EMA_20"].replace(0, np.nan)

    # Time of day
    idx = df.index
    try:
        # Check if intraday (median gap < 12h)
        if len(idx) >= 2:
            diffs = pd.Series(idx).diff().dropna()
            is_intraday = diffs.median() < pd.Timedelta(hours=12)
        else:
            is_intraday = False

        if is_intraday:
            try:
                if hasattr(idx, 'tz') and idx.tz is not None:
                    import pytz
                    idx_ist = idx.tz_convert(pytz.timezone('Asia/Kolkata'))
                else:
                    idx_ist = idx
                bar_minutes = idx_ist.hour * 60 + idx_ist.minute
                open_min = _NSE_OPEN_H * 60 + _NSE_OPEN_M
                elapsed   = np.clip(bar_minutes - open_min, 0, _SESSION_MINS)
                remaining = np.clip(_SESSION_MINS - elapsed, 0, _SESSION_MINS)
            except Exception:
                elapsed   = np.full(len(df), _SESSION_MINS / 2)
                remaining = np.full(len(df), _SESSION_MINS / 2)
        else:
            elapsed   = np.full(len(df), _SESSION_MINS / 2)
            remaining = np.full(len(df), _SESSION_MINS / 2)
    except Exception:
        elapsed   = np.full(len(df), _SESSION_MINS / 2)
        remaining = np.full(len(df), _SESSION_MINS / 2)

    df["minutes_since_open"] = np.array(elapsed, dtype=float) / _SESSION_MINS
    df["minutes_to_close"]   = np.array(remaining, dtype=float) / _SESSION_MINS

    return df.ffill().fillna(0)


def prepare_dataset(
    df: pd.DataFrame,
    feature_cols: list,
    scaler: Optional[MinMaxScaler] = None,
    fit_scaler: bool = True,
) -> Tuple[pd.DataFrame, MinMaxScaler]:
    """Scale feature_cols to [0,1]. Fit only on training data to prevent leakage."""
    df = df.copy()

    if "Close_raw" not in df.columns:
        df["Close_raw"] = df["Close"].values

    cols_present = [c for c in feature_cols if c in df.columns]
    df[cols_present] = df[cols_present].ffill().fillna(0)
    values = df[cols_present].values.astype(np.float32)

    if scaler is not None:
        df[cols_present] = scaler.transform(values)
        return df, scaler

    if fit_scaler:
        scaler = MinMaxScaler(feature_range=(0, 1))
        df[cols_present] = scaler.fit_transform(values)
        return df, scaler

    raise ValueError("Provide a pre-fitted scaler or set fit_scaler=True.")


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split df strictly in time order into train/val/test."""
    n = len(df)
    train_end = int(n * train_ratio)
    val_end   = int(n * (train_ratio + val_ratio))
    train = df.iloc[:train_end].copy().reset_index(drop=True)
    val   = df.iloc[train_end:val_end].copy().reset_index(drop=True)
    test  = df.iloc[val_end:].copy().reset_index(drop=True)
    return train, val, test
