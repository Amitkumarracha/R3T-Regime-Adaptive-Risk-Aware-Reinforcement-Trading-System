# utils/features.py

import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, MACD, EMAIndicator
from ta.momentum import RSIIndicator
from ta.volume import OnBalanceVolumeIndicator
from sklearn.preprocessing import MinMaxScaler

FEATURE_COLS = [
    'Close', 'Volume',
    'SMA_20', 'SMA_50',
    'EMA_5', 'EMA_200',
    'RSI',
    'MACD', 'MACD_signal',
    'OBV',
    # Time-of-day features — critical for intraday models.
    # Without these the model can't distinguish 9:30AM from 3:00PM,
    # so it enters positions late in the day and gets EOD-squaredoff at a loss.
    'minutes_since_open',  # 0 at 9:15 AM → 1.0 at 3:15 PM (375 min session)
    'minutes_to_close',    # 1.0 at 9:15 AM → 0 at 3:15 PM
]

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure columns are flat if they came from yfinance/multiindex before
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # We don't dropna yet to avoid wiping the buffer if EMA_200 is missing history
    df = df.copy()
    close  = df['Close'].squeeze()
    volume = df['Volume'].squeeze()

    df['SMA_20']      = SMAIndicator(close=close, window=20).sma_indicator()
    df['SMA_50']      = SMAIndicator(close=close, window=50).sma_indicator()
    df['EMA_5']       = EMAIndicator(close=close, window=5).ema_indicator()
    df['EMA_200']     = EMAIndicator(close=close, window=200).ema_indicator()
    df['RSI']         = RSIIndicator(close=close, window=14).rsi()

    macd              = MACD(close=close)
    df['MACD']        = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    df['OBV']         = OnBalanceVolumeIndicator(
                            close=close, volume=volume
                        ).on_balance_volume()

    # ── Time-of-day features ───────────────────────────────────────
    # NSE session: 9:15 AM – 3:30 PM = 375 minutes
    SESSION_MINS = 375.0
    MARKET_OPEN_H, MARKET_OPEN_M = 9, 15

    idx = df.index
    # Handle both tz-aware and naive indices
    try:
        if hasattr(idx, 'tz') and idx.tz is not None:
            import pytz
            idx_ist = idx.tz_convert(pytz.timezone('Asia/Kolkata'))
        else:
            idx_ist = idx
        mins_since_open = (
            (idx_ist.hour - MARKET_OPEN_H) * 60
            + (idx_ist.minute - MARKET_OPEN_M)
        ).clip(0, SESSION_MINS)
    except Exception:
        # Fallback for DataFrames with non-datetime index (e.g. integer from reset_index)
        mins_since_open = np.zeros(len(df))

    df['minutes_since_open'] = np.array(mins_since_open, dtype=float) / SESSION_MINS
    df['minutes_to_close']   = 1.0 - df['minutes_since_open']

    return df

def prepare_env_dataframe(df: pd.DataFrame, feature_cols: list):
    # Save raw close price BEFORE normalizing (needed for trade execution)
    df = df.copy()
    df['Close_raw'] = df['Close'].values

    # Fill NaNs with 0 or forward-fill so the DQN doesn't crash
    # But ONLY for the columns in feature_cols
    df[feature_cols] = df[feature_cols].ffill().fillna(0)

    scaler = MinMaxScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    return df, scaler

def train_test_split_timeseries(df: pd.DataFrame, train_ratio: float = 0.8):
    split_idx = int(len(df) * train_ratio)
    train_df  = df.iloc[:split_idx].copy().reset_index(drop=True)
    test_df   = df.iloc[split_idx:].copy().reset_index(drop=True)
    return train_df, test_df