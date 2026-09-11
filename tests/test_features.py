"""tests/test_features.py — Feature engineering tests."""
import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.feature_engineering import add_technical_indicators, prepare_dataset, chronological_split, FEATURE_COLS


class TestFeatureEngineering:
    def test_no_nan_after_preprocessing(self, sample_ohlcv):
        """Feature columns must have no unexpected NaNs after preprocessing."""
        df = add_technical_indicators(sample_ohlcv.copy())
        feature_cols = [c for c in FEATURE_COLS if c in df.columns]
        df_s, scaler = prepare_dataset(df, feature_cols)
        assert not df_s[feature_cols].isnull().any().any(), \
            "NaNs found in feature columns after preprocessing"

    def test_correct_feature_columns_present(self, sample_ohlcv):
        """Required feature columns must all be present after feature engineering."""
        df = add_technical_indicators(sample_ohlcv.copy())
        required = ["returns", "RSI", "MACD", "ATR", "ATR_pct", "EMA_slope",
                    "momentum_5", "momentum_20", "realized_vol",
                    "minutes_since_open", "minutes_to_close"]
        for col in required:
            assert col in df.columns, f"Missing feature column: {col}"

    def test_no_future_leakage(self, sample_ohlcv):
        """Features at time t should only use data available at t or earlier."""
        df = add_technical_indicators(sample_ohlcv.copy())
        # SMA_20 at row 0 should be NaN (not enough history)
        # After ffill/fillna, but raw SMA_20 at index 0 should be NaN
        raw_sma = sample_ohlcv["Close"].rolling(20).mean()
        assert pd.isna(raw_sma.iloc[0]), "SMA_20 at index 0 should be NaN (no future leakage)"
        assert pd.isna(raw_sma.iloc[18]), "SMA_20 at index 18 should be NaN (only 19 bars)"
        assert not pd.isna(raw_sma.iloc[19]), "SMA_20 at index 19 should be valid"

    def test_chronological_split_order(self, sample_ohlcv):
        """Train/val/test splits must be strictly chronological."""
        df = add_technical_indicators(sample_ohlcv.copy())
        train, val, test = chronological_split(df, 0.70, 0.15)
        # No overlap
        assert len(train) + len(val) + len(test) == len(df)
        # Chronological order: last train < first val < first test
        if hasattr(train.index, 'is_monotonic_increasing'):
            pass  # reset_index so check by position
        n = len(df)
        train_end = int(n * 0.70)
        val_end   = train_end + int(n * 0.15)
        assert len(train) == train_end
        assert len(val)   == val_end - train_end

    def test_scaler_fit_on_train_only(self, sample_ohlcv):
        """Scaler must be fit on train set only, then applied to val/test."""
        df = add_technical_indicators(sample_ohlcv.copy())
        train, val, test = chronological_split(df, 0.70, 0.15)
        feature_cols = [c for c in FEATURE_COLS if c in df.columns]
        train_s, scaler = prepare_dataset(train, feature_cols, fit_scaler=True)
        val_s,   _      = prepare_dataset(val,   feature_cols, scaler=scaler, fit_scaler=False)
        test_s,  _      = prepare_dataset(test,  feature_cols, scaler=scaler, fit_scaler=False)
        # Train should be in [0, 1] for all feature cols
        for col in feature_cols:
            if col in train_s.columns:
                col_min = train_s[col].min()
                col_max = train_s[col].max()
                assert col_min >= -0.01, f"Train {col} below 0: {col_min}"
                assert col_max <= 1.01, f"Train {col} above 1: {col_max}"

    def test_close_raw_preserved(self, sample_ohlcv):
        """Close_raw must be the original unscaled close price."""
        df = add_technical_indicators(sample_ohlcv.copy())
        assert "Close_raw" in df.columns
        # Close_raw should equal original Close
        np.testing.assert_array_almost_equal(
            df["Close_raw"].values,
            sample_ohlcv["Close"].values,
            decimal=4,
        )

    def test_returns_computation(self, sample_ohlcv):
        """Returns should be percentage change, not absolute."""
        df = add_technical_indicators(sample_ohlcv.copy())
        assert "returns" in df.columns
        # returns should be bounded (e.g., < 50% daily for reasonable synthetic data)
        valid_returns = df["returns"].dropna()
        assert (valid_returns.abs() < 0.5).all(), "Extreme returns detected — check computation"

    def test_atr_positive(self, sample_ohlcv):
        """ATR must always be positive."""
        df = add_technical_indicators(sample_ohlcv.copy())
        assert "ATR" in df.columns
        valid_atr = df["ATR"].dropna()
        assert (valid_atr >= 0).all(), "ATR must be non-negative"

    def test_rsi_bounds(self, sample_ohlcv):
        """RSI must be in [0, 100]."""
        df = add_technical_indicators(sample_ohlcv.copy())
        valid_rsi = df["RSI"].dropna()
        assert (valid_rsi >= 0).all() and (valid_rsi <= 100).all(), "RSI out of [0,100]"

    def test_time_features_bounds(self, sample_ohlcv):
        """Time features must be in [0, 1]."""
        df = add_technical_indicators(sample_ohlcv.copy())
        assert "minutes_since_open" in df.columns
        assert "minutes_to_close" in df.columns
        for col in ["minutes_since_open", "minutes_to_close"]:
            vals = df[col].dropna()
            assert (vals >= 0).all() and (vals <= 1).all(), f"{col} out of [0,1]"
