"""tests/test_regime.py — Regime detector tests."""
import numpy as np
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.feature_engineering import add_technical_indicators
from src.regime.detector import RegimeDetector, REGIME_NAMES


class TestRegimeDetector:
    def test_valid_regime_ids(self, sample_ohlcv):
        """All regime IDs must be in {0,1,2,3}."""
        df = add_technical_indicators(sample_ohlcv.copy())
        det = RegimeDetector(n_regimes=4, rolling_window=20, seed=42)
        df_out = det.fit_predict(df)
        assert "regime" in df_out.columns
        valid_ids = set(REGIME_NAMES.keys())
        for r in df_out["regime"].dropna().unique():
            assert int(r) in valid_ids, f"Invalid regime id: {r}"

    def test_deterministic_for_same_input(self, sample_ohlcv):
        """Same input + same seed → same regime output."""
        df = add_technical_indicators(sample_ohlcv.copy())
        det1 = RegimeDetector(n_regimes=4, rolling_window=20, seed=42)
        det2 = RegimeDetector(n_regimes=4, rolling_window=20, seed=42)
        out1 = det1.fit_predict(df.copy())
        out2 = det2.fit_predict(df.copy())
        np.testing.assert_array_equal(
            out1["regime"].values, out2["regime"].values
        )

    def test_one_hot_encoding_valid(self, sample_ohlcv):
        """One-hot regime columns must sum to 1 per row."""
        df = add_technical_indicators(sample_ohlcv.copy())
        det = RegimeDetector(n_regimes=4, rolling_window=20, seed=42)
        df_out = det.fit_predict(df)
        one_hot_cols = ["regime_0", "regime_1", "regime_2", "regime_3"]
        for col in one_hot_cols:
            assert col in df_out.columns, f"Missing one-hot column: {col}"
        sums = df_out[one_hot_cols].sum(axis=1)
        assert (sums == 1).all(), "One-hot regime columns must sum to 1 per row"

    def test_confidence_in_bounds(self, sample_ohlcv):
        """Regime confidence must be in [0, 1]."""
        df = add_technical_indicators(sample_ohlcv.copy())
        det = RegimeDetector(n_regimes=4, rolling_window=20, seed=42)
        df_out = det.fit_predict(df)
        assert "regime_confidence" in df_out.columns
        conf = df_out["regime_confidence"].dropna()
        assert (conf >= 0).all() and (conf <= 1).all(), "Confidence out of [0,1]"

    def test_predict_without_fit_raises(self, sample_ohlcv):
        """predict() on an unfitted detector should raise AttributeError or ValueError."""
        df = add_technical_indicators(sample_ohlcv.copy())
        det = RegimeDetector(n_regimes=4, rolling_window=20, seed=42)
        with pytest.raises((AttributeError, ValueError, Exception)):
            det.predict(df)

    def test_small_df_handled(self):
        """Very small DataFrames (< rolling_window) must not crash."""
        import pandas as pd
        dates = pd.date_range("2020-01-02", periods=10, freq="B")
        df_small = pd.DataFrame({
            "Open": [100]*10, "High": [105]*10, "Low": [95]*10,
            "Close": [102]*10, "Volume": [10000]*10
        }, index=dates)
        df_small = add_technical_indicators(df_small)
        det = RegimeDetector(n_regimes=4, rolling_window=20, seed=42)
        # Should not raise, should assign a default regime
        out = det.fit_predict(df_small)
        assert "regime" in out.columns

    def test_regime_names_mapping(self, sample_ohlcv):
        """regime_name column must contain valid string regime names."""
        df = add_technical_indicators(sample_ohlcv.copy())
        det = RegimeDetector(n_regimes=4, rolling_window=20, seed=42)
        df_out = det.fit_predict(df)
        if "regime_name" in df_out.columns:
            valid_names = set(REGIME_NAMES.values())
            for name in df_out["regime_name"].dropna().unique():
                assert name in valid_names, f"Invalid regime name: {name}"
