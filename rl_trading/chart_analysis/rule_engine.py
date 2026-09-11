import pandas as pd
from typing import Dict, Any
from .base_analyzer import BaseChartAnalyzer

class EMAChartAnalyzer(BaseChartAnalyzer):
    """
    Implements the EMA 5 / EMA 200 crossover system with SMA 50 stop loss
    and a profit target, as extracted from Testing_hdfc.ipynb.
    
    Can be configured for different timeframes by passing differently 
    sampled DataFrames to the analyze method, though the underlying
    rules remain the same. Ensure the dataframe has EMA_5, EMA_200, SMA_50
    """
    
    def __init__(self, profit_target_pct: float = 0.03):
        self.profit_target_pct = profit_target_pct

    def analyze(self, df: pd.DataFrame, position_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Returns a signal dict based on the last row of the DataFrame.
        Assume the last row is the current unclosed or just-closed candle.
        """
        if len(df) < 2:
            return {"signal": "NEUTRAL", "confidence": 0.0, "reasons": ["Not enough data"]}

        # Check required columns
        required_cols = ["Close", "EMA_5", "EMA_200", "SMA_50"]
        if not all(col in df.columns for col in required_cols):
             return {"signal": "NEUTRAL", "confidence": 0.0, "reasons": [f"Missing required columns: {required_cols}"]}

        # Un-normalize if we passed the normalized df, we need raw close (assuming we pass un-normalized df with indicators)
        # Actually in production, our features.py keeps Close_raw. Let's just use Close or Close_raw
        price_col = "Close_raw" if "Close_raw" in df.columns else "Close"

        row = df.iloc[-1]
        prev = df.iloc[-2]
        
        price = row[price_col]
        position_qty = position_state.get("qty", 0)
        entry_price = position_state.get("entry_price", None)

        # ── BUY LOGIC ──────────────────────────────────────────────────
        # Require 2-candle confirmation: EMA5 just crossed above EMA200,
        # AND RSI > 55 confirms real upward momentum (not a dead-cat bounce).
        rsi = row.get("RSI", 50)
        two_candle_confirm = (
            prev["EMA_5"] <= prev["EMA_200"] and  # Previous: still below
            row["EMA_5"] > row["EMA_200"]          # Current: just crossed above
        )
        buy_signal = (
            two_candle_confirm and
            row["EMA_5"] > row["SMA_50"] and       # Overall trend is up
            rsi > 55 and                            # Momentum confirmation
            position_qty == 0                       # Not already in position
        )

        if buy_signal:
            return {
                "signal": "BUY",
                "confidence": 1.0,
                "reasons": [f"EMA5 crossed ABOVE EMA200 (RSI={rsi:.1f})", "EMA5 > SMA50 trend confirmed"]
            }

        # ---------------- SELL LOGIC ----------------
        if position_qty > 0 and entry_price is not None:
            sell_reasons = []

            # 1. Trend Reversal
            if prev["EMA_5"] >= prev["EMA_200"] and row["EMA_5"] < row["EMA_200"]:
                sell_reasons.append("EMA5 crossed BELOW EMA200")

            # 2. Stop Loss (EMA5 nearly touching SMA50)
            elif abs(row["EMA_5"] - row["SMA_50"]) / row["SMA_50"] < 0.0005: # 0.05%
                sell_reasons.append("EMA5 touched SMA50 (STOP LOSS)")

            # 3. Profit Target
            elif (price - entry_price) / entry_price >= self.profit_target_pct:
                sell_reasons.append("PROFIT TARGET HIT")

            if sell_reasons:
                return {
                    "signal": "SELL",
                    "confidence": 1.0,
                    "reasons": sell_reasons
                }

        # ---------------- DEFAULT ----------------
        return {"signal": "NEUTRAL", "confidence": 0.0, "reasons": []}
