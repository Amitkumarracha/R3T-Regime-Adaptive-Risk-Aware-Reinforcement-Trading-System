from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any

class BaseChartAnalyzer(ABC):
    """
    Abstract interface for pluggable rule-based chart analysis.
    """

    @abstractmethod
    def analyze(self, df: pd.DataFrame, position_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the current price chart and return a signal.

        Args:
            df (pd.DataFrame): DataFrame containing recent historical candles
                               up to the current tick. Must have the required
                               features calculated (like EMA_5).
            position_state (Dict): Info about current position, e.g.,
                                   {"qty": 0, "entry_price": None}

        Returns:
            Dict: A signal dict:
                {
                    "signal": "BUY" | "SELL" | "NEUTRAL",
                    "confidence": float (0.0 to 1.0),
                    "reasons": list of strings
                }
        """
        pass
