"""Base executor module."""

from abc import ABC, abstractmethod
from typing import Dict, Any

class AbstractExecutor(ABC):
    @abstractmethod
    def execute(self, action: int, current_price: float, state: Dict[str, Any] = None) -> Dict[str, Any]:
        pass
        
    @abstractmethod
    def get_portfolio_state(self) -> Dict[str, Any]:
        pass
        
    @abstractmethod
    def reset(self, initial_capital: float):
        pass

