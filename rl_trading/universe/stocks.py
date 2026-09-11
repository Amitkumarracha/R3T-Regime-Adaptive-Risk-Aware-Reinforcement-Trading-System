import os
import yaml
from typing import List, Dict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UNIVERSE_FILE = os.path.join(PROJECT_ROOT, "config", "universe.yaml")

class StockUniverse:
    def __init__(self, config_path: str = UNIVERSE_FILE):
        self.config_path = config_path
        self._load()

    def _load(self):
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Universe config missing: {self.config_path}")
        with open(self.config_path, "r") as f:
            self.data = yaml.safe_load(f)

    def get_all_stocks(self) -> List[Dict]:
        """Returns flat list of all configuration dicts for every stock."""
        return self.data.get('stocks', [])

    def get_indices(self) -> List[Dict]:
        """Returns list of tracking indices."""
        return self.data.get('indices', [])

    def get_stock(self, ticker: str) -> Dict:
        """Get details for a single ticker"""
        for stock in self.get_all_stocks():
            if stock['ticker'] == ticker:
                return stock
        raise ValueError(f"Ticker {ticker} not found in universe")

    def get_all_tickers(self) -> List[str]:
        return [s['ticker'] for s in self.get_all_stocks()]

    def get_all_yahoo_tickers(self) -> List[str]:
        return [s['yahoo'] for s in self.get_all_stocks()]

if __name__ == "__main__":
    universe = StockUniverse()
    print(f"Loaded {len(universe.get_all_stocks())} stocks from universe.")
    print(f"Tracking {len(universe.get_indices())} indices.")
    print("Tickers:", universe.get_all_tickers())
