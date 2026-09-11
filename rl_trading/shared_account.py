import threading
import json
import os

class SharedAccount:
    def __init__(self, initial_cash=100000.0, state_file="config/account_state.json"):
        self.state_file = state_file
        self.lock = threading.Lock()
        
        # Load existing state if it exists to preserve cash across reboots
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r") as f:
                    data = json.load(f)
                    self.cash = data.get("cash", initial_cash)
                    self.initial_cash = data.get("initial_cash", initial_cash)
            except:
                self.cash = initial_cash
                self.initial_cash = initial_cash
        else:
            self.cash = initial_cash
            self.initial_cash = initial_cash
            self._save()

    def _save(self):
        os.makedirs("config", exist_ok=True)
        with open(self.state_file, "w") as f:
            json.dump({
                "cash": self.cash,
                "initial_cash": self.initial_cash
            }, f)

    def deduct(self, amount):
        with self.lock:
            if self.cash >= amount:
                self.cash -= amount
                self._save()
                return True
            return False

    def add(self, amount):
        with self.lock:
            self.cash += amount
            self._save()
            
    def get_cash(self):
        with self.lock:
            return self.cash

    # ── Position State Persistence ─────────────────────────────────────────
    # Stores shares_held + entry metadata so that PaperTrader can restore
    # its position after a system restart.  Without this, on every restart
    # the agent thinks it holds 0 shares even though cash was already deducted,
    # causing it to re-enter the same position and double-spend capital.

    def save_position(self, shares_held: int, entry_price: float,
                      entry_time_str: str, entry_brokerage: float):
        """Persist open position to disk. Call after every BUY or SELL."""
        with self.lock:
            try:
                with open(self.state_file, "r") as f:
                    data = json.load(f)
            except Exception:
                data = {"cash": self.cash, "initial_cash": self.initial_cash}
            data["shares_held"]      = shares_held
            data["entry_price"]      = entry_price
            data["entry_time_str"]   = entry_time_str
            data["entry_brokerage"]  = entry_brokerage
            os.makedirs("config", exist_ok=True)
            with open(self.state_file, "w") as f:
                json.dump(data, f)

    def load_position(self) -> dict:
        """
        Return persisted position state, or defaults (flat) if none.
        Keys: shares_held, entry_price, entry_time_str, entry_brokerage
        """
        defaults = {
            "shares_held": 0, "entry_price": None,
            "entry_time_str": None, "entry_brokerage": 0.0
        }
        if not os.path.exists(self.state_file):
            return defaults
        try:
            with open(self.state_file, "r") as f:
                data = json.load(f)
            return {
                "shares_held":     data.get("shares_held", 0),
                "entry_price":     data.get("entry_price", None),
                "entry_time_str":  data.get("entry_time_str", None),
                "entry_brokerage": data.get("entry_brokerage", 0.0),
            }
        except Exception:
            return defaults
