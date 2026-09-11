"""
Portfolio Risk Engine.
"""

class PortfolioRiskEngine:
    def __init__(self, config=None):
        self.config = config or {}
        self.max_position_pct = self.config.get("max_position_pct", 25.0)
        
        self.portfolio_value = 0.0
        self.cash = 0.0
        self.positions = {}
        
    def update_portfolio_state(self, portfolio_value, cash, positions, peak_equity, day_start_equity):
        self.portfolio_value = portfolio_value
        self.cash = cash
        self.positions = positions
        
    def check_position_allowed(self, symbol, trade_value, portfolio_value) -> tuple[bool, str]:
        # Assume trade_value is absolute value
        max_allowed_val = portfolio_value * (self.max_position_pct / 100.0)
        
        if trade_value > max_allowed_val:
            return False, f"Trade value {trade_value:.2f} exceeds {self.max_position_pct}% of portfolio ({max_allowed_val:.2f})"
            
        return True, "Allowed"
        
    def get_risk_score(self) -> float:
        """Returns risk score 0-100 based on exposure."""
        if self.portfolio_value <= 0: return 0.0
        invested = self.portfolio_value - self.cash
        exposure = (invested / self.portfolio_value) * 100.0
        return min(100.0, max(0.0, exposure))
        
    def get_risk_summary(self) -> dict:
        return {
            "risk_score": self.get_risk_score(),
            "max_position_pct": self.max_position_pct
        }

