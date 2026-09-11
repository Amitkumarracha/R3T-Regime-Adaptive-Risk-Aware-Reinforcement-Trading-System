"""
ATR-based Position Sizing module.
"""

class PositionSizer:
    def __init__(self, config=None):
        self.config = config or {}
        self.risk_per_trade_pct = self.config.get("risk_per_trade_pct", 1.0)
        
    def calculate_position_size(self, portfolio_value, current_price, atr, atr_multiplier=3.0) -> int:
        if atr <= 0 or current_price <= 0:
            return 0
            
        risk_amount = portfolio_value * (self.risk_per_trade_pct / 100.0)
        stop_loss_dist = atr * atr_multiplier
        
        shares = int(risk_amount / stop_loss_dist)
        
        # Sanity check vs capital
        max_shares = int(portfolio_value / current_price)
        return min(shares, max_shares)

