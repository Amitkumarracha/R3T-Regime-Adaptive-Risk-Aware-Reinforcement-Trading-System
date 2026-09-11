"""
Daily Circuit Breaker.
"""

class DailyCircuitBreaker:
    def __init__(self, config=None):
        self.config = config or {}
        self.max_daily_loss_pct = self.config.get("max_daily_loss_pct", 1.5)
        self.max_drawdown_pct = self.config.get("max_drawdown_pct", 5.0)
        
        self.day_start_equity = 100000.0
        self.peak_equity = 100000.0
        self.is_active = False
        
    def reset(self):
        self.is_active = False
        
    def set_day_start(self, equity):
        self.day_start_equity = equity
        if self.peak_equity < equity:
            self.peak_equity = equity
            
    def update_peak(self, equity):
        if equity > self.peak_equity:
            self.peak_equity = equity
            
    def check(self, current_equity) -> tuple[bool, str]:
        if self.is_active:
            return False, "Circuit breaker is active"
            
        daily_loss_pct = ((self.day_start_equity - current_equity) / self.day_start_equity) * 100.0
        if daily_loss_pct > self.max_daily_loss_pct:
            self.is_active = True
            return False, f"Daily loss ({daily_loss_pct:.2f}%) exceeded limit ({self.max_daily_loss_pct}%)"
            
        drawdown_pct = ((self.peak_equity - current_equity) / self.peak_equity) * 100.0
        if drawdown_pct > self.max_drawdown_pct:
            self.is_active = True
            return False, f"Max drawdown ({drawdown_pct:.2f}%) exceeded limit ({self.max_drawdown_pct}%)"
            
        return True, "OK"
        
    def get_status(self) -> dict:
        return {
            "blocked": self.is_active
        }

