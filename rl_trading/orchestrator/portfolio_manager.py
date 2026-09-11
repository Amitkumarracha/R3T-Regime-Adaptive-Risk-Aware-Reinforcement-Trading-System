class PortfolioManager:
    def __init__(self, total_capital=100000.0):
        self.total_capital = total_capital
        self.current_usage = 0.0

    def can_buy(self, ticker, sector, amount):
        # We allow a single stock to take up to 100% of the capital
        if self.current_usage + amount <= self.total_capital:
            return True
        return False

    def update_usage(self, ticker, sector, amount, action="BUY"):
        if action == "BUY":
            self.current_usage += amount
        else:
            self.current_usage = max(0.0, self.current_usage - amount)

    def get_summary(self):
        return {
            "total": self.total_capital,
            "usage": self.current_usage,
            "available": self.total_capital - self.current_usage
        }
