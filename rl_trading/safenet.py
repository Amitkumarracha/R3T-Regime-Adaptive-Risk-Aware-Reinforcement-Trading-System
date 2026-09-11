class SafeNet:
    """
    Quant-grade risk management for Laplace agents.

    Exit Strategy:
      - Hard Stop Loss  : Exit if position falls -2% from entry (capital protection).
      - Trailing Stop   : Once in profit, trail the stop to lock in gains:
                            >= 1% peak gain → stop at breakeven (+0.1%)
                            >= 2% peak gain → stop at +1% from entry
                            >= 3% peak gain → stop at +2% from entry
                            (i.e., always trail 1% below the peak gain reached)
      - Daily Drawdown  : Emergency exit if portfolio falls -3% from day start.
      - No hard take-profit cap — let winners run.
    """
    def __init__(self, stop_loss_pct=0.02, max_daily_loss_pct=0.03, ticker="UNKNOWN"):
        self.stop_loss_pct = stop_loss_pct
        self.max_daily_loss_pct = max_daily_loss_pct
        self.ticker = ticker
        
        self.position_entry_price = None
        self.daily_start_value = None
        self._peak_price = None  # Highest price seen since entry (for trailing stop)

    def update_position(self, entry_price):
        """Call this when a new position is opened."""
        self.position_entry_price = entry_price
        self._peak_price = entry_price

    def set_daily_start(self, value):
        self.daily_start_value = value

    def check_rules(self, current_price, current_portfolio_value, shares_held):
        """
        Returns: None (no action), or 2 (forced SELL/exit).
        """
        if shares_held == 0:
            self.position_entry_price = None
            self._peak_price = None
            return None

        if self.position_entry_price is None:
            return None

        # Track peak price for trailing stop
        if self._peak_price is None:
            self._peak_price = current_price
        else:
            self._peak_price = max(self._peak_price, current_price)

        entry = self.position_entry_price
        peak  = self._peak_price

        # ── 1. Hard Stop Loss ─────────────────────────────────────────
        loss_from_entry = (current_price - entry) / entry
        if loss_from_entry <= -self.stop_loss_pct:
            print(f"!!! [{self.ticker}] SAFENET: STOP LOSS at ₹{current_price:,.2f} ({loss_from_entry*100:.2f}%)")
            return 2

        # ── 2. Trailing Stop ──────────────────────────────────────────
        peak_gain = (peak - entry) / entry  # How far up have we ever been?

        if peak_gain >= 0.01:  # We're at least +1% at peak — activate trailing
            # Calculate brokerage-aware minimum floor.
            # We pay ~₹20 to enter and ~₹20 to exit = ₹40 total.
            # Add a ₹20 buffer → floor must cover at least ₹60 gross profit.
            invested_amount = entry * abs(shares_held)
            min_gross_to_breakeven = 60.0  # ₹40 fees + ₹20 buffer
            brokerage_floor_pct = min_gross_to_breakeven / invested_amount if invested_amount > 0 else 0.001

            # Trail 1% below the peak gain reached, but never below brokerage breakeven
            trail_floor = max(brokerage_floor_pct, peak_gain - 0.01)
            current_gain = (current_price - entry) / entry

            if current_gain <= trail_floor:
                locked_in = trail_floor * invested_amount - 60.0  # Net after fees
                print(f"!!! [{self.ticker}] SAFENET: TRAILING STOP at ₹{current_price:,.2f} "
                      f"| peak={peak_gain*100:.2f}% | net_locked=₹{locked_in:.0f}")
                return 2

        # ── 3. Daily Drawdown ─────────────────────────────────────────
        if self.daily_start_value:
            daily_perf = (current_portfolio_value - self.daily_start_value) / self.daily_start_value
            if daily_perf <= -self.max_daily_loss_pct:
                print(f"!!! [{self.ticker}] SAFENET: MAX DAILY LOSS ({daily_perf*100:.2f}%)")
                return 2

        return None
