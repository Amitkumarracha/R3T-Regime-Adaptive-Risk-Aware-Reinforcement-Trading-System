import os
import pandas as pd
from datetime import datetime

class PaperTrader:
    """
    Simulates intraday MIS (Margin Intraday Square-off) paper trading.

    Logging Format (research-grade, one row per COMPLETED round-trip):
      Ticker, Date, EntryTime, ExitTime, HoldMinutes, Action,
      Qty, EntryPrice, ExitPrice, GrossPnL, EntryBrokerage, ExitBrokerage,
      TotalBrokerage, NetPnL, ExitReason, AccountCashAfter
    """
    def __init__(self, account, ticker, log_file="logs/trade_history.csv",
                 lot_size=1, sector="MIDCAP", portfolio_manager=None):
        self.account = account
        self.ticker = ticker
        self.lot_size = lot_size
        self.sector = sector
        self.portfolio_manager = portfolio_manager
        self.log_file = log_file

        # Restore position state from account JSON (survives restarts)
        # Without this, on every restart shares_held=0 but cash is already
        # deducted, causing the model to re-buy and double-spend capital.
        pos = account.load_position() if hasattr(account, 'load_position') else {}
        self.shares_held       = pos.get('shares_held', 0)
        self.entry_price       = pos.get('entry_price', None)
        self.entry_time        = None  # datetime — reconstructed below
        self.entry_time_str    = pos.get('entry_time_str', None)
        self.entry_brokerage   = pos.get('entry_brokerage', 0.0)

        # Reconstruct entry_time datetime if we restored an open position
        if self.entry_time_str:
            try:
                from datetime import datetime
                self.entry_time = datetime.strptime(self.entry_time_str, "%Y-%m-%d %H:%M:%S")
                print(f"[PaperTrader:{self.ticker}] RESTORED position: {self.shares_held} shares @ ₹{self.entry_price}")
            except Exception:
                self.entry_time = None

        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        if not os.path.exists(log_file):
            self._write_header()

    def _write_header(self):
        with open(self.log_file, "w") as f:
            f.write(
                "Ticker,Date,EntryTime,ExitTime,HoldMinutes,Action,"
                "Qty,EntryPrice,ExitPrice,GrossPnL,EntryBrokerage,ExitBrokerage,"
                "TotalBrokerage,NetPnL,ExitReason,AccountCashAfter\n"
            )

    def _calc_brokerage(self, trade_value):
        """Zerodha-style: min(₹20, 0.03% of trade value)."""
        return min(20.0, trade_value * 0.0003)

    def execute_trade(self, action, price, quantity, exit_reason="SIGNAL"):
        """
        action: 1 (BUY / open long), 2 (SELL / close long)
        exit_reason: 'SIGNAL', 'TRAILING_STOP', 'STOP_LOSS', 'EOD_SQUAREOFF'
        """
        now = datetime.now()
        now_str = now.strftime("%Y-%m-%d %H:%M:%S")
        trade_value = price * quantity * self.lot_size

        if action == 1:  # BUY — open a long position
            if self.shares_held > 0:
                # Already long, ignore (no pyramid)
                return False
            brokerage = self._calc_brokerage(trade_value)
            if self.account.deduct(trade_value + brokerage):
                self.shares_held = quantity
                self.entry_price = price
                self.entry_time = now
                self.entry_time_str = now_str
                self.entry_brokerage = brokerage
                # Persist position so it survives restarts
                if hasattr(self.account, 'save_position'):
                    self.account.save_position(
                        self.shares_held, self.entry_price,
                        self.entry_time_str, self.entry_brokerage
                    )
                if self.portfolio_manager:
                    self.portfolio_manager.update_usage(
                        self.ticker, self.sector, trade_value, "BUY")
                return True
            return False

        elif action == 2:  # SELL — close an existing long position
            if self.shares_held <= 0:
                return False  # Nothing to sell (short selling disabled)

            qty_to_close = min(self.shares_held, quantity)
            exit_value = price * qty_to_close * self.lot_size
            exit_brokerage = self._calc_brokerage(exit_value)
            total_brokerage = self.entry_brokerage + exit_brokerage

            gross_pnl = (price - self.entry_price) * qty_to_close * self.lot_size
            net_pnl = gross_pnl - total_brokerage

            # Return sale proceeds (net of exit brokerage) to account
            self.account.add(exit_value - exit_brokerage)

            if self.portfolio_manager:
                self.portfolio_manager.update_usage(
                    self.ticker, self.sector,
                    self.entry_price * qty_to_close * self.lot_size, "SELL")

            # Calculate hold duration
            hold_mins = round((now - self.entry_time).total_seconds() / 60, 1) if self.entry_time else 0

            # Log the COMPLETE round-trip trade
            self._log_completed_trade(
                entry_time=self.entry_time_str,
                exit_time=now_str,
                hold_mins=hold_mins,
                action="BUY→SELL",
                qty=qty_to_close,
                entry_price=self.entry_price,
                exit_price=price,
                gross_pnl=gross_pnl,
                entry_brokerage=self.entry_brokerage,
                exit_brokerage=exit_brokerage,
                total_brokerage=total_brokerage,
                net_pnl=net_pnl,
                exit_reason=exit_reason,
                cash_after=self.account.get_cash()
            )

            self.shares_held -= qty_to_close
            if self.shares_held == 0:
                self.entry_price = None
                self.entry_time = None
                self.entry_time_str = None
                self.entry_brokerage = 0.0
            # Persist cleared position so restart knows we're flat
            if hasattr(self.account, 'save_position'):
                self.account.save_position(
                    self.shares_held, self.entry_price,
                    self.entry_time_str, self.entry_brokerage
                )
            return True

        return False

    def _log_completed_trade(self, entry_time, exit_time, hold_mins, action,
                              qty, entry_price, exit_price, gross_pnl,
                              entry_brokerage, exit_brokerage, total_brokerage,
                              net_pnl, exit_reason, cash_after):
        date = entry_time.split(" ")[0] if entry_time else ""
        with open(self.log_file, "a") as f:
            f.write(
                f"{self.ticker},{date},{entry_time},{exit_time},{hold_mins},"
                f"{action},{qty},{entry_price},{exit_price},{gross_pnl:.2f},"
                f"{entry_brokerage:.2f},{exit_brokerage:.2f},{total_brokerage:.2f},"
                f"{net_pnl:.2f},{exit_reason},{cash_after:.2f}\n"
            )

    def get_portfolio_summary(self, current_price):
        return {
            "shares": self.shares_held,
            "position_value": abs(self.shares_held) * current_price * self.lot_size
        }
