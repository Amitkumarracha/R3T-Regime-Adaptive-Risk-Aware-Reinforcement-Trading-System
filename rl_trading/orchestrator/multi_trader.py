import os
import time
import threading
import argparse
import pandas as pd
from datetime import datetime
import json
import pytz
from concurrent.futures import ThreadPoolExecutor

from rl_trading.universe.stocks import StockUniverse
from rl_trading.data.data_manager import DataManager
from rl_trading.live_agent import LiveAgent
from rl_trading.shared_account import SharedAccount

from .portfolio_manager import PortfolioManager

IST = pytz.timezone('Asia/Kolkata')
EOD_SQUAREOFF_TIME = (15, 20)   # 3:20 PM — force-sell all positions
MARKET_OPEN_TIME   = (9, 15)    # 9:15 AM — market open


class MultiTrader:
    def __init__(self, tickers: list, account, data_manager):
        self.tickers = tickers
        self.account = account
        self.data_manager = data_manager
        self.portfolio_manager = PortfolioManager(total_capital=account.cash)
        self.agents = {}
        self._squared_off_today = False  # Prevent double square-off
        self._last_squareoff_date = None
        self._init_agents()
        self.data_manager.register_callback(self._on_tick)
        self.start_time = time.time()

    def _init_agents(self):
        from rl_trading.shared_account import SharedAccount
        for stock in self.tickers:
            ticker = stock['ticker']
            scrip_code = stock['scrip_code']
            index_name = stock.get('index')
            lot_size = stock.get('lot_size', 1)
            sector = stock.get('sector', 'MIDCAP')
            model_path = f"rl_trading/models/dqn_{ticker}.pt"

            # Each agent has its own virtual account that PERSISTS across days.
            # P&L is carried forward naturally — no manual reset needed.
            agent_account = SharedAccount(
                initial_cash=100000.0,
                state_file=f"config/account_{ticker}.json"
            )

            self.agents[ticker] = LiveAgent(
                ticker, scrip_code, model_path, agent_account,
                index_name=index_name, data_manager=self.data_manager,
                lot_size=lot_size, sector=sector, portfolio_manager=None
            )

    def _is_paused(self):
        state_file = "config/trading_state.json"
        if os.path.exists(state_file):
            with open(state_file, "r") as f:
                try:
                    return json.load(f).get("paused", False)
                except:
                    pass
        return False

    def _on_tick(self, ticker: str, latest_df: pd.DataFrame):
        """Called automatically by DataManager on each candle close."""
        if self._is_paused():
            return

        dm = self.data_manager
        is_live = dm.use_live and dm.paisa is not None and dm.paisa.is_connected
        if not is_live:
            return

        if ticker in self.agents:
            self.agents[ticker].run_tick(latest_df)

    # ── EOD Square-Off ────────────────────────────────────────────────────
    def square_off_all(self, reason="EOD_SQUAREOFF"):
        """Force-sell all open positions across all agents."""
        now_ist = datetime.now(IST)
        today = now_ist.date()

        if self._last_squareoff_date == today:
            return  # Already squared off today

        print(f"\n{'='*55}")
        print(f"[EOD SQUARE-OFF] Initiated at {now_ist.strftime('%H:%M:%S IST')}")
        print(f"{'='*55}")

        total_net_pnl = 0.0
        for ticker, agent in self.agents.items():
            if agent.trader.shares_held > 0:
                # Get current price from the buffer
                buf = self.data_manager.buffers.get(ticker)
                if buf is not None and not buf.empty:
                    current_price = float(buf.iloc[-1].get('Close_raw', buf.iloc[-1]['Close']))
                    qty = agent.trader.shares_held
                    entry_price = agent.trader.entry_price or current_price
                    gross_pnl = (current_price - entry_price) * qty

                    success = agent.trader.execute_trade(2, current_price, qty, exit_reason=reason)
                    if success:
                        cash = agent.account.get_cash()
                        total_net_pnl += gross_pnl
                        print(f"  [SQ-OFF] {ticker:15} | SOLD {qty:5} @ ₹{current_price:.2f} | Gross PNL: ₹{gross_pnl:+.2f} | Cash: ₹{cash:,.0f}")
                        # Update SafeNet state
                        agent.safenet.update_position(None)
            else:
                pass  # No open position, skip

        print(f"\n  Total Gross PNL from square-off: ₹{total_net_pnl:+.2f}")
        print(f"  All positions closed. Capital carried forward to tomorrow.")
        print(f"{'='*55}\n")

        self._last_squareoff_date = today

        # Send Telegram notification
        try:
            from rl_trading.utils.notifications import notifier
            notifier.send_message(
                f"🔔 *EOD Square-Off Complete*\n"
                f"All positions closed at 3:20 PM IST.\n"
                f"Gross PNL: ₹{total_net_pnl:+.2f}\n"
                f"Capital carried forward to tomorrow."
            )
        except:
            pass

    def _eod_monitor(self):
        """Background thread: triggers square-off at 3:20 PM IST every trading day."""
        print("[EOD Monitor] Started. Will square-off at 15:20 IST daily.")
        while True:
            now_ist = datetime.now(IST)
            h, m = now_ist.hour, now_ist.minute
            today = now_ist.date()

            # Trigger square-off at exactly 15:20–15:21 IST
            if h == EOD_SQUAREOFF_TIME[0] and m >= EOD_SQUAREOFF_TIME[1] and m < EOD_SQUAREOFF_TIME[1] + 5:
                if self._last_squareoff_date != today:
                    self.square_off_all(reason="EOD_SQUAREOFF")

            time.sleep(30)  # Check every 30 seconds

    def start(self):
        print("====== Laplace Multi-Trader Started ======")
        self.data_manager.load_initial_history()

        # Start the EOD monitor in a background thread
        eod_thread = threading.Thread(target=self._eod_monitor, daemon=True)
        eod_thread.start()
        print("[EOD Monitor] Background thread launched.")

        # Blocking call — streams live data and drives all agent decisions
        self.data_manager.start_streaming()
        # NOTE: start() blocks here until DataManager exits (market close / error).


if __name__ == "__main__":
    # MultiTrader is intended to be launched via app.py (Flask orchestrator).
    # Direct invocation requires a DataManager + SharedAccount, e.g.:
    #   from rl_trading.data.data_manager import DataManager
    #   from rl_trading.shared_account import SharedAccount
    #   from rl_trading.universe.stocks import StockUniverse
    #   dm  = DataManager()
    #   acc = SharedAccount(initial_cash=100000.0)
    #   stocks = StockUniverse().get_all_stocks()
    #   MultiTrader(tickers=stocks, account=acc, data_manager=dm).start()
    print("Use app.py to launch the full trading system.")
