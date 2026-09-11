# rl_trading/live_agent.py

import os
import time
import traceback
from datetime import datetime
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

from .agents.dqn_agent import DQNAgent
from .paper_trader import PaperTrader
from .safenet import SafeNet
from .chart_analysis.rule_engine import EMAChartAnalyzer
from .utils.features import FEATURE_COLS, prepare_env_dataframe

class LiveAgent:
    def __init__(self, ticker: str, scrip_code: int, model_path: str, account, index_name: str = None, data_manager = None, lot_size: int = 1, sector: str = "MIDCAP", portfolio_manager = None):
        self.ticker = ticker
        self.scrip_code = scrip_code
        self.account = account
        self.index_name = index_name
        self.data_manager = data_manager
        self.lot_size = lot_size
        self.sector = sector
        self.portfolio_manager = portfolio_manager
        
        # Load Operational Mode from Environment
        self.use_rule_only = os.getenv("USE_RULE_BASED_ONLY", "false").lower() == "true"
        
        # 1. DQN Agent
        obs_dim = len(FEATURE_COLS) + 2
        self.dqn = DQNAgent(obs_dim=obs_dim)
        if os.path.exists(model_path):
            try:
                self.dqn.load(model_path)
                print(f"[Agent] {ticker} loaded DQN model.")
            except Exception as e:
                print(f"[Agent] Failed to load {ticker} model (likely size mismatch). Starting with random weights. Error: {e}")
        self.dqn.epsilon = 0.0
        
        # 2. Support Systems
        today_str = datetime.now().strftime("%Y-%m-%d")
        log_file = f"logs/trades_{self.ticker}_{today_str}.csv"
        self.trader = PaperTrader(account=self.account, ticker=self.ticker, log_file=log_file, lot_size=self.lot_size, sector=self.sector, portfolio_manager=self.portfolio_manager)
        self.safenet = SafeNet(stop_loss_pct=0.02, max_daily_loss_pct=0.03, ticker=self.ticker)
        self.chart_analyzer = EMAChartAnalyzer(profit_target_pct=0.03)
        
        self.last_action_str = "HOLD"
        self.last_action_time = None
        self.last_chart_signal = "NEUTRAL"
        self.last_chart_reasons = []
        self._tick_count = 0
        self._last_decision_log = 0

        # ── Overtrading Prevention ──────────────────────────────────────
        # Cap: maximum round-trip trades per stock per session.
        # Root cause of the 2026-05-26 event: TCS executed 1,910 trades,
        # paying ₹38,200 in brokerage against only ₹949 gross profit.
        self.MAX_DAILY_TRADES = int(os.getenv("MAX_DAILY_TRADES", "15"))
        # Cooldown: minimum candles to hold before allowing exit/re-entry.
        # At 5-min bars this = 15 min minimum hold.
        self.MIN_HOLD_CANDLES = int(os.getenv("MIN_HOLD_CANDLES", "3"))

        self._daily_trade_count = 0   # round-trips completed today
        self._last_trade_tick  = -999  # tick index of the last executed trade
        self._last_trade_date  = None  # YYYY-MM-DD string, for daily reset

    def run_tick(self, recent_data: pd.DataFrame):
        """Processes the latest candle/tick and makes a trading decision."""
        self._tick_count += 1

        # ── Daily reset (new trading day) ──────────────────────────────
        today_str = datetime.now().strftime("%Y-%m-%d")
        if self._last_trade_date != today_str:
            self._daily_trade_count = 0
            self._last_trade_tick   = -999
            self._last_trade_date   = today_str

        if recent_data is None or recent_data.empty:
            if self._tick_count <= 5:
                print(f"[Agent:{self.ticker}] SKIP: data is None/empty")
            return
            
        if len(recent_data) < 200:
            if self._tick_count <= 5:
                print(f"[Agent:{self.ticker}] SKIP: only {len(recent_data)} bars (need 200)")
            return

        # 1. Feature Preparation (Scaling Fix)
        try:
            scaled_df, _ = prepare_env_dataframe(recent_data, FEATURE_COLS)
            latest_row = scaled_df.iloc[-1]
            raw_row = recent_data.iloc[-1]
        except Exception as e:
            print(f"[Agent:{self.ticker}] Scaling Error: {e}")
            traceback.print_exc()
            return

        current_price = raw_row.get('Close_raw', raw_row['Close'])
        
        # 2. Decision Logic
        action_names = {0: "HOLD", 1: "BUY", 2: "SELL"}
        dqn_action = 0
        
        # A. AI Analysis (DQN)
        if not self.use_rule_only:
            market_obs = latest_row[FEATURE_COLS].values.astype(np.float32)
            current_cash = self.account.get_cash()
            cash_ratio = np.clip(current_cash / self.account.initial_cash, 0, 1)
            shares_ratio = np.clip(abs(self.trader.shares_held) / 100, 0, 1)
            state = np.append(market_obs, [cash_ratio, shares_ratio]).astype(np.float32)
            dqn_action = self.dqn.select_action(state)
        
        final_action = dqn_action

        # B. Rule-Based/Safety Check
        chart_result = self.chart_analyzer.analyze(recent_data, {"qty": self.trader.shares_held, "entry_price": self.safenet.position_entry_price})
        self.last_chart_signal = chart_result["signal"]
        self.last_chart_reasons = chart_result["reasons"]
        
        if chart_result["signal"] == "SELL":
            if self.trader.shares_held > 0:
                final_action = 2 
        elif self.use_rule_only:
            if chart_result["signal"] == "BUY":
                final_action = 1
            elif chart_result["signal"] == "SELL":
                final_action = 2
        elif dqn_action == 1:
            if chart_result["signal"] == "SELL":
                final_action = 0

        # B.5. Profitability Check (Anti-Whiplash Guard)
        # Only allow closing a position if the gross profit exceeds 1% of the invested amount 
        # (or ₹50, whichever is higher to cover brokerage).
        # If it doesn't, we override final_action to HOLD (0). 
        # (This will NOT prevent SafeNet from triggering a hard stop-loss later)
        if self.trader.shares_held != 0 and self.trader.entry_price is not None:
            is_closing_long = (self.trader.shares_held > 0 and final_action == 2)
            is_closing_short = (self.trader.shares_held < 0 and final_action == 1)
            
            if is_closing_long or is_closing_short:
                gross_pnl = 0
                invested_amount = self.trader.entry_price * abs(self.trader.shares_held)
                if is_closing_long:
                    gross_pnl = (current_price - self.trader.entry_price) * abs(self.trader.shares_held)
                elif is_closing_short:
                    gross_pnl = (self.trader.entry_price - current_price) * abs(self.trader.shares_held)
                
                min_profit_required = max(50.0, invested_amount * 0.01) # 1% or ₹50
                if gross_pnl < min_profit_required:
                    final_action = 0  # Override to HOLD

        # C. SafeNet Override (Hard SL/TP)
        safe_action = self.safenet.check_rules(current_price, abs(self.trader.shares_held) * current_price, self.trader.shares_held)
        _safenet_triggered = safe_action is not None and safe_action != 0
        if _safenet_triggered:
            final_action = safe_action

        self.last_action_str = action_names[final_action]

        # ── Overtrading Guard ──────────────────────────────────────────
        # Block new entries if daily cap reached.
        # Block exits if minimum hold candles haven't elapsed (SafeNet overrides this).
        ticks_since_last = self._tick_count - self._last_trade_tick
        cooldown_ok      = ticks_since_last >= self.MIN_HOLD_CANDLES
        cap_ok           = self._daily_trade_count < self.MAX_DAILY_TRADES

        if not cap_ok and self.trader.shares_held == 0:
            # Daily cap hit and flat — block new entries entirely
            if self._tick_count % 60 == 0:  # log once per hour
                print(f"[Agent:{self.ticker}] Daily trade cap ({self.MAX_DAILY_TRADES}) reached. Holding flat.")
            return

        if final_action != 0 and not cooldown_ok and not _safenet_triggered:
            # Still in cooldown window after last trade — suppress signal
            final_action = 0
            self.last_action_str = "HOLD"


        now_ts = time.time()
        if now_ts - self._last_decision_log > 300:  # Every 5 minutes
            cash = self.account.get_cash()
            print(f"[Agent:{self.ticker}] DECISION: price={current_price:.2f} dqn={action_names[dqn_action]} chart={chart_result['signal']} final={action_names[final_action]} shares={self.trader.shares_held} cash={cash:.0f}")
            self._last_decision_log = now_ts

        # 3. Execution (Quantity logic)
        if final_action != 0:
            qty = 0
            if final_action == 1: # BUY / COVER
                if self.trader.shares_held < 0:
                    qty = abs(self.trader.shares_held)
                else:
                    cash = self.account.get_cash() - 20 # Reserve brokerage
                    qty = int(max(0, cash) // current_price)
            elif final_action == 2: # SELL / SHORT
                if self.trader.shares_held > 0:
                    qty = self.trader.shares_held
                else:
                    # Short selling disabled per user request
                    qty = 0
                    final_action = 0

            if qty > 0:
                # Determine exit reason for research logs
                if final_action == 2:
                    if _safenet_triggered:
                        # Check which SafeNet rule fired from the last print
                        exit_reason = "TRAILING_STOP" if self.safenet._peak_price and self.safenet._peak_price > (self.trader.entry_price or 0) else "STOP_LOSS"
                    else:
                        exit_reason = "SIGNAL"
                else:
                    exit_reason = "SIGNAL"

                print(f"[Agent:{self.ticker}] EXECUTING: {action_names[final_action]} qty={qty} @ {current_price:.2f} | reason={exit_reason}")
                success = self.trader.execute_trade(final_action, current_price, qty, exit_reason=exit_reason)
                if success:
                    self._last_trade_tick = self._tick_count  # arm cooldown
                    if final_action == 2:  # completed a round-trip sell
                        self._daily_trade_count += 1
                    if final_action == 1 and self.trader.shares_held > 0:
                        self.safenet.update_position(current_price)  # Arm trailing stop
                    elif final_action == 2 and self.trader.shares_held == 0:
                        self.safenet.update_position(None)  # Reset on close
                    self.last_action_time = datetime.now()
                    action_str = "BUY" if final_action == 1 else "SELL"
                    cash_after = self.account.get_cash()
                    print(f"[Agent:{self.ticker}] TRADE SUCCESS: {action_str} {qty} @ {current_price:.2f} | Cash={cash_after:.0f} | DailyTrades={self._daily_trade_count}/{self.MAX_DAILY_TRADES}")
                    from .utils.notifications import notifier
                    msg = f"{'🟢' if final_action == 1 else '🔴'} *{self.ticker} {action_str}* @ ₹{current_price:.2f} | qty={qty}"
                    notifier.send_message(msg)
                else:
                    print(f"[Agent:{self.ticker}] TRADE REJECTED: {action_names[final_action]} qty={qty} @ {current_price:.2f} (insufficient funds or portfolio limit)")
            elif final_action != 0:
                if self._tick_count <= 10:
                    print(f"[Agent:{self.ticker}] QTY=0: {action_names[final_action]} skipped, cash={self.account.get_cash():.0f}, price={current_price:.2f}")

    def get_status(self, current_price: float) -> Dict[str, Any]:
        summary = self.trader.get_portfolio_summary(current_price)
        return {
            "ticker": self.ticker, "shares": summary["shares"], "value": summary["position_value"],
            "last_action": self.last_action_str, "last_action_time": self.last_action_time,
            "chart_signal": self.last_chart_signal, "chart_reasons": self.last_chart_reasons
        }
