import os
import requests
import time
import threading
from dotenv import load_dotenv
from .state_helper import get_trading_state, set_trading_state, get_account_state

load_dotenv("config/.env")

import sys

class Notifier:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Notifier, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        
        self.token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self.enabled = bool(self.token and self.chat_id)
        self.trader = None # MultiTrader instance
        self._last_update_id = 0
        
        if not self.enabled:
            print("[Notifier] Telegram credentials not found. Notifications disabled.")
        else:
            print("[Notifier] Telegram active. Starting listener thread...")
            self._start_listener()

    def set_trader(self, trader):
        self.trader = trader

    def send_message(self, message: str):
        if not self.enabled:
            return
            
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "Markdown"
        }
        
        try:
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
        except Exception as e:
            print(f"[Notifier] Failed to send message: {e}")

    def _start_listener(self):
        thread = threading.Thread(target=self._listener_loop, daemon=True)
        thread.start()

    def _listener_loop(self):
        while True:
            try:
                updates = self._get_updates()
                for update in updates:
                    self._handle_update(update)
            except Exception as e:
                print(f"[Notifier] Listener error: {e}")
            time.sleep(3)

    def _get_updates(self):
        url = f"https://api.telegram.org/bot{self.token}/getUpdates"
        params = {"offset": self._last_update_id + 1, "timeout": 30}
        response = requests.get(url, params=params, timeout=35)
        if response.status_code == 200:
            data = response.json()
            if data["ok"]:
                return data["result"]
        return []

    def _handle_update(self, update):
        if "message" not in update or "text" not in update["message"]:
            self._last_update_id = update["update_id"]
            return
            
        msg = update["message"]
        text = msg["text"].lower().strip()
        chat_id = str(msg["chat"]["id"])
        self._last_update_id = update["update_id"]
        
        # Security: Only respond to the authorized chat_id
        if chat_id != self.chat_id:
            return

        if text == "/start":
            self.send_message("👋 *Welcome to Laplace Trading Bot!*\n\nCommands:\n/status - Overall status\n/portfolio - Current holdings\n/pause - Stop trading\n/resume - Start trading\n/help - Show this message")
        
        elif text == "/status":
            t_state = get_trading_state()
            a_state = get_account_state()
            status = "🟢 *Trading Active*" if not t_state["paused"] else "🔴 *Trading Paused*"
            stock_count = len(self.trader.tickers) if self.trader and hasattr(self.trader, 'tickers') else "Unknown"
            report = f"📊 *System Status*\n\nStatus: {status}\nCash: ₹{a_state['cash']:.2f}\nTotal Stocks: {stock_count}"
            self.send_message(report)
            
        elif text == "/pause":
            set_trading_state(True)
            self.send_message("🔴 *Trading Paused.* No further trades will be executed.")
            
        elif text == "/resume":
            set_trading_state(False)
            self.send_message("🟢 *Trading Resumed.* Agents are now active.")
            
        elif text == "/portfolio":
            self._send_portfolio()
            
        elif text == "/market":
            self._send_market_summary()
            
        elif text == "/allocation":
            self._send_allocation_summary()
            
        elif text == "/help":
            self.send_message("📊 *Supervisor Commands*:\n/market - Nifty & Expiry status\n/allocation - Basket usage\n/status - Overall status\n/portfolio - Current holdings\n/pause - Stop trading\n/resume - Start trading")

    def _send_market_summary(self):
        from .market_context import get_expiry_context
        ctx = get_expiry_context()
        report = "📉 *Market Vitals*\n\n"
        if ctx["is_expiry_day"]:
            report += f"⚠️ *EXPIRY DAY*: {ctx['expiry_type']}\n"
        else:
            report += "✅ Regular Trading Day\n"
            
        if self.trader and hasattr(self.trader, 'data_manager'):
            dm = self.trader.data_manager
            for idx in ['NIFTY 50', 'BANK NIFTY']:
                buf = dm.buffers.get(idx)
                if buf is not None and not buf.empty:
                    p = buf['Close'].iloc[-1]
                    c = p - buf['Open'].iloc[0]
                    sign = "🟢" if c >= 0 else "🔴"
                    report += f"{sign} *{idx}*: {p:,.2f} ({c:+.2f})\n"
        
        self.send_message(report)

    def _send_allocation_summary(self):
        if not self.trader or not hasattr(self.trader, 'portfolio_manager'):
            self.send_message("⚠️ Portfolio manager not ready.")
            return
        
        summary = self.trader.portfolio_manager.get_summary()
        report = "🧺 *Capital Baskets*\n\n"
        for cat, limit in summary['limits'].items():
            used = summary['usage'][cat]
            pct = (used / limit) * 100
            report += f"*{cat}*: ₹{used:,.0f} / ₹{limit:,.0f} ({pct:.1f}%)\n"
        
        self.send_message(report)

    def _send_portfolio(self):
        if not self.trader:
            self.send_message("⚠️ Trader instance not linked yet. Try again in a moment.")
            return
            
        report = "💼 *Current Portfolio*\n\n"
        has_positions = False
        
        for ticker, agent in self.trader.agents.items():
            shares = agent.trader.shares_held
            if shares > 0:
                has_positions = True
                # Get current price from buffer if available
                dm = self.trader.data_manager
                current_price = 0.0
                if ticker in dm.buffers and not dm.buffers[ticker].empty:
                    current_price = dm.buffers[ticker].iloc[-1]['Close']
                
                report += f"*{ticker}*: {shares} shares"
                if current_price > 0:
                    val = shares * current_price
                    report += f" (₹{val:,.2f})"
                report += "\n"
        
        if not has_positions:
            report += "_No active positions._"
            
        self.send_message(report)

# Global instance
notifier = Notifier()
