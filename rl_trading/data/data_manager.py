# rl_trading/data/data_manager.py

import os
import time
import json
import traceback
import pandas as pd
from datetime import datetime, timedelta
import pytz
import threading
from typing import Dict, Optional, Callable, List
from concurrent.futures import ThreadPoolExecutor

from .paisa_helper import FivePaisaHelper
from ..universe.stocks import StockUniverse
from ..utils.features import add_technical_indicators

class DataManager:
    def __init__(self, buffer_size_candles: int = 300, interval_minutes: int = 5):
        self.universe = StockUniverse()
        self.buffer_size = buffer_size_candles
        self.interval_mins = interval_minutes
        
        # We now REQUIRE live data for any operation in the new architecture
        self.use_live = True 
        self.buffers: Dict[str, pd.DataFrame] = {}
        
        # Reverse mapping for ScripCode -> Ticker
        self._scrip_to_ticker: Dict[int, str] = {
            s['scrip_code']: s['ticker'] for s in self.universe.get_all_stocks()
        }
        for idx in self.universe.get_indices():
            self._scrip_to_ticker[idx['scrip_code']] = idx['name']

        self.callbacks = []
        # Thread pool for offloading agent logic (prevents ws blocking)
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Tick counters for diagnostics
        self._tick_count = 0
        self._last_tick_log_time = None
        self._candle_updates = 0
        
        self.paisa = FivePaisaHelper()
        if not self.paisa.is_connected:
            print("[DataManager] CRITICAL: 5Paisa connection failed. System will be inactive.")
            self.use_live = False

    def load_initial_history(self, interval: str = "5m", period_days: int = 7):
        """Pre-fill buffers using 5Paisa Historical Data only."""
        if not self.use_live: return

        stocks = self.universe.get_all_stocks()
        indices = self.universe.get_indices()
        print(f"[DataManager] Loading 5m history ({period_days} days) via 5Paisa...")

        tz = pytz.timezone('Asia/Kolkata')
        today = datetime.now(tz).strftime("%Y-%m-%d")
        from_date = (datetime.now(tz) - timedelta(days=period_days)).strftime("%Y-%m-%d")

        targets = [{"ticker": s['ticker'], "scrip": s['scrip_code'], "is_index": False} for s in stocks]
        targets += [{"ticker": idx['name'], "scrip": idx['scrip_code'], "is_index": True} for idx in indices]

        for target in targets:
            ticker = target['ticker']
            try:
                df = self.paisa.get_historical(target['scrip'], timeframe=interval, from_dt=from_date, to_dt=today, is_index=target['is_index'])
                if df is not None and not df.empty:
                    df['Close_raw'] = df['Close']
                    if not target['is_index']:
                        df = add_technical_indicators(df)
                    self.buffers[ticker] = df.tail(self.buffer_size).copy()
                    print(f"[DataManager] Initialized {ticker} with {len(self.buffers[ticker])} bars.")
                else:
                    print(f"[DataManager] WARNING: Empty history for {ticker} (scrip={target['scrip']})")
            except Exception as e:
                print(f"[DataManager] Failed to load history for {ticker}: {e}")

    def register_callback(self, callback: Callable):
        self.callbacks.append(callback)

    def _notify(self, ticker: str, latest_df: pd.DataFrame):
        """Asynchronously trigger agent logic."""
        self.executor.submit(self._run_callbacks, ticker, latest_df)

    def _run_callbacks(self, ticker, df):
        for cb in self.callbacks:
            try: cb(ticker, df)
            except Exception as e:
                print(f"[DataManager] Callback Error [{ticker}]: {e}")
                traceback.print_exc()

    def start_streaming(self):
        from rl_trading.utils.notifications import notifier
        from rl_trading.utils.market_context import is_market_currently_open

        print("[DataManager] Starting Master Scheduling Loop...")
        print(f"[DataManager] Scrip map has {len(self._scrip_to_ticker)} entries: {list(self._scrip_to_ticker.values())}")
        print(f"[DataManager] Buffers loaded for: {list(self.buffers.keys())}")
        print(f"[DataManager] Registered callbacks: {len(self.callbacks)}")
        self._is_streaming_live = False
        
        while True:
            market_open = is_market_currently_open()
            if market_open:
                if self.use_live and self.paisa.is_connected:
                    if not self._is_streaming_live:
                        print("[DataManager] Market Open: Engaging 5Paisa Live Stream")
                        notifier.send_message("🟢 *Market Opened*\nData stream engaged via 5Paisa Native Websockets.")
                        self._is_streaming_live = True
                        self._tick_count = 0
                        self._candle_updates = 0
                        self._last_tick_log_time = time.time()
                        codes = list(self._scrip_to_ticker.keys())
                        print(f"[DataManager] Subscribing to {len(codes)} scrip codes: {codes}")
                        threading.Thread(target=self._safe_subscribe, args=(codes,), daemon=True).start()
                    else:
                        # Periodic heartbeat every 60s while streaming
                        now = time.time()
                        if self._last_tick_log_time and (now - self._last_tick_log_time) > 300:
                            print(f"[DataManager] HEARTBEAT: {self._tick_count} ticks received, {self._candle_updates} candle updates triggered since last report")
                            self._tick_count = 0
                            self._candle_updates = 0
                            self._last_tick_log_time = now
            else:
                if self._is_streaming_live:
                    print("[DataManager] Market Closed: Disengaging 5Paisa.")
                    print(f"[DataManager] Session totals: {self._tick_count} ticks, {self._candle_updates} candle updates")
                    notifier.send_message("🔴 *Market Closed or Holiday*\nStream disengaged.")
                    self._is_streaming_live = False
            
            time.sleep(60)

    def _safe_subscribe(self, codes):
        """Wrapper to catch and log any subscription errors."""
        try:
            self.paisa.subscribe_live(codes, self._on_tick)
        except Exception as e:
            print(f"[DataManager] CRITICAL: subscribe_live crashed: {e}")
            traceback.print_exc()
            self._is_streaming_live = False

    def _on_tick(self, ws, message):
        """Processes real-time ticks and aggregates into 5-minute candles."""
        try:
            self._tick_count += 1
            
            # Log first few ticks for debugging
            if self._tick_count <= 5:
                print(f"[DataManager] RAW TICK #{self._tick_count}: type={type(message).__name__}, preview={str(message)[:200]}")
            
            if isinstance(message, (bytes, bytearray)): 
                message = message.decode('utf-8')
            parsed = json.loads(message) if isinstance(message, str) else message
            
            # 5Paisa sends data as a JSON ARRAY of tick objects: [{...}, {...}]
            # Unwrap the list and process each tick
            if isinstance(parsed, list):
                for item in parsed:
                    self._process_single_tick(item)
            elif isinstance(parsed, dict):
                self._process_single_tick(parsed)
            else:
                if self._tick_count <= 10:
                    print(f"[DataManager] TICK UNKNOWN FORMAT: type={type(parsed).__name__}")
                
        except Exception as e:
            print(f"[DataManager] TICK ERROR: {e}")
            traceback.print_exc()

    def _process_single_tick(self, data):
        """Process a single tick dict from the WebSocket feed."""
        if not isinstance(data, dict):
            return
            
        # 5Paisa payload mapping
        scrip_code = data.get('Token') or data.get('ScripCode') or data.get('scripCode')
        ltp = data.get('LastRate') or data.get('LastTradedPrice') or data.get('ltp') or data.get('Ltp')
        
        if scrip_code is None or ltp is None:
            if self._tick_count <= 10:
                print(f"[DataManager] TICK SKIP: scrip_code={scrip_code}, ltp={ltp}, keys={list(data.keys())}")
            return

        ticker = self._scrip_to_ticker.get(int(scrip_code))
        if ticker is None:
            return
            
        if ticker not in self.buffers:
            return

        ltp = float(ltp)
        now_ist = datetime.now(pytz.timezone('Asia/Kolkata'))
        
        # --- Native 5-Minute Candle Logic ---
        candle_time = now_ist.replace(second=0, microsecond=0)
        candle_time = candle_time - timedelta(minutes=candle_time.minute % self.interval_mins)
        
        buf = self.buffers[ticker]
        stock_tickers = [s['ticker'] for s in self.universe.get_all_stocks()]

        if buf.empty or candle_time > buf.index[-1]:
            # ── CANDLE CLOSE EVENT ──────────────────────────────────────
            # A new 5-min interval just started = the PREVIOUS candle just CLOSED.
            # This is the ONLY moment we notify agents — on closed, confirmed candles.
            # This reduces agent decisions from ~1000/day to exactly ~12/day per stock.
            if not buf.empty and ticker in stock_tickers:
                closed_df = add_technical_indicators(buf.copy())
                self._notify(ticker, closed_df)

            # Now append the new developing candle
            new_row = pd.DataFrame({
                'Open': [ltp], 'High': [ltp], 'Low': [ltp], 'Close': [ltp], 
                'Volume': [0], 'Close_raw': [ltp]
            }, index=[candle_time])
            self.buffers[ticker] = pd.concat([buf, new_row]).tail(self.buffer_size)
            self._candle_updates += 1
            if self._candle_updates <= 40:
                print(f"[DataManager] NEW CANDLE: {ticker} @ {candle_time} LTP={ltp:.2f} (buf={len(self.buffers[ticker])})")
        else:
            # ── INTRA-CANDLE TICK ──────────────────────────────────────
            # Update the developing candle's OHLCV in-place.
            # Do NOT notify agents — we only act on closed candles.
            last_idx = buf.index[-1]
            self.buffers[ticker].at[last_idx, 'Close'] = ltp
            self.buffers[ticker].at[last_idx, 'Close_raw'] = ltp
            self.buffers[ticker].at[last_idx, 'High'] = max(buf.at[last_idx, 'High'], ltp)
            self.buffers[ticker].at[last_idx, 'Low'] = min(buf.at[last_idx, 'Low'], ltp)
            
            vol = data.get('TotalQty') or data.get('Volume') or data.get('VolumeTradedToday')
            if vol is not None: 
                self.buffers[ticker].at[last_idx, 'Volume'] = int(vol)

