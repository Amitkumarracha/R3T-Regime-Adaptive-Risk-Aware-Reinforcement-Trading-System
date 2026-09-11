# rl_trading/data/paisa_helper.py

import os
from typing import Dict, List, Callable
import pandas as pd

class FivePaisaHelper:
    """
    Wrapper for py5paisa SDK.
    Handles TOTP-based auto-login using pyotp, historic data fetch, and live WebSocket.
    """
    
    def __init__(self):
        self._load_credentials()
        self.client = None
        self.is_connected = False
        
        # We only attempt login if USE_LIVE_DATA is true
        self.use_live = os.getenv("USE_LIVE_DATA", "false").lower() == "true"
        
        if self.use_live:
            try:
                from py5paisa import FivePaisaClient
                import pyotp
            except ImportError:
                print("WARNING: py5paisa or pyotp not installed. Cannot use live data.")
                self.use_live = False
                return

            self._login()
            
    def _load_credentials(self):
        self.app_name = os.getenv("FIVEPAISA_APP_NAME")
        self.app_source = os.getenv("FIVEPAISA_APP_SOURCE")
        self.user_key = os.getenv("FIVEPAISA_USER_KEY")
        self.user_id = os.getenv("FIVEPAISA_USER_ID")
        self.encryption_key = os.getenv("FIVEPAISA_ENCRYPTION_KEY")
        self.client_code = os.getenv("FIVEPAISA_CLIENT_CODE")
        self.password = os.getenv("FIVEPAISA_PASSWORD")
        self.totp_secret = os.getenv("FIVEPAISA_TOTP_SECRET")

    def _login(self):
        print("Authenticating with 5paisa via TOTP...")
        try:
            from py5paisa import FivePaisaClient
            import pyotp
            totp_token = pyotp.TOTP(self.totp_secret).now()
            self.client = FivePaisaClient(
                cred={
                    "APP_NAME": self.app_name,
                    "APP_SOURCE": self.app_source,
                    "USER_KEY": self.user_key,
                    "USER_ID": self.user_id,
                    "PASSWORD": self.password,
                    "ENCRYPTION_KEY": self.encryption_key
                }
            )
            res = self.client.get_totp_session(self.client_code, str(totp_token), self.password)
            print(f"5paisa Login Response: {res}")
            if self.client.Jwt_token:
                self.is_connected = True
                print("Connected to 5paisa!")
            else:
                print("5paisa Login Failed: JWT token is empty.")
                self.is_connected = False
                self.use_live = False
        except Exception as e:
            print(f"Failed to connect to 5paisa: {e}")
            self.is_connected = False
            self.use_live = False

    def get_historical(self, scrip_code: int, timeframe: str = "5m",
                        from_dt: str = None, to_dt: str = None, exchange: str = "N", is_index: bool = False) -> pd.DataFrame:
        """
        Fetch historical OHLCV candles from 5paisa.
        timeframe: "5m", "15m", "1h", "1D"
        from_dt / to_dt: "YYYY-MM-DD"
        """
        if not self.use_live or not self.client:
            raise EnvironmentError("FivePaisa client not logged in.")
        
        # For indices, exchange type is often 'C' as well in 5paisa for Nifty 50 etc.
        exch_type = "C" 
        
        df = self.client.historical_data(exchange, exch_type, scrip_code, timeframe, from_dt, to_dt)
        if df is None or (hasattr(df, "empty") and df.empty):
            return pd.DataFrame()
        
        # 5Paisa returns columns like 'Datetime', 'Open', etc.
        if "Datetime" in df.columns:
            df["Datetime"] = pd.to_datetime(df["Datetime"])
            # The historical data from 5Paisa is in IST but naive. We need to localize it so it matches live ticks.
            import pytz
            ist = pytz.timezone('Asia/Kolkata')
            df["Datetime"] = df["Datetime"].dt.tz_localize(ist, nonexistent='shift_forward', ambiguous='NaT')
            df = df.set_index("Datetime")
            df.index.name = None
        
        # Ensure standard OHLCV columns
        required = ["Open", "High", "Low", "Close", "Volume"]
        available = [col for col in required if col in df.columns]
        df = df[available]
        return df

    def subscribe_live(self, scrip_codes: List[int], callback: Callable):
        if not self.use_live or not self.client:
            raise EnvironmentError("FivePaisa client not logged in.")

        # Request market feed ('mf') for stocks - subscribe ('s')
        req = [{"Exch": "N", "ExchType": "C", "ScripCode": code} for code in scrip_codes]
        print(f"[5Paisa WS] Building feed request for {len(req)} scrips...")
        
        try:
            payload = self.client.Request_Feed('mf', 's', req)
            print(f"[5Paisa WS] Feed payload built. Connecting WebSocket...")
            self.client.connect(payload)
            print(f"[5Paisa WS] WebSocket connected. Starting receive_data...")
        except Exception as e:
            print(f"[5Paisa WS] ERROR during connect: {e}")
            import traceback
            traceback.print_exc()
            return

        def on_message(ws, message):
            try:
                callback(ws, message)
            except Exception as e:
                print(f"[5Paisa WS] Callback error: {e}")
                import traceback
                traceback.print_exc()

        try:
            self.client.receive_data(on_message)
        except Exception as e:
            print(f"[5Paisa WS] receive_data crashed: {e}")
            import traceback
            traceback.print_exc()
        
    def unsubscribe_all(self):
        if self.use_live and self.client:
            print("Unsubscribing from 5paisa...")
            # py5paisa 0.7.21 doesn't have a simple disconnect, but closing the process works.

