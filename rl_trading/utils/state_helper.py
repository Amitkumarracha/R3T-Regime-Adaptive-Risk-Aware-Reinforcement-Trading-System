import os
import json

STATE_FILE = "config/trading_state.json"
ACCOUNT_FILE = "config/account_state.json"

def get_trading_state():
    state = {"paused": False}
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                data = json.load(f)
                state["paused"] = data.get("paused", False)
        except:
            pass
    return state

def set_trading_state(paused):
    os.makedirs("config", exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump({"paused": paused}, f)

def get_account_state():
    state = {"cash": 100000.0, "initial_cash": 100000.0}
    if os.path.exists(ACCOUNT_FILE):
        try:
            with open(ACCOUNT_FILE, "r") as f:
                data = json.load(f)
                state["cash"] = data.get("cash", 100000.0)
                state["initial_cash"] = data.get("initial_cash", 100000.0)
        except:
            pass
    return state
