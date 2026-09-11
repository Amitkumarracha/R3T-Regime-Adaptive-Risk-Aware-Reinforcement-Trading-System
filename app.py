import os
import csv
import json
import threading
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

load_dotenv("config/.env")

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from rl_trading.universe.stocks import StockUniverse
from rl_trading.orchestrator.multi_trader import MultiTrader
from rl_trading.utils.ui_helpers import DOMAIN_MAP

# Set up Flask from root directory
app = Flask(__name__, template_folder='dashboard/templates', static_folder='dashboard/static')
CORS(app)
universe = StockUniverse()

# Global trader reference — set once background thread starts
trader_instance = None

from rl_trading.utils.state_helper import get_trading_state, set_trading_state, get_account_state

def get_state():
    t_state = get_trading_state()
    a_state = get_account_state()
    return {
        "paused": t_state["paused"],
        "cash": a_state["cash"]
    }

def set_state(paused):
    set_trading_state(paused)

# Helper to read latest trade info
def get_ticker_status(ticker):
    import glob
    from datetime import datetime
    log_files = glob.glob(f"logs/trades_{ticker}_*.csv")
    
    if not log_files:
        return {
            "ticker": ticker,
            "status": "No Data",
            "position": 0,
            "unrealized_pnl": 0.0,
            "last_action": "NONE",
            "last_price": 0.0
        }
        
    latest_file = max(log_files, key=os.path.getctime)
    
    try:
        with open(latest_file, "r") as f:
            lines = f.readlines()
            if len(lines) > 1:
                headers = lines[0].strip().split(",")
                last_line = lines[-1].strip().split(",")
                data = dict(zip(headers, last_line))
                
                return {
                    "ticker": ticker,
                    "status": "Active",
                    "position": int(float(data.get("Qty", 0))) if data.get("Action") == "BUY" else 0,
                    "last_action": data.get("Action", "NONE"),
                    "last_price": float(data.get("Exit") if data.get("Exit") else data.get("Entry", 0.0)),
                    "total_value": float(data.get("PositionValue", 0.0)),
                    "net_pnl": float(data.get("NetPnL", 0.0))
                }
    except Exception as e:
        pass

    return {
        "ticker": ticker,
        "status": "Active (No Trades)",
        "position": 0,
        "unrealized_pnl": 0.0,
        "last_action": "HOLD",
        "last_price": 0.0
    }

@app.route("/")
def index():
    stocks = universe.get_all_stocks()
    # Group by Sector
    sectors = {}
    for stock in stocks:
        stock['domain'] = DOMAIN_MAP.get(stock['ticker'], 'nseindia.com')
        sec = stock['sector']
        if sec not in sectors:
            sectors[sec] = []
        sectors[sec].append(stock)
        
    state = get_state()
    return render_template("index.html", sectors=sectors, paused=state['paused'])

@app.route("/api/stocks")
def api_stocks():
    stocks = universe.get_all_stocks()
    status_list = []
    
    for s in stocks:
        status = get_ticker_status(s['ticker'])
        status['sector'] = s['sector']
        status['name'] = s['ticker']
        status['domain'] = DOMAIN_MAP.get(s['ticker'], 'nseindia.com')
        status_list.append(status)
        
    global_state = get_state()
    return jsonify({
        "stocks": status_list,
        "global_cash": global_state.get("cash", 15000.0)
    })

@app.route("/api/portfolio")
def api_portfolio():
    global trader_instance
    if trader_instance and hasattr(trader_instance, 'portfolio_manager'):
        return jsonify(trader_instance.portfolio_manager.get_summary())
    return jsonify({"error": "Orchestrator not initialized"}), 503

@app.route("/api/market_vitals")
def api_market_vitals():
    from rl_trading.utils.market_context import get_expiry_context
    from rl_trading.utils.state_helper import get_account_state
    
    acc = get_account_state()
    expiry = get_expiry_context()
    
    indices = {}
    global trader_instance
    if trader_instance and hasattr(trader_instance, 'data_manager'):
        dm = trader_instance.data_manager
        for idx in ['NIFTY 50', 'BANK NIFTY', 'NIFTY IT']:
            buf = dm.buffers.get(idx)
            if buf is not None and not buf.empty:
                indices[idx] = {
                    "price": float(buf['Close'].iloc[-1]),
                    "change": float(buf['Close'].iloc[-1] - buf['Open'].iloc[0])
                }
    
    return jsonify({
        "account": acc,
        "expiry": expiry,
        "indices": indices
    })

@app.route("/api/logs/<ticker>")
def api_logs(ticker):
    import glob
    log_files = glob.glob(f"logs/trades_{ticker}_*.csv")
    if not log_files:
         return jsonify([])
         
    latest_file = max(log_files, key=os.path.getctime)
    trades = []
    try:
        with open(latest_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                trades.append(row)
    except Exception as e:
        pass
        
    return jsonify(trades)

@app.route("/api/chart/<ticker>")
def api_chart(ticker):
    from rl_trading.utils.features import add_technical_indicators
    import yfinance as yf

    interval = request.args.get('interval', '15m')

    # --- Try DataManager buffer first (live 5paisa data) ---
    global trader_instance
    df = None
    if trader_instance and hasattr(trader_instance, 'data_manager'):
        dm = trader_instance.data_manager
        buf = dm.buffers.get(ticker)
        if buf is not None and not buf.empty:
            df = buf.copy()

    # --- Fallback: fetch from yfinance ---
    if df is None or df.empty:
        if interval == "15m" or interval == "1h":
            period = "60d"
        elif interval == "1d":
            period = "1y"
        else:
            period = "60d"
            interval = "15m"

        stocks = universe.get_all_stocks()
        scrip = next((s for s in stocks if s['ticker'] == ticker), None)
        if not scrip:
            return jsonify({"error": "Ticker not found"}), 404

        df = yf.download(scrip['yahoo'], period=period, interval=interval,
                         progress=False, auto_adjust=True)
        if df.empty:
            return jsonify({"error": "No data"}), 404

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df[~df.index.duplicated(keep='last')]
        df.sort_index(inplace=True)
        df.dropna(inplace=True)

        try:
            df = add_technical_indicators(df)
        except Exception:
            pass

    if df is None or df.empty:
        return jsonify({"error": "No data"}), 404

    df = df[~df.index.duplicated(keep='last')]
    df.sort_index(inplace=True)

    # For intraday view: restrict chart to today's session only (day trading)
    if interval in ("15m", "1h"):
        import pytz
        tz = pytz.timezone("Asia/Kolkata")
        today_str = datetime.now(tz).strftime("%Y-%m-%d")
        idx = df.index
        if hasattr(idx, 'tz') and idx.tz is None:
            idx = idx.tz_localize("UTC").tz_convert(tz)
        elif hasattr(idx, 'tz') and idx.tz is not None:
            idx = idx.tz_convert(tz)
        today_mask = idx.strftime("%Y-%m-%d") == today_str
        if today_mask.any():
            df = df[today_mask]

    # Serialize to ApexCharts format
    candles_data = []
    lines_data   = []
    volume_data  = []
    ema5_data    = []
    sma50_data   = []
    ema200_data  = []

    for ts, row in df.iterrows():
        try:
            time_val = int(ts.timestamp() * 1000)
        except Exception:
            continue

        candles_data.append({"x": time_val,
            "y": [float(row.get('Open', 0)), float(row.get('High', 0)),
                  float(row.get('Low', 0)),  float(row.get('Close', 0))]})
        lines_data.append({"x": time_val, "y": float(row.get('Close', 0))})

        if 'Volume' in row:
            try:
                volume_data.append({"x": time_val, "y": int(row['Volume'])})
            except Exception:
                pass

        if 'EMA_5' in df.columns and not pd.isna(row.get('EMA_5')):
            ema5_data.append({"x": time_val, "y": float(row['EMA_5'])})
        if 'SMA_50' in df.columns and not pd.isna(row.get('SMA_50')):
            sma50_data.append({"x": time_val, "y": float(row['SMA_50'])})
        if 'EMA_200' in df.columns and not pd.isna(row.get('EMA_200')):
            ema200_data.append({"x": time_val, "y": float(row['EMA_200'])})

    return jsonify({
        "candles": candles_data,
        "lines":   lines_data,
        "volume":  volume_data,
        "ema5":    ema5_data,
        "sma50":   sma50_data,
        "ema200":  ema200_data,
        "source":  "5paisa" if (trader_instance and hasattr(trader_instance, 'data_manager')
                               and ticker in trader_instance.data_manager.buffers) else "yfinance"
    })

@app.route("/api/pause/status")
def pause_status():
    return jsonify(get_state())

@app.route("/api/pause/toggle", methods=["POST"])
def toggle_pause():
    state = get_state()
    state["paused"] = not state["paused"]
    set_state(state["paused"])
    return jsonify({"paused": state["paused"]})


# Background Orchestrator Thread
def run_orchestrator():
    global trader_instance
    from rl_trading.orchestrator.multi_trader import MultiTrader
    from rl_trading.data.data_manager import DataManager
    from rl_trading.shared_account import SharedAccount
    
    dm = DataManager()
    acc = SharedAccount(initial_cash=100000.0)
    stocks = universe.get_all_stocks()
    
    trader_instance = MultiTrader(tickers=stocks, account=acc, data_manager=dm)
    
    # Link to Telegram Notifier
    from rl_trading.utils.notifications import notifier
    notifier.set_trader(trader_instance)
    
    trader_instance.start()

if __name__ == "__main__":
    # Start Orchestrator Thread
    bg_thread = threading.Thread(target=run_orchestrator, daemon=True)
    bg_thread.start()
    
    # Start Flask UI
    port = int(os.environ.get("DASHBOARD_PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
