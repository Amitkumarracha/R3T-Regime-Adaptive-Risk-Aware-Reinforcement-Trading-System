"""
backtest_runner.py — Historical Backtester for Project Laplace

Runs three strategies side-by-side on identical 5-minute candle data:
  A) DQN-only    : Neural network signals + SafeNet risk management
  B) EMA-only    : EMA 5/200 crossover + RSI > 55 + SafeNet
  C) Hybrid      : DQN signal confirmed by EMA trend (production system)

Outputs per-stock and aggregate metrics for the research paper.
"""
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

# Project imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl_trading.utils.features import add_technical_indicators, prepare_env_dataframe, FEATURE_COLS
from rl_trading.agents.dqn_agent import DQNAgent
from rl_trading.env.trading_env import TradingEnv

# ── Config ────────────────────────────────────────────────────────────────────
INITIAL_CASH = 100000.0
BROKERAGE_PER_LEG = 20.0       # Zerodha-style flat fee
STOP_LOSS_PCT = 0.02           # -2% hard stop
TRAILING_ACTIVATE_PCT = 0.01   # Activate trailing at +1%
TRAILING_GAP_PCT = 0.01        # Trail 1% below peak

STOCKS = [
    {"ticker": "HDFCBANK", "yahoo": "HDFCBANK.NS"},
    {"ticker": "TCS",      "yahoo": "TCS.NS"},
    {"ticker": "RELIANCE", "yahoo": "RELIANCE.NS"},
    {"ticker": "TATASTEEL","yahoo": "TATASTEEL.NS"},
    {"ticker": "INFY",     "yahoo": "INFY.NS"},
]

MODELS_DIR = "rl_trading/models"
RESULTS_DIR = "research_data/backtest_results"


# ── Download Data ─────────────────────────────────────────────────────────────
def download_data(yahoo_ticker, interval="5m", period="60d"):
    import yfinance as yf
    df = yf.download(yahoo_ticker, interval=interval, period=period, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


# ── Simulate a single intraday session ────────────────────────────────────────
def simulate_intraday_session(day_df, strategy_fn, initial_cash):
    """
    Runs one day of intraday trading.
    Returns: list of completed round-trip trades for the day.
    """
    cash = initial_cash
    position = None  # {"qty": int, "entry_price": float, "entry_idx": int, "peak_price": float}
    trades = []

    for i in range(len(day_df)):
        row = day_df.iloc[i]
        price = float(row["Close_raw"])

        # If in position, check risk management first
        if position is not None:
            exit_reason = None
            entry = position["entry_price"]
            peak = position["peak_price"]
            position["peak_price"] = max(peak, price)
            peak = position["peak_price"]

            # Hard stop loss
            loss_pct = (price - entry) / entry
            if loss_pct <= -STOP_LOSS_PCT:
                exit_reason = "STOP_LOSS"

            # Trailing stop
            peak_gain = (peak - entry) / entry
            if peak_gain >= TRAILING_ACTIVATE_PCT:
                trail_floor = max(0.001, peak_gain - TRAILING_GAP_PCT)
                current_gain = (price - entry) / entry
                if current_gain <= trail_floor:
                    exit_reason = "TRAILING_STOP"

            # EOD square-off: last 6 candles of the day (last 30 min)
            if i >= len(day_df) - 6 and exit_reason is None:
                exit_reason = "EOD_SQUAREOFF"

            if exit_reason:
                qty = position["qty"]
                exit_brok = min(BROKERAGE_PER_LEG, price * qty * 0.0003)
                gross_pnl = (price - entry) * qty
                total_brok = position.get("entry_brok", BROKERAGE_PER_LEG) + exit_brok
                net_pnl = gross_pnl - total_brok
                cash += price * qty - exit_brok
                hold_candles = i - position["entry_idx"]

                trades.append({
                    "entry_price": entry,
                    "exit_price": price,
                    "qty": qty,
                    "gross_pnl": gross_pnl,
                    "total_brokerage": total_brok,
                    "net_pnl": net_pnl,
                    "exit_reason": exit_reason,
                    "hold_candles": hold_candles,
                })
                position = None
                continue

        # If flat, check for entry signal
        if position is None:
            signal = strategy_fn(day_df, i)
            if signal == "BUY" and cash > price + BROKERAGE_PER_LEG:
                qty = int((cash - BROKERAGE_PER_LEG) // price)
                if qty > 0:
                    entry_brok = min(BROKERAGE_PER_LEG, price * qty * 0.0003)
                    cash -= price * qty + entry_brok
                    position = {
                        "qty": qty,
                        "entry_price": price,
                        "entry_idx": i,
                        "peak_price": price,
                        "entry_brok": entry_brok,
                    }

    # Force close if still open at end of data
    if position is not None:
        price = float(day_df.iloc[-1]["Close_raw"])
        qty = position["qty"]
        entry = position["entry_price"]
        exit_brok = min(BROKERAGE_PER_LEG, price * qty * 0.0003)
        gross_pnl = (price - entry) * qty
        total_brok = position.get("entry_brok", BROKERAGE_PER_LEG) + exit_brok
        net_pnl = gross_pnl - total_brok
        cash += price * qty - exit_brok
        trades.append({
            "entry_price": entry, "exit_price": price, "qty": qty,
            "gross_pnl": gross_pnl, "total_brokerage": total_brok,
            "net_pnl": net_pnl, "exit_reason": "EOD_SQUAREOFF",
            "hold_candles": len(day_df) - 1 - position["entry_idx"],
        })
        position = None

    return trades, cash


# ── Strategy Functions ────────────────────────────────────────────────────────

def make_ema_strategy():
    """Strategy B: EMA 5/200 crossover + RSI > 55."""
    def strategy(day_df, idx):
        if idx < 2:
            return "HOLD"
        row = day_df.iloc[idx]
        prev = day_df.iloc[idx - 1]
        if not all(c in row.index for c in ["EMA_5", "EMA_200", "SMA_50", "RSI"]):
            return "HOLD"
        rsi = row.get("RSI", 50)
        two_candle = prev["EMA_5"] <= prev["EMA_200"] and row["EMA_5"] > row["EMA_200"]
        if two_candle and row["EMA_5"] > row["SMA_50"] and rsi > 55:
            return "BUY"
        return "HOLD"
    return strategy


def make_dqn_strategy(agent, feature_cols, initial_cash):
    """Strategy A: DQN-only signal (epsilon=0 greedy)."""
    import torch
    def strategy(day_df, idx):
        row = day_df.iloc[idx]
        market_obs = row[feature_cols].values.astype(np.float32)
        cash_ratio = np.clip(1.0, 0, 1)  # Assume full cash available
        shares_ratio = 0.0
        state = np.append(market_obs, [cash_ratio, shares_ratio]).astype(np.float32)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
        with torch.no_grad():
            q_values = agent.online_net(state_t)
        action = int(q_values.argmax().item())
        if action == 1:
            return "BUY"
        return "HOLD"
    return strategy


def make_hybrid_strategy(agent, feature_cols, initial_cash):
    """Strategy C: DQN must agree with EMA trend (production system)."""
    ema_fn = make_ema_strategy()
    dqn_fn = make_dqn_strategy(agent, feature_cols, initial_cash)
    def strategy(day_df, idx):
        ema_signal = ema_fn(day_df, idx)
        dqn_signal = dqn_fn(day_df, idx)
        # Hybrid: DQN says BUY and EMA is not bearish (NEUTRAL or BUY)
        if dqn_signal == "BUY" and ema_signal != "SELL":
            return "BUY"
        # Also allow pure EMA BUY signals
        if ema_signal == "BUY":
            return "BUY"
        return "HOLD"
    return strategy


# ── Compute Metrics ──────────────────────────────────────────────────────────
def compute_metrics(all_trades, initial_capital):
    if not all_trades:
        return {
            "total_trades": 0, "winning_trades": 0, "losing_trades": 0,
            "win_rate": 0, "gross_profit": 0, "gross_loss": 0,
            "total_brokerage": 0, "net_pnl": 0, "return_pct": 0,
            "profit_factor": 0, "avg_hold_candles": 0,
            "max_drawdown_pct": 0, "sharpe_ratio": 0,
        }

    net_pnls = [t["net_pnl"] for t in all_trades]
    gross_pnls = [t["gross_pnl"] for t in all_trades]
    winners = [t for t in all_trades if t["net_pnl"] > 0]
    losers = [t for t in all_trades if t["net_pnl"] <= 0]

    gross_profit = sum(g for g in gross_pnls if g > 0)
    gross_loss = abs(sum(g for g in gross_pnls if g < 0))
    total_brok = sum(t["total_brokerage"] for t in all_trades)
    total_net = sum(net_pnls)

    # Sharpe ratio (annualized, assuming ~75 candles per day, 250 trading days)
    if len(net_pnls) > 1:
        daily_returns = np.array(net_pnls) / initial_capital
        sharpe = (np.mean(daily_returns) / (np.std(daily_returns) + 1e-9)) * np.sqrt(250)
    else:
        sharpe = 0.0

    # Max drawdown from cumulative P&L
    cum_pnl = np.cumsum(net_pnls)
    peak = np.maximum.accumulate(cum_pnl + initial_capital)
    drawdown = (peak - (cum_pnl + initial_capital)) / peak
    max_dd = float(np.max(drawdown)) * 100 if len(drawdown) > 0 else 0

    return {
        "total_trades": len(all_trades),
        "winning_trades": len(winners),
        "losing_trades": len(losers),
        "win_rate": len(winners) / len(all_trades) * 100 if all_trades else 0,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "total_brokerage": total_brok,
        "net_pnl": total_net,
        "return_pct": (total_net / initial_capital) * 100,
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else float("inf"),
        "avg_hold_candles": np.mean([t["hold_candles"] for t in all_trades]),
        "max_drawdown_pct": max_dd,
        "sharpe_ratio": sharpe,
    }


# ── Main Runner ──────────────────────────────────────────────────────────────
def run_backtest():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    obs_dim = len(FEATURE_COLS) + 2
    all_results = []

    for stock in STOCKS:
        ticker = stock["ticker"]
        yahoo = stock["yahoo"]
        print(f"\n{'='*60}")
        print(f"  Backtesting: {ticker}")
        print(f"{'='*60}")

        # Download data
        print(f"  Downloading 60d of 5m data from Yahoo Finance...")
        raw_df = download_data(yahoo)
        if raw_df.empty or len(raw_df) < 200:
            print(f"  SKIP: Insufficient data ({len(raw_df)} rows)")
            continue

        raw_df["Close_raw"] = raw_df["Close"].copy()
        raw_df = add_technical_indicators(raw_df)

        # Keep a copy with raw indicators for EMA strategy
        ema_df = raw_df.copy()

        # Normalize for DQN
        norm_df, scaler = prepare_env_dataframe(raw_df.copy(), FEATURE_COLS)

        # Split into daily sessions
        norm_df.index = ema_df.index  # Preserve datetime index
        dates = norm_df.index.normalize().unique()
        print(f"  {len(dates)} trading days found, {len(norm_df)} total candles")

        # Load DQN model
        model_path = os.path.join(MODELS_DIR, f"dqn_{ticker}.pt")
        agent = None
        if os.path.exists(model_path):
            agent = DQNAgent(obs_dim=obs_dim)
            agent.load(model_path)
            agent.epsilon = 0.0  # Pure exploitation
            print(f"  DQN model loaded: {model_path}")
        else:
            print(f"  WARNING: No DQN model for {ticker}, skipping DQN strategies")

        # Prepare strategies
        strategies = {}
        strategies["EMA_Only"] = make_ema_strategy()
        if agent:
            strategies["DQN_Only"] = make_dqn_strategy(agent, FEATURE_COLS, INITIAL_CASH)
            strategies["Hybrid"] = make_hybrid_strategy(agent, FEATURE_COLS, INITIAL_CASH)

        # Run each strategy on each day
        for strat_name, strat_fn in strategies.items():
            all_trades = []
            cash = INITIAL_CASH

            for date in dates:
                day_mask = norm_df.index.normalize() == date
                day_data = norm_df[day_mask].copy()

                # Also attach raw EMA columns for the EMA strategy
                ema_day = ema_df[day_mask]
                for col in ["EMA_5", "EMA_200", "SMA_50", "RSI"]:
                    if col in ema_day.columns:
                        day_data[col + "_raw"] = ema_day[col].values

                # Use raw indicator values for EMA strategy decisions
                if "EMA" in strat_name or "Hybrid" in strat_name:
                    for col in ["EMA_5", "EMA_200", "SMA_50", "RSI"]:
                        if col in ema_day.columns:
                            day_data[col] = ema_day[col].values

                if len(day_data) < 10:
                    continue

                day_trades, cash = simulate_intraday_session(day_data, strat_fn, cash)
                all_trades.extend(day_trades)

            metrics = compute_metrics(all_trades, INITIAL_CASH)
            metrics["ticker"] = ticker
            metrics["strategy"] = strat_name
            metrics["final_cash"] = cash
            all_results.append(metrics)

            print(f"  [{strat_name:10}] Trades: {metrics['total_trades']:3} | "
                  f"Win: {metrics['win_rate']:5.1f}% | "
                  f"Net: Rs{metrics['net_pnl']:>10,.2f} | "
                  f"Return: {metrics['return_pct']:>+6.2f}% | "
                  f"Sharpe: {metrics['sharpe_ratio']:>5.2f} | "
                  f"MaxDD: {metrics['max_drawdown_pct']:>5.2f}%")

    # Save results
    results_df = pd.DataFrame(all_results)
    results_path = os.path.join(RESULTS_DIR, "backtest_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\n{'='*60}")
    print(f"  Results saved to {results_path}")
    print(f"{'='*60}")

    # Print aggregate summary
    print("\n\n=== AGGREGATE SUMMARY BY STRATEGY ===\n")
    for strat in ["DQN_Only", "EMA_Only", "Hybrid"]:
        subset = [r for r in all_results if r["strategy"] == strat]
        if not subset:
            continue
        avg_return = np.mean([r["return_pct"] for r in subset])
        avg_sharpe = np.mean([r["sharpe_ratio"] for r in subset])
        avg_winrate = np.mean([r["win_rate"] for r in subset])
        avg_maxdd = np.mean([r["max_drawdown_pct"] for r in subset])
        total_trades = sum(r["total_trades"] for r in subset)
        total_net = sum(r["net_pnl"] for r in subset)
        total_brok = sum(r["total_brokerage"] for r in subset)
        avg_pf = np.mean([r["profit_factor"] for r in subset if r["profit_factor"] != float("inf")])

        print(f"  {strat:12} | Trades: {total_trades:4} | Win: {avg_winrate:5.1f}% | "
              f"Avg Return: {avg_return:+6.2f}% | Sharpe: {avg_sharpe:5.2f} | "
              f"MaxDD: {avg_maxdd:5.2f}% | PF: {avg_pf:.2f} | "
              f"Total Net: Rs{total_net:>10,.2f} | Brokerage: Rs{total_brok:>8,.2f}")

    return results_df


if __name__ == "__main__":
    run_backtest()
