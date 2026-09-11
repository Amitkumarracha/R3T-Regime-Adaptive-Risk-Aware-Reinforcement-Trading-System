"""
refine_daily.py — Nightly DQN Refinement using 5Paisa Historical Data

Runs every night at 4:00 PM IST (after market closes at 3:30 PM).
Fetches the latest 10 days of 5-minute candles via 5Paisa API,
fine-tunes each stock's DQN model on recent price action,
and saves the updated model weights back to disk.

The live trading system picks up the updated models automatically on next restart.
"""
import os
import sys
import time
import pytz
from datetime import datetime, timedelta

# Add the project root to path so imports work when run from cron
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from rl_trading.universe.stocks import StockUniverse
from rl_trading.data.paisa_helper import FivePaisaHelper
from rl_trading.utils.features import add_technical_indicators, prepare_env_dataframe, FEATURE_COLS
from rl_trading.env.trading_env import TradingEnv
from rl_trading.agents.dqn_agent import DQNAgent

# ── Config ────────────────────────────────────────────────────────────────────
INTERVAL      = "5m"
LOOKBACK_DAYS = 10        # Retrain on last 10 trading days
N_EPISODES    = 20        # Fine-tuning episodes (not full retrain)
EPSILON       = 0.05      # Low exploration — we're refining, not discovering
INITIAL_CASH  = 100000.0
MODELS_DIR    = os.path.join(PROJECT_ROOT, "rl_trading", "models")
IST           = pytz.timezone('Asia/Kolkata')

def refine_stock(ticker, scrip_code, paisa: FivePaisaHelper, obs_dim: int):
    model_path = os.path.join(MODELS_DIR, f"dqn_{ticker}.pt")

    if not os.path.exists(model_path):
        print(f"  [SKIP] {ticker}: No base model found.")
        return False

    # ── Fetch recent data via 5Paisa ─────────────────────────────────────
    today = datetime.now(IST).strftime("%Y-%m-%d")
    from_date = (datetime.now(IST) - timedelta(days=LOOKBACK_DAYS + 3)).strftime("%Y-%m-%d")

    df = paisa.get_historical(scrip_code, timeframe=INTERVAL, from_dt=from_date, to_dt=today)
    if df is None or df.empty or len(df) < 50:
        print(f"  [SKIP] {ticker}: Not enough data ({len(df) if df is not None else 0} bars).")
        return False

    df['Close_raw'] = df['Close']
    df = add_technical_indicators(df)
    df, _ = prepare_env_dataframe(df, FEATURE_COLS)

    if len(df) < 50:
        print(f"  [SKIP] {ticker}: Not enough data after indicators ({len(df)} bars).")
        return False

    # ── Load existing model ───────────────────────────────────────────────
    agent = DQNAgent(obs_dim=obs_dim)
    try:
        agent.load(model_path)
        agent.epsilon = EPSILON
    except Exception as e:
        print(f"  [ERROR] {ticker}: Failed to load model — {e}")
        return False

    # ── Fine-tune on recent market data ──────────────────────────────────
    env = TradingEnv(df, feature_cols=FEATURE_COLS, initial_cash=INITIAL_CASH)
    total_reward = 0.0

    for episode in range(1, N_EPISODES + 1):
        state, _ = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            agent.learn()
            state = next_state
            ep_reward += reward
        total_reward += ep_reward

    avg_reward = total_reward / N_EPISODES
    agent.save(model_path)
    print(f"  [OK] {ticker}: Refined over {N_EPISODES} episodes | Avg reward: {avg_reward:.2f}")
    return True


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Laplace Nightly Refinement — {datetime.now(IST).strftime('%Y-%m-%d %H:%M IST')}")
    print(f"{'='*60}\n")

    # Connect to 5Paisa
    paisa = FivePaisaHelper()
    if not paisa.is_connected:
        print("CRITICAL: 5Paisa connection failed. Aborting refinement.")
        return

    universe = StockUniverse()
    stocks = universe.get_all_stocks()
    obs_dim = len(FEATURE_COLS) + 2

    start = datetime.now()
    success_count = 0

    for stock in stocks:
        ticker = stock['ticker']
        scrip_code = stock['scrip_code']
        print(f"\n>>> Refining {ticker} (scrip={scrip_code})...")
        try:
            ok = refine_stock(ticker, scrip_code, paisa, obs_dim)
            if ok:
                success_count += 1
        except Exception as e:
            print(f"  [ERROR] {ticker}: Unexpected error — {e}")
        time.sleep(1)  # Avoid hammering 5Paisa API

    elapsed = datetime.now() - start
    print(f"\n{'='*60}")
    print(f"  Refinement complete: {success_count}/{len(stocks)} models updated")
    print(f"  Time taken: {elapsed}")
    print(f"{'='*60}\n")

    # Optional Telegram notification
    try:
        from rl_trading.utils.notifications import notifier
        notifier.send_message(
            f"🧠 *Nightly Model Refinement Complete*\n"
            f"{success_count}/{len(stocks)} models updated with latest {LOOKBACK_DAYS}-day data.\n"
            f"Time: {elapsed}"
        )
    except:
        pass


if __name__ == "__main__":
    main()
