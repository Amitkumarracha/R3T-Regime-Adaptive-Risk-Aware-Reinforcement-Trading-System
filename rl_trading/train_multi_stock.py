import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from rl_trading.universe.stocks import StockUniverse
from rl_trading.utils.features import add_technical_indicators, prepare_env_dataframe, FEATURE_COLS, train_test_split_timeseries
import yfinance as yf

def download_data(yahoo_ticker, interval="5m", period="60d"):
    try:
        df = yf.download(yahoo_ticker, interval=interval, period=period, progress=False)
        df.dropna(inplace=True)
        return df
    except Exception as e:
        print(f"Error downloading {yahoo_ticker}: {e}")
        import pandas as pd
        return pd.DataFrame()
from rl_trading.env.trading_env import TradingEnv
from rl_trading.agents.dqn_agent import DQNAgent

# ── Phase 2 Config ────────────────────────────────────────────
INTERVAL     = "5m"
PERIOD       = "60d" # Max for 5m via yfinance
N_EPISODES   = 40
INITIAL_CASH = 15000.0
PRETRAINED   = "rl_trading/models/dqn_india_v3.pt"
MODELS_DIR   = "rl_trading/models"

def universal_train_stock(ticker, yahoo_ticker, obs_dim):
    print(f"\n==================================================")
    print(f"Universal Phase 2 Training: {ticker} ({INTERVAL})")
    print(f"==================================================")
    
    agent = DQNAgent(obs_dim=obs_dim)
    
    # Check if we have a base model or a previous session model
    possible_models = [
        os.path.join(MODELS_DIR, f"dqn_{ticker}.pt"),
        PRETRAINED
    ]
    
    loaded = False
    for mp in possible_models:
        if os.path.exists(mp):
            try:
                agent.load(mp)
                print(f"Loaded existing model: {mp}")
                loaded = True
                break
            except: pass
    
    if not loaded:
        print("Starting from random initialization.")

    save_path = os.path.join(MODELS_DIR, f"dqn_{ticker}.pt")
    
    # Download data
    print(f"Downloading 60 days of 5m data for {yahoo_ticker}...")
    df = download_data(yahoo_ticker, interval=INTERVAL, period=PERIOD)
    
    if df.empty or len(df) < 500:
        print(f"Skipping {ticker} due to insufficient data ({len(df)} rows).")
        return
        
    df = add_technical_indicators(df)
    df, scaler = prepare_env_dataframe(df, FEATURE_COLS)
    train_df, test_df = train_test_split_timeseries(df)
    
    env = TradingEnv(train_df, feature_cols=FEATURE_COLS, initial_cash=INITIAL_CASH)
    
    portfolio_history = []
    
    for episode in range(1, N_EPISODES + 1):
        state, _ = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _, info = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            agent.learn()
            state = next_state
            total_reward += reward

        final_pv = info['portfolio_value']
        portfolio_history.append(final_pv)

        if episode % 10 == 0 or episode == 1:
            ret_pct = ((final_pv / INITIAL_CASH) - 1) * 100
            print(f"Ep {episode:2d}/{N_EPISODES} | Return: {ret_pct:+6.1f}% | PV: ₹{final_pv:,.2f} | ε: {agent.epsilon:.3f}")

    agent.save(save_path)
    
    # Save training plot
    os.makedirs(f"logs/training_plots", exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(portfolio_history)
    plt.axhline(INITIAL_CASH, color='red', linestyle='--')
    plt.title(f'Phase 2 - 5m Training - {ticker}')
    plt.savefig(f"logs/training_plots/{ticker}_5m.png")
    plt.close()
    
    print(f"Training complete for {ticker}. Saved to {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true", help="Skip stocks that already have a saved model file")
    args = parser.parse_args()

    os.makedirs(MODELS_DIR, exist_ok=True)
    universe = StockUniverse()
    stocks = universe.get_all_stocks()
    obs_dim = len(FEATURE_COLS) + 2
    
    if args.dry_run:
        stocks = stocks[:1]
        global N_EPISODES
        N_EPISODES = 1
        print("DRY RUN MODE")

    start_time = datetime.now()
    print(f"Starting Universal 5m Retraining at {start_time}")
    
    for stock in stocks:
        if args.skip_existing:
            save_path = os.path.join(MODELS_DIR, f"dqn_{stock['ticker']}.pt")
            if os.path.exists(save_path):
                print(f"Skipping {stock['ticker']}: Model already exists.")
                continue
        try:
            universal_train_stock(stock['ticker'], stock['yahoo'], obs_dim)
        except Exception as e:
            print(f"ERROR training {stock['ticker']}: {e}")
            
    end_time = datetime.now()
    print(f"Universal Retraining Finished. Total time: {end_time - start_time}")

if __name__ == "__main__":
    main()
